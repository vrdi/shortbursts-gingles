from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom, propose_random_flip
from functools import (partial, reduce)
import numpy as np
import random


def config_markov_chain(initial_part, iters=1000, epsilon=0.05, 
                        compactness=True, pop="TOT_POP", accept_func=None):
    ideal_population = sum(initial_part["population"].values()) / len(initial_part)

    proposal = partial(recom,
                       pop_col=pop,
                       pop_target=ideal_population,
                       epsilon=epsilon,
                       node_repeats=1)

    if compactness:
        compactness_bound = constraints.UpperBound(lambda p: len(p["cut_edges"]),
                            2*len(initial_part["cut_edges"]))
        cs = [constraints.within_percent_of_ideal_population(initial_part, epsilon),
              compactness_bound]
    else:
        cs = [constraints.within_percent_of_ideal_population(initial_part, epsilon)]


    if accept_func == None: accept_func = accept.always_accept

    return MarkovChain(proposal=proposal, constraints=cs,
                       accept=accept_func, initial_state=initial_part,
                       total_steps=iters)



class Gingleator:
    """
    Gingleator class

    This class represents a set of methods used to find plans with greater numbers
    of gingles districts.
    """

    def __init__(self, initial_partition, threshold=0.4, 
                 score_funct=None, minority_perc_col=None,
                 pop_col="TOTPOP", epsilon=0.05):
        self.part = initial_partition
        self.threshold = threshold
        self.score = self.num_opportunity_dists if score_funct == None else score_funct
        self.minority_prec = minority_prec_col
        self.pop_col = pop_col
        self.epsilon = epsilon


    def init_minority_perc_col(self, minority_pop_col, total_pop_col,
                               minority_perc_col):
        """
         init_minority_prec_col takes the string corresponding to the minority
         population column and the total population column attributes in the 
         partition updaters as well as the desired name of the minority percent 
         column and updates the partition updaters accordingly
        """
        prec_up = {minority_perc_col:
                   lambda part: {k: part[minority_pop_col][k] / part[total_pop_col][k]
                                 for k in part.parts.keys()}}
        self.part.updaters.update(prec_up)


    """
    Types of Markov Chains:
    The following methods are different strategies for searching for the maximal
    number of Gingles districts
    """

    def short_burst_run(self, num_bursts, num_steps, verbose=False,
                        maximize=True): #checkpoint_file=None):
        max_part = (self.part, self.score(self.part, self.minority_prec,
                    self.threshold)) 
        """
        short_burst_run: preforms a short burst run using the instance's score function.
                         Each burst starts at the best preforming plan of the previous
                         burst.  If there's a tie, the later observed one is selected.
        args:
            num_steps:  how many steps to run an unbiased markov chain for during each burst
            num_bursts: how many bursts to preform
            verbose:    flag - indicates whether to prints the burst number at the beginning of 
                               each burst
            maximize:   flag - indicates where to prefer plans with higher or lower scores.
        """
        observed_num_ops = np.zeros((num_bursts, num_steps))

        for i in range(num_bursts):
            if verbose: print("*", end="", flush=True)
            chain = config_markov_chain(max_part[0], iters=num_steps,
                                        epsilon=self.epsilon, pop=self.pop_col)

            for j, part in enumerate(chain):
                part_score = self.score(part, self.minority_prec, self.threshold)
                observed_num_ops[i][j] = part_score
                if maximize:
                    max_part = (part, part_score) if part_score >= max_part[1] else max_part
                else:
                    max_part = (part, part_score) if part_score <= max_part[1] else max_part

        return (max_part, observed_num_ops)


    def variable_len_short_burst(self, num_iters, stuck_buffer=10,
                                 maximize=True, verbose=False):
        """
        variable_len_short_burst: preforms a variable length short burst run using the instance's 
                                  score function. Each burst starts at the best preforming plan of 
                                  the previous burst.  If there's a tie, the later observed one is 
                                  selected.
        args:
            num_iters:      the total number of steps to take (aka plans to sample)
            stuck_buffer:   Factor specifying how long to tolerate no improvement, before increasing
                            the burst length.
            verbose:        flag - indicates whether to prints the burst number at the beginning 
                                    of each burst
            maximize:       flag - indicates where to prefer plans with higher or lower scores.
        """
        max_part = (self.part, self.score(self.part, self.minority_prec,
                        self.threshold))
        observed_num_ops = np.zeros(num_iters)
        time_stuck = 0
        burst_len = 2
        i = 0

        while(i < num_iters):
            if verbose: print("*", end="", flush=True)
            chain = config_markov_chain(max_part[0], iters=burst_len,
                                        epsilon=self.epsilon, pop=self.pop_col)
            for j, part in enumerate(chain):
                part_score = self.score(part, self.minority_prec, self.threshold)
                observed_num_ops[i] = part_score

                if part_score <= max_part[1]: time_stuck += 1
                else: time_stuck = 0

                if maximize:
                    max_part = (part, part_score) if part_score >= max_part[1] else max_part
                else:
                    max_part = (part, part_score) if part_score <= max_part[1] else max_part
                
                i += 1
                if i >= num_iters: break
            if time_stuck >= stuck_buffer*burst_len : burst_len *= 2

        return (max_part, observed_num_ops)


    def biased_run(self, num_iters, p=0.25, maximize=True, verbose=False):
        """
        biased_run: preforms a biased (or tilted) run using the instance's score function.  The
                    chain always accepts a new proposal with the same or a better score and accepts
                    proposals with a worse score with some probability.
        args:
            num_iters:  total number of steps to take (aka plans to sample)
            p:          probability of a plan with a worse preforming score
            verbose:    flag - indicates whether to prints the burst number at the beginning 
                                    of each burst
            maximize:   flag - indicates where to prefer plans with higher or lower scores.
        """
        max_part = (self.part, self.score(self.part, self.minority_prec,
                    self.threshold))
        observed_num_ops = np.zeros(num_iters)
        
        def biased_acceptance_function(part):
            if part.parent == None: return True
            part_score = self.score(part, self.minority_prec, self.threshold)
            prev_score = self.score(part.parent, self.minority_prec, self.threshold)
            if maximize and part_score >= prev_score: return True
            elif not maximize and part_score <= prev_score: return True
            else: return random.random() < p

        chain = config_markov_chain(self.part, iters=num_iters,
                                    epsilon=self.epsilon, pop=self.pop_col,
                                    accept_func= biased_acceptance_function)
        for i, part in enumerate(chain):
            if verbose and i % 100 == 0: print("*", end="", flush=True)
            part_score = self.score(part, self.minority_prec, self.threshold)
            observed_num_ops[i] = part_score
            if maximize:
                max_part = (part, part_score) if part_score >= max_part[1] else max_part
            else:
                max_part = (part, part_score) if part_score <= max_part[1] else max_part

        return (max_part, observed_num_ops)


    def biased_short_burst_run(self, num_bursts, num_steps, p=0.25, 
                              verbose=False, maximize=True):
        """
        biased_short_burst_run: preforms a biased short burst run using the instance's score function.
                                Each burst is a biased run markov chain, starting at the best preforming 
                                plan of the previous burst.  If there's a tie, the later observed 
                                one is selected.
        args:
            num_steps:  how many steps to run an unbiased markov chain for during each burst
            num_bursts: how many bursts to preform
            p:          probability of a plan with a worse preforming score, within a burst
            verbose:    flag - indicates whether to prints the burst number at the beginning of 
                               each burst
            maximize:   flag - indicates where to prefer plans with higher or lower scores.
        """
        max_part = (self.part, self.score(self.part, self.minority_prec,
                    self.threshold)) 
        observed_num_ops = np.zeros((num_bursts, num_steps))

        def biased_acceptance_function(part):
            if part.parent == None: return True
            part_score = self.score(part, self.minority_prec, self.threshold)
            prev_score = self.score(part.parent, self.minority_prec, self.threshold)
            if maximize and part_score >= prev_score: return True
            elif not maximize and part_score <= prev_score: return True
            else: return random.random() < p

        for i in range(num_bursts):
            if verbose: print("Burst:", i)
            chain = config_markov_chain(max_part[0], iters=num_steps,
                                        epsilon=self.epsilon, pop=self.pop_col,
                                        accept_func= biased_acceptance_function)

            for j, part in enumerate(chain):
                part_score = self.score(part, self.minority_prec, self.threshold)
                observed_num_ops[i][j] = part_score
                if maximize:
                    max_part = (part, part_score) if part_score >= max_part[1] else max_part
                else:
                    max_part = (part, part_score) if part_score <= max_part[1] else max_part
    
        return (max_part, observed_num_ops)

    """
    Score Functions
    """

    @classmethod
    def num_opportunity_dists(cls, part, minority_perc, threshold):
        """
        num_opportunity_dists: given a partition, name of the minority percent updater, and a
                               threshold, returns the number of opportunity districts.
        """
        dist_precs = part[minority_prec].values()
        return sum(list(map(lambda v: v >= threshold, dist_precs)))


    @classmethod
    def reward_partial_dist(cls, part, minority_perc, threshold):
        """
        reward_partial_dist: given a partition, name of the minority percent updater, and a
                             threshold, returns the number of opportunity districts + the 
                             percentage of the next highest district.
        """
        dist_precs = part[minority_prec].values()
        num_opport_dists = sum(list(map(lambda v: v >= threshold, dist_precs)))
        next_dist = max(i for i in dist_precs if i < 0.4)
        return num_opport_dists + next_dist


    @classmethod
    def reward_next_highest_close(cls, part, minority_perc, threshold):
        """
        reward_next_highest_close: given a partition, name of the minority percent updater, and a
                                   threshold, returns the number of opportunity districts, if no 
                                   additional district is within 10% of reaching the threshold.  If one is, 
                                   the distance that district is from the threshold is scaled between 0 
                                   and 1 and added to the count of opportunity districts.
        """
        dist_precs = part[minority_prec].values()
        num_opport_dists = sum(list(map(lambda v: v >= threshold, dist_precs)))
        next_dist = max(i for i in dist_precs if i < threshold)

        if next_dist < threshold - 0.1:
            return num_opport_dists
        else: 
            return num_opport_dists + (next_dist - threshold + 0.1)*10


    @classmethod
    def penalize_maximum_over(cls, part, minority_perc, threshold):
        """
        penalize_maximum_over: given a partition, name of the minority percent updater, and a
                               threshold, returns the number of opportunity districts + 
                               (1 - the maximum excess) scaled to between 0 and 1.
        """
        dist_precs = part[minority_prec].values()
        num_opportunity_dists = sum(list(map(lambda v: v >= threshold, dist_precs)))
        if num_opportunity_dists == 0:
            return 0
        else:
            max_dist = max(dist_precs)
            return num_opportunity_dists + (1 - max_dist)/(1 - threshold)


    @classmethod
    def penalize_avg_over(cls, part, minority_perc, threshold):
        """
        penalize_maximum_over: given a partition, name of the minority percent updater, and a
                               threshold, returns the number of opportunity districts + 
                               (1 - the average excess) scaled to between 0 and 1.
        """
        dist_precs = part[minority_prec].values()
        opport_dists = list(filter(lambda v: v >= threshold, dist_precs))
        if opport_dists == []:
            return 0
        else:
            num_opportunity_dists = len(opport_dists)
            avg_opportunity_dist = np.mean(opport_dists)
            return num_opportunity_dists + (1 - avg_opportunity_dist)/(1 - threshold)
