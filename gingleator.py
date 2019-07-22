from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom, propose_random_flip
from functools import (partial, reduce)
import numpy as np
import random


def config_markov_chain(initial_part, iters=1000, epsilon=0.05, 
                        compactness=True, pop="TOT_POP", accept_func = None):
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



"""
Gingleator class

This class represents a set of methods used to find plans with greater numbers
of gingles districts.
"""
class Gingleator:

    def __init__(self, initial_partition, threshold=0.4, 
                 score_funct=None, minority_prec_col=None,
                 pop_col="TOTPOP", epsilon=0.05):
        self.part = initial_partition
        self.threshold = threshold
        self.score = self.num_opportunity_dists if score_funct == None else score_funct
        self.minority_prec = minority_prec_col
        self.pop_col = pop_col
        self.epsilon = epsilon

    ## init_minority_prec_col takes the string corresponding to the minority
    ## population column and the total population column attributes in the 
    ## partition updaters as well as the desired name of the minority percent 
    ## column and updates the partition updaters accordingly
    def init_minority_prec_col(self, minority_pop_col, total_pop_col,
                               minority_prec_col):
        prec_up = {minority_prec_col:
                   lambda part: {k: part[minority_pop_col][k] / part[total_pop_col][k]
                                 for k in part.parts.keys()}}
        self.part.updaters.update(prec_up)


    """
    Types of Markov Chains:
    The following methods are different strategies for searching for the maximal
    number of Gingles districts
    """

    ## short_burst_run looks for districting plans that preform well via the 
    ## short burst method: run the markov chain for num_step take the plan that
    ## preforms the highest by the instances score functions and repeat starting
    ## from that plan for num_bursts iterations. The verbose flag, prints the
    ## burst number at the beginning of each burst and the maximize flag
    ## indicates whether or not to favour plans with higher of lower scores.
    def short_burst_run(self, num_bursts, num_steps, verbose=False,
                        maximize=True):
        max_part = (self.part, self.score(self.part, self.minority_prec,
                    self.threshold)) 
        observed_num_ops = np.zeros((num_bursts, num_steps))

        for i in range(num_bursts):
            if verbose: print("Burst:", i)
            chain = config_markov_chain(max_part[0], iters=num_steps,
                                        epsilon=self.epsilon, pop=self.pop_col)

            for j, part in enumerate(chain):
                part_score = self.score(part, self.minority_prec, self.threshold)
                observed_num_ops[i][j] = part_score
                if maximize:
                    max_part = (part, part_score) if part_score >= max_part[1] else max_part
                else:
                    max_part = (part, part_score) if part_score <= max_part[1] else max_part
    
        return (max_part[0], observed_num_ops)


    ## biased_run runs a markov chain for num_iters, where the chain always
    ## accepts the new proposal if it preforms the same or better on the score
    ## and accept the new proposal with probability p if it preforms worse on
    ## the score.  The maximize flag indicate whether to favour plans with
    ## higher or lower scores.
    def biased_run(self, num_iters, p=0.25, maximize=True):
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
            part_score = self.score(part, self.minority_prec, self.threshold)
            observed_num_ops[i] = part_score
            if maximize:
                max_part = (part, part_score) if part_score >= max_part[1] else max_part
            else:
                max_part = (part, part_score) if part_score <= max_part[1] else max_part

        return (max_part[0], observed_num_ops)


    def biased_short_burst_run(self, num_bursts, num_steps, p=0.25, 
                              verbose=False, maximize=True):
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
    
        return (max_part[0], observed_num_ops)

    """
    Score Functions
    """

    ## For the passed partition num_opportunity_dists returns the number of
    ## opportunity districts in the passed partitions.
    @classmethod
    def num_opportunity_dists(cls, part, minority_prec, threshold):
        dist_precs = part[minority_prec].values()
        return sum(list(map(lambda v: v >= threshold, dist_precs)))


    ## For the passed partition reward_partial_dist returns the number of
    ## opportunity districts in the passed partitions + the percentage of the
    ## next largest district.
    @classmethod
    def reward_partial_dist(cls, part, minority_prec, threshold):
        dist_precs = part[minority_prec].values()
        num_opport_dists = sum(list(map(lambda v: v >= threshold, dist_precs)))
        next_dist = max(i for i in dist_precs if i < 0.4)
        return num_opport_dists + next_dist


    ## For the passed partition reward_next_highest_close returns the number of
    ## opportunity districts in the passed partitions and adds the next largest
    ## difference scaled if it is within 0.1 of the threshold.
    @classmethod
    def reward_next_highest_close(cls, part, minority_prec, threshold):
        dist_precs = part[minority_prec].values()
        num_opport_dists = sum(list(map(lambda v: v >= threshold, dist_precs)))
        next_dist = max(i for i in dist_precs if i < threshold)

        if next_dist < threshold - 0.1:
            return num_opport_dists
        else: 
            return num_opport_dists + (next_dist - threshold + 0.1)*10


    ## For the passed partition penalize_maximum_over returns the number of
    ## opportunity districts + (1 - the maximum excess) scaled to between
    ## zero and one.
    @classmethod
    def penalize_maximum_over(cls, part, minority_prec, threshold):
        dist_precs = part[minority_prec].values()
        num_opportunity_dists = sum(list(map(lambda v: v >= threshold, dist_precs)))
        if num_opportunity_dists == 0:
            return 0
        else:
            max_dist = max(dist_precs)
            return num_opportunity_dists + (1 - max_dist)/(1 - threshold)


    ## For the passed partition penalize_avg_over returns the number of
    ## opportunity districts + (1 - the average excess) scaled to between
    ## zero and one.
    @classmethod
    def penalize_avg_over(cls, part, minority_prec, threshold):
        dist_precs = part[minority_prec].values()
        opport_dists = list(filter(lambda v: v >= threshold, dist_precs))
        if opport_dists == []:
            return 0
        else:
            num_opportunity_dists = len(opport_dists)
            avg_opportunity_dist = np.mean(opport_dists)
            return num_opportunity_dists + (1 - avg_opportunity_dist)/(1 - threshold)
