from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom, propose_random_flip
from functools import (partial, reduce)
import numpy as np


def config_markov_chain(initial_part, iters=1000, epsilon=0.05, 
                        compactness=True, pop="TOT_POP"):
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

    return MarkovChain(proposal=proposal, constraints=cs,
                       accept=accept.always_accept, initial_state=initial_part,
                       total_steps=iters)


"""
Score Functions
"""



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


    def short_burst_run(self, num_bursts, num_steps):
        max_part = self.part
        observed_num_ops = np.zeros((num_bursts, num_steps))

        for i in range(num_bursts):
            chain = config_markov_chain(max_part, iters=num_steps,
                                        epsilon=self.epsilon, pop=self.pop_col)
            burst_parts = []

            for part in chain:
                score = self.score(part, self.minority_prec, self.threshold)
                burst_parts.append((part, score))

            parts, num_ops = list(zip(*burst_parts))

            observed_num_ops[i] = num_ops
            max_i = np.argmax(num_ops)
            max_part = parts[max_i]
    
        return (max_part, observed_num_ops)


    ## For the passed partition num_opportunity_dists returns the number of
    ## opportunity districts in the passed partitions.
    @classmethod
    def num_opportunity_dists(cls, part, minority_prec, threshold):
        dist_precs = part[minority_prec].values()
        return sum(list(map(lambda v: v >= threshold, dist_precs)))