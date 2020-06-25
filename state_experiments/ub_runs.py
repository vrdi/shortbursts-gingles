import argparse
import geopandas as gpd
import numpy as np
import pickle
from functools import partial
from gerrychain import Graph, GeographicPartition, Partition, Election, accept
from gerrychain.updaters import Tally, cut_edges
from gerrychain import MarkovChain
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from gerrychain import constraints
from gerrychain.tree import recursive_tree_part
from gingleator import Gingleator
from little_helpers import *
import json

## Read in 
parser = argparse.ArgumentParser(description="UB Chain run", 
                                 prog="ub_runs.py")
parser.add_argument("state", metavar="state id", type=str,
                    choices=["VA", "TX", "AR", "CO", "LA", "NM"],
                    help="which state to run chains on")
parser.add_argument("iters", metavar="Length of runs", type=int,
                    help="how long to run each chain")
args = parser.parse_args()


num_h_districts = {"VA": 100, "TX": 150, "AR": 100, "CO": 65, "LA": 105, "NM": 70}


NUM_DISTRICTS = num_h_districts[args.state]
ITERS = args.iters
POP_COL = "TOTPOP"
N_SAMPS = 10
EPS = 0.045


## Setup graph, updaters, elections, and initial partition

print("Reading in Data/Graph")

graph = Graph.from_json("/cluster/tufts/mggg/jmatth03/shapes/BG_{}.json".format(args.state))


my_updaters = {"population" : Tally(POP_COL, alias="population"),
               "VAP": Tally("VAP"),
               "BVAP": Tally("BVAP"),
               "HVAP": Tally("HVAP"),
               "WVAP": Tally("WVAP"),
               "nWVAP": lambda p: {k: v - p["WVAP"][k] for k,v in p["VAP"].items()},
               "cut_edges": cut_edges}


print("Creating seed plan", flush=True)

total_pop = sum([graph.nodes()[n][POP_COL] for n in graph.nodes()])
ideal_pop = total_pop / NUM_DISTRICTS

seed_bal = {"AR": "05", "CO": "02", "LA": "04", "NM": "04", "TX": "02", "VA": "02"}

with open("seeds/{}_house_seed_{}.json".format(args.state, seed_bal[args.state]), "r") as f:
    cddict = json.load(f)

cddict = {int(k):v for k,v in cddict.items()}

init_partition = Partition(graph, assignment=cddict, updaters=my_updaters)


## Setup chain

proposal = partial(recom, pop_col=POP_COL, pop_target=ideal_pop, epsilon=EPS, 
                   node_repeats=1)

compactness_bound = constraints.UpperBound(lambda p: len(p["cut_edges"]), 
                                             2*len(init_partition["cut_edges"]))

chain = MarkovChain(
        proposal,
        constraints=[
            constraints.within_percent_of_ideal_population(init_partition, EPS),
            compactness_bound],
        accept=accept.always_accept,
        initial_state=init_partition,
        total_steps=ITERS)


## Run chain

print("Starting Markov Chain runs")

for n in range(N_SAMPS):
    print("\tStarting chain {}".format(n), flush=True)
    chain_results = {"cutedges": np.zeros(ITERS),
                     "BVAP": np.zeros((ITERS, NUM_DISTRICTS)),
                     "HVAP": np.zeros((ITERS, NUM_DISTRICTS)),
                     "WVAP": np.zeros((ITERS, NUM_DISTRICTS))}

    output = "/cluster/tufts/mggg/jmatth03/unbiased_LA/{}_dists{}_{:.1%}_{}_unbiased_{}.p".format(args.state,
                                                        NUM_DISTRICTS, EPS, 
                                                        ITERS, n)


    for i, part in enumerate(chain):
        chain_results["cutedges"][i] = len(part["cut_edges"])
        chain_results["BVAP"][i] = sorted([v / part["VAP"][k] for k,v in part["BVAP"].items()])
        chain_results["HVAP"][i] = sorted([v / part["VAP"][k] for k,v in part["HVAP"].items()])
        chain_results["WVAP"][i] = sorted([v / part["VAP"][k] for k,v in part["WVAP"].items()])
        
        if i % 1000 == 0:
            print("*", end="", flush=True)

    with open(output, "wb") as f_out:
        pickle.dump(chain_results, f_out)

    print()
