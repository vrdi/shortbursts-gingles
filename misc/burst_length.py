import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.tree import recursive_tree_part
from gerrychain.metrics import mean_median, efficiency_gap, polsby_popper, partisan_gini
from functools import (partial, reduce)
import pandas
import geopandas as gp
import numpy as np
import networkx as nx
import pickle
import seaborn as sns
import pprint
import operator
import scipy
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize
import random
from nltk.util import bigrams
from nltk.probability import FreqDist
from gingleator import Gingleator



print("Setting up dual graph")
graph_PA = pickle.load(open("PA_graph.p", "rb"))
df_PA = pickle.load(open("PA_df.p", "rb"))

PA_updaters = {"population": updaters.Tally("TOT_POP", alias="population"),
               "bvap": updaters.Tally("BLACK_POP", alias="bvap"),
               "vap": updaters.Tally("VAP", alias="vap"),
               "bvap_prec": lambda part: {k: part["bvap"][k] / part["population"][k] for k in part["bvap"]}}

enacted_senate = GeographicPartition(graph_PA, assignment="SSD", 
                                     updaters=PA_updaters)


g = Gingleator(enacted_senate, pop_col="TOT_POP", minority_prec_col="bvap_prec",
               epsilon=0.1)

print("Plotting")
plt.figure(figsize=(10,8))
plt.xlim(-.5, 8)
plt.xlabel("Number of opportunity districts")
plt.ylabel("Steps")
plt.title("PA short bursts of different lengths")

total_steps = 1000

for color, len_burst in [("k", 1), ("b", 5), ("r", 25), ("g", 50), ("y", 100),
                         ("cyan", 1000)]:
    print(len_burst)
    num_bursts =  int(total_steps / len_burst)
    _, observations = g.short_burst_run(num_bursts, len_burst)

    for i in range(num_bursts):
        plt.plot(observations[i], range(len_burst*i, len_burst*(i+1)),
                 color=color, alpha=0.5, marker=".", markevery=[0,-1])
    plt.plot([], color=color, label=("Burst_len " + str(len_burst)))
    
plt.legend()
plt.show()