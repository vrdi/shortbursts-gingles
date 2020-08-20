import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
from gerrychain import Graph, GeographicPartition, Partition, Election, accept
from gerrychain.updaters import Tally, cut_edges
import glob
import functools
import operator
import argparse

parser = argparse.ArgumentParser(description="Plot State Run Results", 
                                 prog="plot_state_results.py")
parser.add_argument("state", metavar="state id", type=str,
                    choices=["VA", "TX", "CO", "LA", "NM"],
                    help="which state to run chains on")
parser.add_argument("race_col", metavar="Race_column", type=str,
                    choices=["BVAP", "HVAP", "nWVAP"],
                    help="Which race_col")
args = parser.parse_args()


## Configure State parameters
num_h_districts = {"VA": 100, "TX": 150, "AR": 100, "CO": 65, "LA": 105, "NM": 70} ## Number of state House districts
pop_bal = {"CO": 2.0, "LA": 4.5, "NM": 4.5, "TX": 2.0, "VA": 2.0}                  ## Population balance of runs

ST = args.state
NUM_DISTRICTS = num_h_districts[ST]
MIN_COL = args.race_col
iters = 100000

def foldl(func, acc, xs):
  return functools.reduce(func, xs, acc)

foldr = lambda func, acc, xs: functools.reduce(lambda x, y: func(y, x), xs[::-1], acc)

def get_state_runs(state, seats, iters=100000, pop_bal=2.0, min_col="nWVAP",
                   ls=[2,5,10,20,40,80], ps=[0.25, 0.125, 0.0625]):
    """ Reads in short burst and biased run results. and returns dictionary of parameter and matrix of run results.
        Args:
          * state: str      -- abbreviation of state to pull data from
          * seats: int      -- number of state House districts
          * iters: int      -- number of steps (observed plans) in chain
          * pop_bal: float  -- population balance of plans in chain
          * min_col: str    -- column name of population to look at majority-minority districts for
          * ls: int list    -- list of the burst lengths for the short burst runs
          * ps: float list  -- list of acceptance probabilities of "worse preforming" plans for the biased runs
    """
    results = {}

    for l in ls:
        sb_runs = glob.glob("data/states/{}_dists{}_{}opt_{}%_100000_sbl{}_score0_*.npy".format(state, seats, 
                                                                                                min_col, pop_bal, l))
        results[str(l)] = np.zeros((len(sb_runs), iters))
        for i, run in enumerate(sb_runs):
            results[str(l)][i] = np.load(run).flatten()
    for p in ps:
        tilt_runs = glob.glob("data/states/{}_dists{}_{}opt_{}%_100000_p{}_*.npy".format(state, seats, 
                                                                                         min_col, pop_bal, p))
        results[str(p)] = np.zeros((len(tilt_runs), iters))
        for i, run in enumerate(tilt_runs):
            results[str(p)][i] = np.load(run).flatten()
    return results


def create_state_df(runs, iters=100000):
  """ Create and return a dataframe from the run dictionary
      Args:
        * runs: dict -- dictionary of run results
        * iters: int -- number of steps (observed plans) in chain
  """
    df_st = pd.DataFrame()
    for l in runs.keys():
        for i in range(runs[l].shape[0]):
            df = pd.DataFrame()
            df["Step"] = np.arange(iters)
            df["Maximum"] = np.maximum.accumulate(runs[l][i].flatten())
            df["run-type"] = "Short Burst" if float(l) > 1 else "Biased Run" if float(l) < 1 else "Unbiased Run"
            df["param"] = "b = {}".format(l) if float(l) > 1 else "q = {}".format(l)
            df_st = df_st.append(df, ignore_index=True)
    return df_st


## gather biased and short burst runs
state_runs = get_state_runs(ST, NUM_DISTRICTS, ls=[2,5,10,25,50,100,200], 
                              pop_bal=pop_bal[ST], min_col=MIN_COL)

## add unbiased results to state_runs
ubs = glob.glob("data/sample_unbiased/{}_dists{}_{}%_100000_unbiased_*.npy".format(ST, NUM_DISTRICTS, pop_bal[ST]))
ub_runs = {}
for i, run in enumerate(ubs):
     with open(run, "rb") as f:
        ub_runs[i] = pickle.load(f)

runs = list(ub_runs.values())
num_ub_runs = len(runs)


if MIN_COL == 'nWVAP':
    min_col_vap = foldl(lambda x,y: np.concatenate((x,y["WVAP"]), axis=0), runs[0]["WVAP"], runs[1:])
    state_runs['1'] = (min_col_vap.reshape((num_ub_runs, iters, NUM_DISTRICTS)) < 0.5).sum(axis=2)
else:
    min_col_vap = foldl(lambda x,y: np.concatenate((x,y[MIN_COL]), axis=0), runs[0][MIN_COL], runs[1:])
    state_runs['1'] = (min_col_vap.reshape((num_ub_runs, iters, NUM_DISTRICTS)) > 0.5).sum(axis=2)


df_state = create_state_df(state_runs)


## Plot runs
cmap_no_light = sns.color_palette(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', 
                                   '#f58231', '#911eb4', '#46f0f0', '#f032e6', 
                                   '#808000', '#008080', '#9a6324', '#800000', 
                                   '#aaffc3', '#000075'], n_colors=len(df_state.param.unique()))


plt.figure(figsize=(12,8))

plt.title("{} State House ({} seats)".format(ST, NUM_DISTRICTS),fontsize=14)

sns.lineplot(x="Step", y="Maximum", hue="param",style="run-type", palette=cmap_no_light,
             data=df_state, ci="sd", estimator='mean', alpha=0.75)

# enacted = 28 # check number
# plt.axhline(enacted, label="Enacted Plan", c="k", linestyle='dashdot')

plt.ylabel("Expected Maximum number of {} gingles districts".format(MIN_COL),fontsize=12)
plt.xlabel("Step",fontsize=12)
plt.legend()
plt.savefig("plots/{}_maxes_all_{}.png".format(ST, MIN_COL), dpi=200, bbox_inches='tight')
plt.close()

