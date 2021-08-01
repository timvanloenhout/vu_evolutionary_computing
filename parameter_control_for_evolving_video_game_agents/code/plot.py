import sys, os
sys.path.insert(0, 'evoman')

from environment import Environment
from demo_controller import player_controller

from deap import tools
from deap import base, creator
import array
import random

import numpy as np
import pickle
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

N_POP = 100
N_PROCESSES = 10
N_GENERATIONS = 100 + 1
N_BATTLES = 5

GROUPS = ["Group 1", "Group 2"]
ENEMY_MAP = {"Group 1": [1, 2, 5], "Group 2": [4, 6, 7]}
RESULTS_DIR = "./task2/plots/"



sns.set_theme(style="darkgrid")


def generate_ind(icls, scls, size, ind_min, ind_max, mut_init):
	ind = icls(random.uniform(ind_min, ind_max) for _ in range(size))
	ind.strategy = scls(mut_init for _ in range(size))
	# ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
	return ind


def init_DEAP():
	toolbox = base.Toolbox()
	n_weights = 265
	ind_min = -1  # mimimum weight value
	ind_max = 1  # maximum weight value
	mut_init = 1  # initial mutation rate for every weight
	mut_lb = 0.005  # lower bound on the mutation rates

	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
	creator.create("Strategy", array.array, typecode="d")  # array containing the mutation sigma's

	toolbox.register("individual", generate_ind, creator.Individual, creator.Strategy,
					 n_weights, ind_min, ind_max, mut_init)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	return toolbox


def get_fitness_data(enemy, mut_type):
	means = np.zeros((N_GENERATIONS, N_PROCESSES))
	maxs = np.zeros((N_GENERATIONS, N_PROCESSES))

	if mut_type == "SELF":
		directory = f"./task2/enemy_{enemy}_dynamic/"
	else:
		directory = f"./task2/enemy_{enemy}_time_based/"

	for j in range(N_GENERATIONS):
		for i in range(N_PROCESSES):
			stats = pickle.load(open(f"{directory}rep_{i + 1}/stats_{j}.pkl", "rb"))
			means[j, i] = stats["mean"]
			maxs[j, i] = stats["max"]


	df_means = pd.DataFrame(means, columns=[f"Mean {mut_type}" for i in range(N_PROCESSES)])
	df_maxs = pd.DataFrame(maxs, columns=[f"Max {mut_type}" for i in range(N_PROCESSES)])

	return df_means, df_maxs


def get_gain_data(enemy, mut_type):
	gains = np.zeros((N_PROCESSES, 1))

	if mut_type == "SELF":
		directory = f"task2/enemy_{enemy}_dynamic/"
	else:
		directory = f"task2/enemy_{enemy}_time_based/"

	for i in range(N_PROCESSES):
		for j in range(N_BATTLES):
			filename = f"{directory}rep_{i + 1}/Champion_scores_{j + 1}_all.pkl"

			with open(filename, "rb") as f:
				scores = pickle.load(f)
			
			for ind_enemy in range(1, 9):
				_, phealth, ehealth, _ = scores[ind_enemy]
				gains[i,0] += phealth - ehealth

	return gains


def get_genetic_data(enemy, mut_type, group_name):
	dist_measure = np.zeros((N_GENERATIONS, N_PROCESSES))

	if mut_type == "SELF":
		directory = f"task2/enemy_{enemy}_dynamic/"
	else:
		directory = f"task2/enemy_{enemy}_time_based/"

	for j in range(N_GENERATIONS):
		for i in range(N_PROCESSES):
			with open(f"{directory}rep_{i + 1}/whole_pop_{j}.pkl", "rb") as f:
				pop = pickle.load(f)

			pop_array = np.array(pop).T
			pop_mean = np.expand_dims(np.mean(pop, axis=0), 1)

			diff = (pop_array - pop_mean).T
			dist_measure[j, i] = np.mean(np.linalg.norm(diff, ord=1, axis=1))


	df_dist_measure = pd.DataFrame(dist_measure, columns=[f"{group_name} {mut_type}" for i in range(N_PROCESSES)])
	
	return df_dist_measure
	

def get_sigma_data(enemy, mut_type, group_name):
	data = np.zeros((N_GENERATIONS, N_PROCESSES))

	
	if mut_type == "SELF":
		directory = f"task2/enemy_{enemy}_dynamic/"
	else:
		directory = f"task2/enemy_{enemy}_time_based/"

	for j in range(N_GENERATIONS):
		if mut_type == "DET":
			data[j, :] = 1 - 0.9 * ((j+1)/N_GENERATIONS)
		else:
			for i in range(N_PROCESSES):
				with open(f"{directory}rep_{i + 1}/whole_pop_{j}.pkl", "rb") as f:
					pop = pickle.load(f)
				
				pop_strategy = np.array([pop[k].strategy for k in range(N_POP)]).squeeze()

				data[j, i] =  np.mean(pop_strategy)


	df_data = pd.DataFrame(data, columns=[f"{group_name} {mut_type}" for i in range(N_PROCESSES)])

	return df_data
	

def plot_lines():
	for group in GROUPS:
		enemy = ENEMY_MAP[group]

		df_means_SELF, df_maxs_SELF = get_fitness_data(enemy, "SELF")
		df_means_DET, df_maxs_DET = get_fitness_data(enemy, "DET")
		
		data = [df_means_SELF, df_maxs_SELF, df_means_DET, df_maxs_DET]
		data = pd.concat(data, 1)

		sns_plot = sns.lineplot(data=data, color=sns.color_palette())

		leg_lines = sns_plot.legend().get_lines()
		for i in range(4):
			if i % 2 == 0:
				sns_plot.lines[i].set_linestyle(":")
				leg_lines[i].set_linestyle(':')
			else:
				sns_plot.lines[i].set_linestyle("-")
				leg_lines[i].set_linestyle('-')

		sns_plot.set_title(f"Population- and Max-Fitness with {group}")
		sns_plot.set_xlabel("Generations")
		sns_plot.set_ylabel("Fitness Value")
		plt.savefig(f"{RESULTS_DIR}/lineplot_{group}.pdf", bbox_inches = 'tight', pad_inches = 0)

		plt.show()
		plt.clf()


def plot_boxes():
	for group in GROUPS:
		enemy = ENEMY_MAP[group]
		
		gains_SELF = get_gain_data(enemy, mut_type="SELF")
		gains_DET = get_gain_data(enemy, mut_type="DET")

		gains_SELF[:, 0] /= 5
		gains_DET[:, 0] /= 5

		df_means_SELF = pd.DataFrame(gains_SELF, columns=["SELF"])
		df_means_DET = pd.DataFrame(gains_DET, columns=["DET"])

		print(f"enemy: {enemy}, SELF mean: {np.mean(gains_SELF[:,0]):.3f}, DET mean: {np.mean(gains_DET[:,0]):.3f}")

		tval, pval = ttest_ind(gains_SELF, gains_DET, equal_var=False)

		data = [df_means_SELF, df_means_DET]
		data = pd.concat(data, 1)
		custom_dims = (3, 5)
		_, ax = plt.subplots(figsize=custom_dims)
		ax.set(ylim=(-500, 100))

		sns_plot = sns.boxplot(ax=ax, data=data, width=0.5, fliersize=0)

		# Add individual points
		swarm = pd.DataFrame(columns=['fitness', 'experiment'])

		for gain in gains_SELF:
			swarm = swarm.append({'fitness': gain[0], '': 'SELF'}, ignore_index=True)

		for gain in gains_DET:
			swarm = swarm.append({'fitness': gain[0], '': 'DET'}, ignore_index=True)
			
		sns_plot = sns.swarmplot(x="", y="fitness", data=swarm, color=".25")

		sns_plot.set_title(f"Champions with {group}\n t-statistic={tval[0]:.3f} p-value={pval[0]:.3f}")
		sns_plot.set_ylabel("Average Individual Gain Against All 8 Enemies")
		plt.tight_layout()
		plt.savefig(f"{RESULTS_DIR}/boxplot_{group}.pdf", bbox_inches = 'tight', pad_inches = 0)
		
		plt.show()
		plt.clf()	


def plot_diversity(plot_data, **kwargs):
	data = pd.DataFrame()
	for group in GROUPS:
		enemy = ENEMY_MAP[group]
		df_SELF = plot_data["func"](enemy, "SELF", group, **kwargs)
		df_DET= plot_data["func"](enemy, "DET", group, **kwargs)

		inter_data = [df_SELF, df_DET, data]
		data = pd.concat(inter_data, 1)

	sns_plot = sns.lineplot(data=data, color=sns.color_palette(),)
	
	leg_lines = sns_plot.legend().get_lines()
	for i in range(4):
			if i % 2 == 0:
				sns_plot.lines[i].set_linestyle(":")
				leg_lines[i].set_linestyle(':')
			else:
				sns_plot.lines[i].set_linestyle("-")
				leg_lines[i].set_linestyle('-')

	sns_plot.set_title(plot_data["title"])
	sns_plot.set_xlabel("Generations")
	sns_plot.set_ylabel(plot_data["ylabel"])
	plt.savefig(f"{RESULTS_DIR}/{plot_data[0]}.pdf", bbox_inches = 'tight', pad_inches = 0)
	plt.show()
	plt.clf()


def plot(what):
	if not os.path.exists(RESULTS_DIR):
		os.makedirs(RESULTS_DIR)

	if what == "fitness":
		plot_lines()

	elif what == "genetic_diversity":
		plot_data = {
			0: "manhattan_genetic_diversity",
			"title": "Manhattan Measure of Genetic Diversity",
			"ylabel": "Manhattan Distance",
			"func": get_genetic_data
		}
		tb = init_DEAP()
		plot_diversity(plot_data)

	elif what == "sigma":
		plot_data = {
			0: "sigma",
			"title": "Population Average of Standard Deviation of Gaussian Mutation Operator",
			"ylabel": "Standard Deviation",
			"func": get_sigma_data
		}
		tb = init_DEAP()
		plot_diversity(plot_data)
	
	elif what == "gain":
		plot_boxes()

	else:
		print("The type of plot you've specified could not be found!")


if __name__ == "__main__":
	plot("fitness")
	plot("sigma")
	plot("genetic_diversity")
	plot("gain")

