import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_theme(style="darkgrid")

n_processes = 10
n_generations = 100
gens = [i for i in range(n_generations)]
means_TRNMT = np.zeros((n_generations, n_processes))
maxs_TRNMT = np.zeros((n_generations, n_processes))
means_SUS = np.zeros((n_generations, n_processes))
maxs_SUS = np.zeros((n_generations, n_processes))

for enemy in [1, 4, 8]:
	dir_TRNMT = f"data_TRNMT/enemy_{enemy}/"
	dir_SUS = f"data_SUS/enemy_{enemy}/"

	for j in range(n_generations):
		for i in range(n_processes):
			# Tournament selection with a tournament size of 2
			stats = pickle.load(open(f"{dir_TRNMT}rep_{i + 1}/stats_{j}.pkl", "rb"))
			means_TRNMT[j, i] = stats["mean"]
			maxs_TRNMT[j, i] = stats["max"]

			# Stochastic Universal Sampling (SUS)
			stats = pickle.load(open(f"{dir_SUS}rep_{i + 1}/stats_{j}.pkl", "rb"))
			means_SUS[j, i] = stats["mean"]
			maxs_SUS[j, i] = stats["max"]

	df_means_TRNMT = pd.DataFrame(means_TRNMT, columns=["Mean TNMNT" for i in range(n_processes)])
	df_maxs_TRNMT = pd.DataFrame(maxs_TRNMT, columns=["Max TNMNT" for i in range(n_processes)])

	df_means_SUS = pd.DataFrame(means_SUS, columns=["Mean SUS" for i in range(n_processes)])
	df_maxs_SUS = pd.DataFrame(maxs_SUS, columns=["Max SUS" for i in range(n_processes)])

	data = [df_means_TRNMT, df_maxs_TRNMT, df_means_SUS, df_maxs_SUS]
	data = pd.concat(data, 1)

	sns_plot = sns.lineplot(data=data, color=sns.color_palette())
	sns_plot.set_title(f"Enemy {enemy}")
	sns_plot.set_xlabel("Generations")
	sns_plot.set_ylabel("Fitness")
	plt.show()
