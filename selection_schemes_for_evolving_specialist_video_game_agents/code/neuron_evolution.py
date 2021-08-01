################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

################################################################################################
#                               Evolutionary Computing Task 1                                  #
#                               By Tim, Alex, Xiaojin and Lando                                #
################################################################################################

import sys, os
sys.path.insert(0, 'evoman')

from environment import Environment
from neuron_controller import Neuron_Controller

from deap import tools
from deap import base, creator

import argparse
import random
import numpy as np
import pickle
import array
from multiprocessing import Process

def process_args():
    global args

    parser = argparse.ArgumentParser(description='neural network evolution by playing evoman')

    # Crossover
    parser.add_argument("--CXPB", metavar='', type=float, default=0.5,
                        help="Crossover probability.")
    # parser.add_argument("--CXINDPB", metavar='', type=float, default=0.1,
    #                     help="In uniform crossover, the probability for each weight to cross over.")

    # Mutation
    parser.add_argument("--MUTPB", metavar='', type=float, default=0.2,
                        help="Individual probability of undergoing mutation.")
    parser.add_argument("--MUTINDPB", metavar='', type=float, default=1.0,
                        help="The probability for each weight being altered during mutation.")

    # Other
    parser.add_argument("--POP_SIZE", metavar='', type=int, default=50,
                        help="Population size. This size is kept fixed during evolution.")
    parser.add_argument("--TOURNAMENT_SIZE", metavar='', type=int, default=2,
                        help="Size for tournament parameter.")
    parser.add_argument("--MAX_GENERATIONS", metavar='', type=int, default=100,
                        help="Maximum number of generations being created.")
    parser.add_argument("--ENEMY", metavar='', type=int, default=1,
                        help="Which enemy to fight. Note that you can only specify a single enemy!")
    parser.add_argument("--SAVE_DIR", metavar='', type=str, default="standard_ffnn_task1",
                        help="The directory where results are saved.")
    parser.add_argument("--BLOCK_GAME_PRINTS", metavar='', type=bool, default=True,
                        help="Blocks the print statements that happen in .play().")
    parser.add_argument("--SELECTION", metavar='', type=str, default='tour',
                        help="The selection method for parents, tour or sus")
    parser.add_argument("--NOTES", metavar='', type=str, default='default_settings',
                        help="Extra notes on the experiment, for instance the use of alternative settings.")

    args = parser.parse_args()
    
    assert args.SELECTION.lower() in ['tour', 'sus'], "SELECTION must be 'tour' or 'sus'"


def init_env():
    results_dir = '{}/enemy_{}_{}/'.format(args.SAVE_DIR, args.ENEMY, args.NOTES)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return Environment(experiment_name=results_dir,
                       enemies=[args.ENEMY],
                       playermode="ai",
                       player_controller=Neuron_Controller(),
                       enemymode="static",
                       level=2,
                       speed="fastest")


def init_DEAP(env, evaluator):
    toolbox = base.Toolbox()
    n_weights = 265

    # init by sampling from normal distribution
    def init_func():
        return np.random.uniform(-1, 1)

    # Maximize reward
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # A population is made from individuals, which are each initialized with init_func
    toolbox.register("attribute", init_func)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attribute, n=n_weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=args.MUTINDPB)

    if args.SELECTION == 'tour':
        toolbox.register("select", tools.selTournament, tournsize=args.TOURNAMENT_SIZE)
    if args.SELECTION == 'sus':
        toolbox.register("select", tools.selStochasticUniversalSampling)

    toolbox.register("evaluate", evaluator.evaluate_individual, env=env)

    pop = toolbox.population(n=args.POP_SIZE)

    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    return pop, toolbox

class Evaluator():
    def __init__(self, p_num):

        self.max_ind_gain = -float("inf")
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("max", np.max)
        self.stats.register("mean", np.mean)
        self.stats.register("std", np.std)
        self.n_wins = 0
        self.latest_stats = None

        if not os.path.exists(args.SAVE_DIR):
            os.makedirs(args.SAVE_DIR)

        # The directory specific to this run
        arg_dir_name = ''.join([char if char.isalnum() or char =='.' else '_' for char in str(vars(args)).replace(' ','').replace("'",'')])
        self.run_dir = f"{args.SAVE_DIR}/enemy_{args.ENEMY}_{args.NOTES}/{arg_dir_name}/rep_{p_num}/"

        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

    def evaluate_population(self, population, generation):
        statistics = self.stats.compile(population)
        self.pickle_dump(statistics, f"stats_{generation}")
        # pickle.dump(statistics, open(f"{self.run_dir}stats_{generation}.pkl", "wb"))
        return statistics


    def evaluate_individual(self, individual, env):

        # Disables stdout
        if args.BLOCK_GAME_PRINTS:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

        # Play the game
        score, phealth, ehealth, time = env.play(np.asarray(individual))

        # Enables stdout
        if args.BLOCK_GAME_PRINTS:
            sys.stdout = old_stdout

        if ehealth == 0:
            self.n_wins += 1

        ind_gain = phealth - ehealth
        if ind_gain > self.max_ind_gain:
            self.max_ind_gain = ind_gain
            self.pickle_dump(individual, "champion")
            # pickle.dump(individual, open(self.run_dir + "champion.pkl", "wb"))

        return score,

    def pickle_dump(self, variable, name):
        pickle.dump(variable, open(f"{self.run_dir}{name}.pkl", "wb"))

    def get_champion(self):
        return pickle.load(open(self.run_dir + "champion.pkl", "rb"))


def print_summary(worker_id, gen, stats, wins, champ_score):
    summary = "\n"
    summary += "====================================================\n"
    summary += f"Worker {worker_id:2} generation {gen:3} summary:\n"
    summary += f"mean: {stats['mean']:.3f}, max: {stats['max']:.3f}, std: {stats['std']:.3f}\n"
    summary += f"Enemies defeated so far: {wins}\n"
    summary += f"Champion individual gain: {champ_score:.3f}\n"
    summary += "====================================================\n"
    print(summary)


def evolve_population(env, worker_id, pop, toolbox, evaluator):

    for g in range(args.MAX_GENERATIONS):
        stats = evaluator.evaluate_population(pop, g)
        print_summary(worker_id, g, stats, evaluator.n_wins, evaluator.max_ind_gain)


        # Code below is from the DEAP example: https://deap.readthedocs.io/en/master/overview.html
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        random.shuffle(offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < args.CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < args.MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    # One final evaluation, gen 100!
    stats = evaluator.evaluate_population(pop, args.MAX_GENERATIONS)
    print_summary(worker_id, args.MAX_GENERATIONS, stats, evaluator.n_wins, evaluator.max_ind_gain)
    evaluator.pickle_dump(pop, "final_pop")  # Just in case, save the final population

    print(f"Worker {worker_id} done with evolutions. Now testing champion.")
    champion = evaluator.get_champion()
    for t in range(1, 6):

        # Disables stdout
        if args.BLOCK_GAME_PRINTS:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

        # Play the game
        scores = env.play(np.asarray(champion))

        # Enables stdout
        if args.BLOCK_GAME_PRINTS:
            sys.stdout = old_stdout

        evaluator.pickle_dump(scores, f"Champion_scores_{t}")



def main(id):
    print(f"Worker {id} now started.")

    process_args()
    env = init_env()
    evaluator = Evaluator(id)
    pop, toolbox = init_DEAP(env, evaluator)
    evolve_population(env, id, pop, toolbox, evaluator)

    print(f"Worker {id} finished!")


if __name__ == "__main__":

    # main(1)  # Uncomment for testing

    # Multiprocessing: https://docs.python.org/2/library/multiprocessing.html
    pool = []
    for i in range(1, 11):
        p = Process(target=main, args=[i])
        pool.append(p)

    active = 0
    running = []
    for p in pool:
        while active >= 5:
            for r in running:
                r.join(1)  # Waits at most 1 second
                if r.exitcode is not None:
                    running.remove(r)
                    active -= 1
                    break

        p.start()
        running.append(p)
        active += 1

    for p in running:
        p.join()

    print(f"Experiment finished!")
