################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

################################################################################################
#                               Evolutionary Computing Task 2                                  #
#                            By Tim, Eui Yeon, Xiaojin and Lando                               #
################################################################################################

import sys, os

sys.path.insert(0, 'evoman')

from environment import Environment
from demo_controller import player_controller

from deap import tools
from deap import base, creator

import argparse
import random
import numpy as np
import pickle
import array
from multiprocessing import Process
import time


def process_args():
    global args

    parser = argparse.ArgumentParser(description='neural network evolution by playing evoman')

    # Mutation
    parser.add_argument("--MUT_METHOD", metavar='', type=str, default="time_based",
                        help="Individual probability of undergoing mutation.")
    parser.add_argument("--MUTINDPB", metavar='', type=float, default=0.2,
                        help="Chromosome probability of undergoing mutation.")

    # Other
    parser.add_argument("--POP_SIZE", metavar='', type=int, default=100,
                        help="Population size. This size is kept fixed during evolution.")
    parser.add_argument("--TOURNAMENT_SIZE", metavar='', type=int, default=2,
                        help="Size for tournament parameter.")
    parser.add_argument("--MAX_GENERATIONS", metavar='', type=int, default=100,
                        help="Maximum number of generations being created.")
    parser.add_argument("--ENEMIES", metavar='', type=list, default=[4, 6, 7],
                        help="The group of enemies to train against.")
    parser.add_argument("--SAVE_DIR", metavar='', type=str, default="task2",
                        help="The directory where results are saved.")
    parser.add_argument("--BLOCK_GAME_PRINTS", metavar='', type=bool, default=True,
                        help="Blocks the print statements that happen in .play().")

    args = parser.parse_args()
    
    assert args.MUT_METHOD in ['dynamic', 'time_based'], "Mutation Method must be 'dynamic' or 'time_based'"


def init_env():
    if sys.platform == 'darwin':
        os.putenv('SDL_VIDEODRIVER', 'fbcon')
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    results_dir = '{}/enemy_{}_{}/'.format(args.SAVE_DIR, args.ENEMIES, args.MUT_METHOD)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    envs = []
    for i in range(len(args.ENEMIES)):
        env = Environment(experiment_name=results_dir,
                          enemies=[args.ENEMIES[i]],
                          playermode="ai",
                          player_controller=player_controller(10),  # 10 HIDDEN NEURONS
                          enemymode="static",
                          level=2,
                          speed="fastest")
        envs.append(env)

    return envs


# Creates an individual representation with weights initialized by uniformly
# sampling between ind_min and ind_max, and for each weight adding a mutation
# rate of mut_init.
def generate_ind(icls, scls, size, ind_min, ind_max, mut_init):
    ind = icls(random.uniform(ind_min, ind_max) for _ in range(size))
    ind.strategy = scls(mut_init for _ in range(size))
    return ind


# In order to prevent too much exploitation, check whether all sigma's
# are above the threshold mut_lb and if not, set sigma to mut_lb.
def check_mutation_lb(mut_lb):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < mut_lb:
                        child.strategy[i] = mut_lb
            return children

        return wrappper

    return decorator


def init_DEAP(envs, evaluator):
    toolbox = base.Toolbox()
    n_weights = 265
    ind_min = -1  # minimum weight value
    ind_max = 1  # maximum weight value
    mut_init = 1  # initial mutation rate for every weight
    mut_lb = 0.005  # lower bound on the mutation rates

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
    creator.create("Strategy", array.array, typecode="d")  # array containing the mutation sigma's

    toolbox.register("individual", generate_ind, creator.Individual, creator.Strategy,
                     n_weights, ind_min, ind_max, mut_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    if args.MUT_METHOD == 'dynamic':
        toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=args.MUTINDPB)
        toolbox.decorate("mutate", check_mutation_lb(mut_lb))
    else:
        toolbox.register("mutate", tools.mutGaussian, indpb=args.MUTINDPB, mu=0)

    toolbox.register("select", tools.selTournament, tournsize=args.TOURNAMENT_SIZE)
    # toolbox.register("evaluate", evaluator.evaluate_individual, env=env)  # , env=env ???????

    pop = toolbox.population(n=args.POP_SIZE)

    for ind in pop:
        ind.fitness.values = evaluator.evaluate_individual(ind, envs)

    return pop, toolbox


class Evaluator():
    def __init__(self, p_num):

        self.max_ind_gain = -float("inf")
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("max", np.max)
        self.stats.register("mean", np.mean)
        self.stats.register("std", np.std)
        self.n_wins = [0 for _ in range(len(args.ENEMIES))]
        self.latest_stats = None

        if not os.path.exists(args.SAVE_DIR):
            os.makedirs(args.SAVE_DIR)

        # The directory specific to this run
        self.run_dir = f"{args.SAVE_DIR}/enemy_{args.ENEMIES}_{args.MUT_METHOD}/rep_{p_num}/"

        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

    def evaluate_population(self, population, generation):
        statistics = self.stats.compile(population)
        self.pickle_dump(statistics, f"stats_{generation}")
        self.pickle_dump(population, f"whole_pop_{generation}")
        return statistics

    def evaluate_individual(self, individual, envs):
        # Disables stdout
        if args.BLOCK_GAME_PRINTS:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

        # Play the game
        scores = []
        phealths = []
        ehealths = []
        for index, env in enumerate(envs):
            score, phealth, ehealth, time = env.play(np.asarray(individual))
            scores.append(score)
            phealths.append(phealth)
            ehealths.append(ehealth)

            if ehealth == 0:
                self.n_wins[index] += 1
        score = np.mean(scores) - np.std(scores)
        phealth = np.sum(phealths)
        ehealth = np.sum(ehealths)

        # Enables stdout
        if args.BLOCK_GAME_PRINTS:
            sys.stdout = old_stdout

        ind_gain = phealth - ehealth
        if ind_gain > self.max_ind_gain:
            self.max_ind_gain = ind_gain
            self.pickle_dump(individual, "champion")

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


def evolve_population(envs, worker_id, pop, toolbox, evaluator):
    for g in range(args.MAX_GENERATIONS):
        start_time = time.time()
        stats = evaluator.evaluate_population(pop, g)
        print_summary(worker_id, g, stats, evaluator.n_wins, evaluator.max_ind_gain)

        # Select the parents. We generate mu offspring from 2 * mu parents
        parents = toolbox.select(pop, 2 * len(pop))
        # Clone the selected individuals
        parents = list(map(toolbox.clone, parents))

        # Apply crossover on the offspring. Crossover is performed in place
        for parent1, parent2 in zip(parents[::2], parents[1::2]):
            alpha = random.random()

            # Crossover on all x_i and sigma_i
            for i in range(265):  # This number is fixed!
                parent1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
                if args.MUT_METHOD == 'dynamic':
                    parent1.strategy[i] = alpha * parent1.strategy[i] + (1 - alpha) * parent2.strategy[i]

            del parent1.fitness.values

        # Two parents create 1 child, like in the paper!
        # Since crossover was performed in place, we simply ditch the second parent.
        offspring = parents[::2]

        if args.MUT_METHOD == 'dynamic':
            # NOTE Mutation probability is present in the lognormal operator, so no
            # if-condition required here
            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        else:
            new_sigma = 1 - 0.9 * ((g + 1) / args.MAX_GENERATIONS)
            for mutant in offspring:
                toolbox.mutate(mutant, sigma=new_sigma)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        mu_plus_lambda = pop + offspring
        new_pop = []
        for i in range(args.POP_SIZE):
            ind1 = random.choice(mu_plus_lambda)
            ind2 = random.choice(mu_plus_lambda)

            if not ind1.fitness.values:
                ind1.fitness.values = evaluator.evaluate_individual(ind1, envs)

            if not ind2.fitness.values:
                ind2.fitness.values = evaluator.evaluate_individual(ind2, envs)

            if ind1.fitness.values[0] > ind2.fitness.values[0]:
                new_pop.append(toolbox.clone(ind1))
            else:
                new_pop.append(toolbox.clone(ind2))

        pop = new_pop
        print(f"time: {time.time() - start_time}")

    # One final evaluation, gen 100!
    stats = evaluator.evaluate_population(pop, args.MAX_GENERATIONS)
    print_summary(worker_id, args.MAX_GENERATIONS, stats, evaluator.n_wins, evaluator.max_ind_gain)
    evaluator.pickle_dump(pop, "final_pop")  # Just in case, save the final population

    print(f"Worker {worker_id} done with evolutions. Now testing champion.")
    champion = evaluator.get_champion()

    enemies_rest = [3, 4, 6, 7, 8] if args.ENEMIES == [1, 2, 5] else [1, 2, 3, 5, 8]
    results_dir = '{}/enemy_{}_{}/'.format(args.SAVE_DIR, args.ENEMIES, args.MUT_METHOD)

    for i in range(len(enemies_rest)):
        env = Environment(experiment_name=results_dir,
                          enemies=[enemies_rest[i]],
                          playermode="ai",
                          player_controller=player_controller(10),  # 10 HIDDEN NEURONS
                          enemymode="static",
                          level=2,
                          speed="fastest")
        envs.append(env)

    for t in range(1, 6):
        # Disables stdout
        if args.BLOCK_GAME_PRINTS:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

        # Play the game
        scores = []
        for env in envs:
            score = env.play(np.asarray(champion))
            scores.append(score)

        # Enables stdout
        if args.BLOCK_GAME_PRINTS:
            sys.stdout = old_stdout

        evaluator.pickle_dump(scores, f"Champion_scores_{t}_all")


def main(id):
    print(f"Worker {id} now started.")

    process_args()
    envs = init_env()
    evaluator = Evaluator(id)
    pop, toolbox = init_DEAP(envs, evaluator)
    evolve_population(envs, id, pop, toolbox, evaluator)

    print(f"Worker {id} finished!")


if __name__ == "__main__":

    # main(1)  # Uncomment for testing

    # Multiprocessing: https://docs.python.org/2/library/multiprocessing.html
    pool = []
    # for i in range(1, 11):
    for i in range(3, 4):

        p = Process(target=main, args=[i])
        pool.append(p)

    active = 0
    running = []
    for p in pool:
        while active >= 1: # NOTE: the number of threads
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
