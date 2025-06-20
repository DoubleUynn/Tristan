import signal
import sys
import multiprocessing

def suppress_ctrl_c():
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

MINDS_DIR = 'minds'
POPULATION_SIZE = 100
PARENTS_SIZE = 20

INITIAL_MUTATION_RATE = 0.1
FINAL_MUTATION_RATE = 0.02
MUTATION_DECAY_GENERATIONS = 500
MUTATION_FREQUENCY = 15
CROSSING_PROBABILITY = 50

EPOCHS = 1000

# Multi-threading configuration
# Set to 0 to automatically use all available CPU cores
MAX_WORKERS = 0  # Will default to multiprocessing.cpu_count() if set to 0

ELITE_COUNT = 2
ELITE_DIR = 'elites'
TOURNAMENT_SIZE = 3

