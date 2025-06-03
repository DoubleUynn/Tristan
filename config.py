import signal
import sys
import multiprocessing

def suppress_ctrl_c():
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

MINDS_DIR = 'minds'
POPULATION_SIZE = 60
PARENTS_SIZE = 8
MUTATION_RATE = 60
MUTATION_FREQUENCY = 25
CROSSING_PROBABILITY = 10
EPOCHS = 700

# Multi-threading configuration
# Set to 0 to automatically use all available CPU cores
MAX_WORKERS = 0  # Will default to multiprocessing.cpu_count() if set to 0
