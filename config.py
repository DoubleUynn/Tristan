# TODO: Add configuration variables here
import signal
import sys
import multiprocessing

def suppress_ctrl_c():
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

MINDS_DIR = 'minds'
POPULATION_SIZE = 50 
PARENTS_SIZE = 2
MUTATION_RATE = 75 
MUTATION_FREQUENCY = 70
CROSSING_PROBABILITY = 30
EPOCHS = 500

# Multi-threading configuration
# Set to 0 to automatically use all available CPU cores
MAX_WORKERS = 0  # Will default to multiprocessing.cpu_count() if set to 0
