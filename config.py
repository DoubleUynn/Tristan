# TODO: Add configuration variables here
import signal
import sys

def suppress_ctrl_c():
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

MINDS_DIR = 'minds'
POPULATION_SIZE = 100 
PARENTS_SIZE = 10
MUTATION_RATE = 75 
MUTATION_FREQUENCY = 50
CROSSING_PROBABILITY = 10
EPOCHS = 500
