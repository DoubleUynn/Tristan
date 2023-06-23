from prep import preparation
from headless_train import train
import multiprocessing as mp

if __name__ == '__main__':
    preparation()
    train()
