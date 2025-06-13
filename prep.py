import os
import shutil
import torch
import config as cfg
from genetic_algorithm import Brain

def preparation():
    confirm = input(f'Do you want to create {cfg.POPULATION_SIZE} new models? [y/n] '

    if confirm == 'y':
        try:
            shutil.rmtree(cfg.MINDS_DIR)
        except FileNotFoundError:
            pass
        os.mkdir(cfg.MINDS_DIR)

        for iterator in range(cfg.POPULATION_SIZE):
            temp = Brain()
            torch.save(temp.state_dict(), '{}/{}.pt'.format(cfg.MINDS_DIR, iterator))

        print('Done!')
