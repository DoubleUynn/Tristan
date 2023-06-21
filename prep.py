import os
import shutil
import torch
import config as cfg
import genetic_algorithm as ga

def preparation():
    confirm = input('Do you want to create {} new models? [y/n] '.format(cfg.POPULATION_SIZE))

    if confirm == 'y':
        try:
            shutil.rmtree('minds')
        except FileNotFoundError:
            pass
        os.mkdir('minds')

        for iterator in range(cfg.POPULATION_SIZE):
            temp = ga.Brain()
            torch.save(temp.state_dict(), '{}/{}.pt'.format(cfg.MINDS_DIR, iterator))

        print('Done!')
