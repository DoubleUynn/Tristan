import os
import shutil
import torch
import config as cfg
import genetic_algorithm

def preparation():
    confirm = input('Do you want to create {} new models? [y/n] '.format(cfg.POPULATION_SIZE))

    if confirm == 'y':
        try:
            shutil.rmtree('data')
        except FileNotFoundError:
            pass
        os.mkdir('data')

        for iterator in range(cfg.POPULATION_SIZE):
            temp = genetic_algorithm.Brain()
            torch.save(temp.state_dict(), 'data/{}.pt'.format(iterator))

        print('Done!')
