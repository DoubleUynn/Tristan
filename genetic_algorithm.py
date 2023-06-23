from random import randint
import torch
import torch.nn as nn
import numpy as np
import config as cfg
import numpy as np


def sort_best(score_array):
    best_scores = [0] * cfg.PARENTS_SIZE
    for it in range(cfg.PARENTS_SIZE):
        best_scores[it] = score_array.index(max(score_array))
        score_array[best_scores[it]] = -1
    best_scores.sort()
    return best_scores


def save_best(list_of_bests):
    for iterator in range(len(list_of_bests)):
        model_file = '{}/{}.pt'.format(cfg.MINDS_DIR, list_of_bests[iterator])
        temp = Brain()
        temp.load_state_dict(torch.load(model_file))
        torch.save(temp.state_dict(), '{}/{}.pt'.format(cfg.MINDS_DIR, iterator))


def crossing_over(first_parent, second_parent):

    child = Brain()
    for layer_name, _ in child.named_parameters():
        child_params = child.state_dict()[layer_name]
        first_params = first_parent.state_dict()[layer_name]
        second_params = second_parent.state_dict()[layer_name]
        for tensor in range(len(child_params)):
            try:
                for value in range(len(child_params[tensor])):
                    probability = randint(1, 100)
                    if probability <= cfg.CROSSING_PROBABILITY:
                        child_params[tensor][value] = second_params[tensor][value]
                    else:
                        child_params[tensor][value] = first_params[tensor][value]

            except TypeError:
                probability = randint(1, 100)
                if probability <= cfg.CROSSING_PROBABILITY:
                    child_params[tensor] = second_params[tensor]
                else:
                    child_params[tensor] = first_params[tensor]

        child.state_dict()[layer_name] = child_params

    return child


def mutation(model):

    for layer_name, _ in model.named_parameters():
        layer_params = model.state_dict()[layer_name]
        for tensor in range(len(layer_params)):
            try:  # when tensor is weight tensor
                for value in range(len(layer_params[tensor])):
                    probability = randint(1, 100)
                    change = randint(-cfg.MUTATION_RATE, cfg.MUTATION_RATE)
                    if probability <= cfg.MUTATION_FREQUENCY:
                        layer_params[tensor][value] = layer_params[tensor][value] \
                                                      + layer_params[tensor][value] \
                                                      * (change / 1000)

            except TypeError:  # when tensor is bias tensor
                probability = randint(1, 100)
                change = randint(-cfg.MUTATION_RATE, cfg.MUTATION_RATE)
                if probability <= cfg.MUTATION_FREQUENCY:
                    layer_params[tensor] = layer_params[tensor] + layer_params[tensor] * (change / 1000)
        model.state_dict()[layer_name] = layer_params
    return model


def breeding(first_parent, second_parent, file_number):
    half_offset = (cfg.POPULATION_SIZE - cfg.PARENTS_SIZE) // cfg.PARENTS_SIZE

    for iterator in range(half_offset):
        child = crossing_over(first_parent, second_parent)
        child = mutation(child)
        torch.save(child.state_dict(), '{}/{}.pt'.format(cfg.MINDS_DIR, file_number))
        file_number += 1

        child = crossing_over(second_parent, first_parent)
        child = mutation(child)
        torch.save(child.state_dict(), '{}/{}.pt'.format(cfg.MINDS_DIR, file_number))
        file_number += 1

    return file_number


def mating():
    counter = cfg.PARENTS_SIZE
    for it in range(0, cfg.PARENTS_SIZE, 2):
        first = Brain()
        first.load_state_dict(torch.load('{}/{}.pt'.format(cfg.MINDS_DIR, it)))
        second = Brain()
        second.load_state_dict(torch.load('{}/{}.pt'.format(cfg.MINDS_DIR, it + 1)))
        counter = breeding(first, second, counter)

def fitness(ending_board, score, time):
    ones = ending_board.count(1)
    zeros = ending_board.count(0)
    filled_density = ones / (ones + zeros)
    unfilled_density = zeros / (ones + zeros)

    # Calculate the fitness score
    fitness = 3 * score - 2 * unfilled_density + 4 * filled_density + 0.25 * time
    return fitness

# Here's a network that we could potentially use
class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()

        self.in_nodes = 208
        self.hidden_nodes1 = 200 
        self.hidden_nodes2 = 175
        self.hidden_nodes3 = 150
        self.hidden_nodes4 = 100
        self.hidden_nodes5 = 75
        self.hidden_nodes6 = 50
        self.out_nodes = 5
        
        self.net = nn.Sequential(nn.Linear(self.in_nodes, self.hidden_nodes1),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_nodes1, self.hidden_nodes2),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_nodes2, self.hidden_nodes3),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_nodes3, self.hidden_nodes4),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_nodes4, self.hidden_nodes5),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_nodes5, self.hidden_nodes6),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_nodes6, self.out_nodes),
                                 nn.ReLU())

    def activate(self, inputs):
        # Get the next move from the network
        inputs = torch.tensor(inputs).float()
        net_product = self.net(inputs).tolist()

        return net_product

    def get_weights(self):
        weights = []
        for i in self.net:
            if not isinstance(i, torch.nn.modules.activation.ReLU):
                weights.append(i.weight)

        return weights
