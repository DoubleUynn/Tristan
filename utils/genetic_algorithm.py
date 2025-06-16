from random import randint
import torch
import torch.nn as nn
import numpy as np
import utils.config as cfg
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        temp.load_state_dict(torch.load(model_file, weights_only=True))
        torch.save(temp.state_dict(), '{}/{}.pt'.format(cfg.MINDS_DIR, iterator))


def crossing_over(first_parent, second_parent):

    child = Brain().to(device)
    
    with torch.no_grad():
        for layer_name in child.state_dict():
            first_params = first_parent.state_dict()[layer_name].to(device)
            second_params = first_parent.state_dict()[layer_name].to(device)

            crossover_prob = torch.rand_like(first_params, device=device) * 100
            crossover_mask = crossover_prob <= cfg.CROSSING_PROBABILITY

            child_params = torch.where(crossover_mask, first_params, second_params)
            child.state_dict()[layer_name].copy_(child_params)

    return child

def mutation(model):

    with torch.no_grad():
        for layer_name in model.state_dict():
            layer_params = model.state_dict()[layer_name]

            mutation_prob = torch.rand_like(layer_params, device=device) * 100
            mutation_mask = mutation_prob <= cfg.MUTATION_FREQUENCY

            mutation_changes = torch.randint(-cfg.MUTATION_RATE, cfg.MUTATION_RATE + 1, layer_params.shape, device=device, dtype=torch.float)
            mutation_factor = 1.0 + (mutation_changes / 1000)
            
            mutated_params = torch.where(mutation_mask, layer_params * mutation_factor, layer_params)
            model.state_dict()[layer_name].copy_(mutated_params)

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
        first.load_state_dict(torch.load('{}/{}.pt'.format(cfg.MINDS_DIR, it), weights_only=True))
        second = Brain()
        second.load_state_dict(torch.load('{}/{}.pt'.format(cfg.MINDS_DIR, it + 1), weights_only=True))
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
        
        # We're going to treat our two frames of data as two different channels for the purpose of convolution
        self.conv = nn.Sequential(
                nn.Conv2d(2, 32, 2, stride=2),
                nn.ReLU(),
                nn.Flatten())

        # Our output shape should now be 20x18x8, which is 2880
        # We can then append our "next piece" inputs to these sequential layers, which makes it 2887
        self.dense = nn.Sequential(
                nn.Linear(1607, 1028),
                nn.ReLU(),
                nn.Linear(1028, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 5),
                nn.ReLU())

    def activate(self, board, last_board, next_piece):
        board = torch.Tensor(board)
        last_board = torch.Tensor(last_board)
        next_piece = torch.Tensor(next_piece)

        combined_boards = torch.stack([board.squeeze(), last_board.squeeze()], dim=0)
        input_tensor = combined_boards.view(1, 2, 20, 10)

        conv_result = self.conv(input_tensor)
        dense_inputs = torch.cat([conv_result.squeeze(), next_piece])

        output = self.dense(dense_inputs).tolist()

        return output 

    def get_weights(self):
        weights = []
        for i in self.conv:
            if not isinstance(i, torch.nn.modules.activation.ReLU):
                weights.append(i.weight)
        for i in self.dense:
            if not isinstance(i, torch.nn.modules.activation.ReLU):
                weights.append(i.weight)

        return weights
