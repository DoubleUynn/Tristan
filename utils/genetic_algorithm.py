from random import randint, sample
import torch
import torch.nn as nn
import numpy as np
import utils.config as cfg
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_initial_diversity():
    models = []
    for i in range(cfg.POPULATION_SIZE):
        model = Brain()
        model.load_state_dict(torch.load(f'{cfg.MINDS_DIR}/{i}.pt', weights_only=True))
        models.append(model)

    total_differences = []
    comparisons = 0

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_diff = 0
            param_count = 0

            for (name1, param1), (name2, param2) in zip(models[i].named_parameters(), models[j].named_parameters()):
                if param1.requires_grad:
                    diff = torch.mean(torch.abs(param1 - param2)).item()
                    model_diff += diff
                    param_count += 1

            avg_model_diff = model_diff / param_count
            total_differences.append(avg_model_diff)
            comparisons += 1

    avg_diversity = sum(total_differences) / len(total_differences)
    min_diversity = min(total_differences)
    max_diversity = max(total_differences)

    print(f'Population diversity stats:')
    print(f'    Average differences: {avg_diversity:.6f}')
    print(f'    Min difference: {min_diversity:.6f}')
    print(f'    Max difference: {max_diversity:.6f}')
    print(f'    Total comparisons: {comparisons}')

    # Thresholds
    if avg_diversity < 0.001:
        print("CRITICAL: Population has almost no diversity!")
    elif avg_diversity < 0.01:
        print("WARNING: Low population diversity.")
    elif avg_diversity > 0.1:
        print("INFO: Very high diversity")
    else:
        print("Population diversity looks reasonable")

def quick_diversity_check():
    indices = sample(range(cfg.POPULATION_SIZE), min(10, cfg.POPULATION_SIZE))

    models = []
    for i in indices:
        model = Brain()
        model.load_state_dict(torch.load(f'{cfg.MINDS_DIR}/{i}.pt', weights_only=True))
        models.append(model)

    total_differences = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_diff = 0
            param_count = 0
            for (name1, param1), (name2, param2) in zip(models[i].named_parameters(), models[j].named_parameters()):
                if param1.requires_grad:
                    diff = torch.mean(torch.abs(param1 - param2)).item()
                    model_diff += diff
                    param_count += 1

            avg_model_diff = model_diff / param_count
            total_differences.append(avg_model_diff)

    return sum(total_differences) / len(total_differences) if total_differences else 0 


def sort_best(scores):
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:cfg.PARENTS_SIZE]

def save_best(list_of_bests):
    for iterator in range(len(list_of_bests)):
        model_file = '{}/{}.pt'.format(cfg.MINDS_DIR, list_of_bests[iterator])
        temp = Brain()
        temp.load_state_dict(torch.load(model_file, weights_only=True))
        torch.save(temp.state_dict(), '{}/{}.pt'.format(cfg.MINDS_DIR, iterator))


def crossing_over(first_parent, second_parent):
    child = Brain().to(device)
    
    with torch.no_grad():
        first_state = first_parent.state_dict()
        second_state = second_parent.state_dict()

        for name, param in child.named_parameters():
            if param.requires_grad:
                first_param = first_state[name].to(device)
                second_param = second_state[name].to(device)

                crossover_prob = torch.rand_like(first_param, device=device) * 100
                crossover_mask = crossover_prob <= cfg.CROSSING_PROBABILITY
                child_param = torch.where(crossover_mask, first_param, second_param)

                param.data.copy_(child_param)
            else:
                param.data.copy_(first_state[name])

    return child

def mutation(model, generation=0):
    if generation < 100:
        mutation_freq = cfg.MUTATION_FREQUENCY * 1.5
        mutation_rate = 400
    elif generation < 300:
        mutation_freq = cfg.MUTATION_FREQUENCY
        mutation_rate = cfg.MUTATION_RATE
    else:
        mutation_freq = cfg.MUTATION_FREQUENCY * 0.7
        mutation_rate = 30

    mutations_made = 0
    total_params = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                mutation_prob = torch.rand_like(param, device=device) * 100
                mutation_mask = mutation_prob <= cfg.MUTATION_FREQUENCY

                mutation_changes = torch.randint(-cfg.MUTATION_RATE, cfg.MUTATION_RATE + 1, param.shape, device=device, dtype=torch.float)
                mutation_factor = 1.0 + (mutation_changes / 1000)
                
                mutated_param = torch.where(mutation_mask, param * mutation_factor, param)
                param.data.copy_(mutated_param)

                mutations_made += torch.sum(mutation_mask).item()
                total_params += param.numel()

    practical_mutation_rate = mutations_made / total_params
    print(f'Mutation rate: {practical_mutation_rate:.3%} ({mutations_made} / {total_params})')

    return model

def breeding(first_parent, second_parent, file_number, generation=0):
    half_offset = (cfg.POPULATION_SIZE - cfg.PARENTS_SIZE) // cfg.PARENTS_SIZE
    # (60 - 18 = 42) // 18

    for iterator in range(half_offset):
        child = crossing_over(first_parent, second_parent)
        child = mutation(child, generation)
        torch.save(child.state_dict(), '{}/{}.pt'.format(cfg.MINDS_DIR, file_number))
        file_number += 1

        child = crossing_over(second_parent, first_parent)
        child = mutation(child, generation)
        torch.save(child.state_dict(), '{}/{}.pt'.format(cfg.MINDS_DIR, file_number))
        file_number += 1

    return file_number


def mating(generation=0):
    counter = cfg.PARENTS_SIZE
    for it in range(0, cfg.PARENTS_SIZE, 2):
        first = Brain()
        first.load_state_dict(torch.load('{}/{}.pt'.format(cfg.MINDS_DIR, it), weights_only=True))
        second = Brain()
        second.load_state_dict(torch.load('{}/{}.pt'.format(cfg.MINDS_DIR, it + 1), weights_only=True))
        counter = breeding(first, second, counter, generation)

def fitness(ending_board, score, time):
    board = [ending_board[i * 10:(i + 1) * 10] for i in range(20)]
    heights = [0] * 10
    for x in range(10):
        for y in range(20):
            if board[y][x] == 1:
                heights[x] = 20 - y
                break

    holes = 0
    for x in range(10):
        block_found = False
        for y in range(20):
            if board[y][x] == 1:
                block_found = True
            elif block_found and board[y][x] == 0:
                holes += 1

    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(9))
    total_height = sum(heights)

    # Calculate the fitness score
    fitness = (
            2 * score -
            8 * holes - 
            4 * bumpiness # +
            # 0.03 * time
    )

    return fitness

# Here's a network that we could potentially use
class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()
        
        # We're going to treat our two frames of data as two different channels for the purpose of convolution
        self.conv = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4,3)),
                nn.Flatten())

        # Our output shape should now be 20x18x8, which is 2880
        # We can then append our "next piece" inputs to these sequential layers, which makes it 2887
        self.dense = nn.Sequential(
                nn.Linear((32 * 4 * 3) + 7, 64),
                nn.ReLU(inplace=True),

                nn.Linear(64, 32),
                nn.ReLU(inplace=True),

                nn.Linear(32, 5))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

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
