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
        model.load_state_dict(torch.load(f'{cfg.MINDS_DIR}/{i}.pt' ))
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
        model.load_state_dict(torch.load(f'{cfg.MINDS_DIR}/{i}.pt'))
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
    for iterator in range(cfg.ELITE_COUNT):

        model_file = '{}/{}.pt'.format(cfg.MINDS_DIR, list_of_bests[iterator])
        temp = Brain()
        temp.load_state_dict(torch.load(model_file))
        torch.save(temp.state_dict(), '{}/{}.pt'.format(cfg.ELITE_DIR, iterator))

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
    if generation < cfg.MUTATION_DECAY_GENERATIONS:
        progress = generation / cfg.MUTATION_DECAY_GENERATIONS
        mutation_rate = cfg.INITIAL_MUTATION_RATE * (1 - progress) + cfg.FINAL_MUTATION_RATE * progress
    else:
        mutation_freq = cfg.FINAL_MUTATION_RATE

    mutations_made = 0
    total_params = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                mutation_prob = torch.rand_like(param, device=device) * 100
                mutation_mask = mutation_prob <= cfg.MUTATION_FREQUENCY

                noise = torch.randn_like(param,device=device) * mutation_rate
                mutated_param = torch.where(mutation_mask, param.data * noise, param.data)
                param.data.copy_(mutated_param)

                mutations_made += torch.sum(mutation_mask).item()
                total_params += param.numel()

    practical_mutation_rate = mutations_made / total_params
    # print(f'Mutation rate: {practical_mutation_rate:.3%} ({mutations_made} / {total_params})')

    return model

def model_diff(m1, m2):
    diff = 0
    count = 0
    for (_, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if p1.requires_grad:
            diff += torch.mean(torch.abs(p1 - p2)).item()
            count += 1

    return diff / count

def breeding(first_parent, second_parent, file_number, generation=0):
    half_offset = (cfg.POPULATION_SIZE - cfg.PARENTS_SIZE) // cfg.PARENTS_SIZE

    for iterator in range(half_offset):
        child = crossing_over(first_parent, second_parent)
        child = mutation(child, generation)
        
        difference_1 = model_diff(child, first_parent)
        difference_2 = model_diff(child, second_parent)

        if difference_1 < 0.001 and difference_2 < 0.001:
            # print("Child identical to parents!")
            pass
        elif difference_1 < 0.01 or difference_2 < 0.01:
            # print("Child is very similar to one parent")
            pass
        else:
            # print("Breeding is working!")
            pass

        child_cpu = Brain()
        child_cpu.load_state_dict({k: v.cpu() for k, v in child.state_dict().items()})
        torch.save(child_cpu.state_dict(), '{}/{}.pt'.format(cfg.MINDS_DIR, file_number))
        file_number += 1

        del child, child_cpu
        torch.cuda.empty_cache()

        child = crossing_over(second_parent, first_parent)
        child = mutation(child, generation)

        difference_1 = model_diff(child, first_parent)
        difference_2 = model_diff(child, second_parent)

        if difference_1 < 0.001 and difference_2 < 0.001:
            # print("Child identical to parents!")
            pass
        elif difference_1 < 0.01 or difference_2 < 0.01:
            # print("Child is very similar to one parent")
            pass
        else:
            # print("Breeding is working!")
            pass

        child_cpu = Brain()
        child_cpu.load_state_dict({k: v.cpu() for k, v in child.state_dict().items()})
        torch.save(child_cpu.state_dict(), '{}/{}.pt'.format(cfg.MINDS_DIR, file_number))
        file_number += 1

        del child, child_cpu
        torch.cuda.empty_cache()

    return file_number


def mating(generation=0):
    counter = cfg.PARENTS_SIZE
    for it in range(0, cfg.PARENTS_SIZE, 2):
        first = Brain().to(device)
        first.load_state_dict(torch.load('{}/{}.pt'.format(cfg.MINDS_DIR, it)))
        second = Brain().to(device)
        second.load_state_dict(torch.load('{}/{}.pt'.format(cfg.MINDS_DIR, it + 1)))
        counter = breeding(first, second, counter, generation)

def elitism_mating(generation=0):
    elite_count = cfg.ELITE_COUNT

    counter = cfg.PARENTS_SIZE

    for it in range(0, cfg.PARENTS_SIZE - 1, 2):
        if it < elite_count:
            continue

        first = Brain().to(device)
        first.load_state_dict(torch.load(f'{cfg.MINDS_DIR}/{it}.pt'))

        second_idx = min(it + 1, cfg.PARENTS_SIZE - 1)
        second = Brain().to(device)
        second.load_state_dict(torch.load(f'{cfg.MINDS_DIR}/{second_idx}.pt'))

        counter = breeding(first, second, counter, generation)

        del first, second
        torch.cuda.empty_cache()

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
    almost_lines = 0
    for y in range(20):
        filled = sum(1 for x in range(10) if board[y][x] == 1)
        if filled >= 8:
            almost_lines += filled - 7

    survival_bonus = min(time, 1000) * 0.1

    flat_segments = 0
    for i in range(9):
        if abs(heights[i] - heights[i+1]) <= 1:
            flat_segments += 1
    flatness_bonus = flat_segments * 2

    # Calculate the fitness score
    fitness = (
            10.0 * score +
            -0.5 * total_height +
            -1.0 * holes +
            -0.3 * bumpiness +
            survival_bonus +
            almost_lines +
            flatness_bonus
    )

    return fitness

class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()
        
        # We're going to treat our two frames of data as two different channels for the purpose of convolution
        self.conv = nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4,3)),
                nn.Flatten())

        # Our output shape should now be 20x18x8, which is 2880
        # We can then append our "next piece" inputs to these sequential layers, which makes it 2887
        self.dense = nn.Sequential(
                nn.Linear((64 * 4 * 3) + 7, 256),
                nn.ReLU(inplace=True),

                nn.Linear(256, 128),
                nn.ReLU(inplace=True),

                nn.Linear(128, 64),
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
        self.cpu()

        board = torch.Tensor(board).to('cpu')
        last_board = torch.Tensor(last_board).to('cpu')
        next_piece = torch.Tensor(next_piece).to('cpu')

        combined_boards = torch.stack([board.squeeze(), last_board.squeeze()], dim=0)
        input_tensor = combined_boards.view(1, 2, 20, 10)

        with torch.no_grad():
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
