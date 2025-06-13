from utils.prep import preparation
from utils.run import run
from utils.genetic_algorithm import sort_best, save_best, mating
import utils.config as cfg
from cynes.windowed import WindowedNES
import statistics as st

cfg.suppress_ctrl_c()

def initialize():
    # Create a new NES instance for the process
    nes = WindowedNES("roms/tetris.nes")
    while not nes[0x0048] or (nes[0x0058] and (nes[0x0048] == 10)):
        for i in range(4):
            nes.controller = NES_INPUT_START
            nes.step(frames=20)
            nes.controller = 0
        nes.step()
    return nes

def run_generation():
    scores = [0] * cfg.POPULATION_SIZE
    
    for i in range(cfg.POPULATION_SIZE):
        score = run(i, initialize)
        scores.append(score)
    return scores

def train():
    best_score = 0
    best_epoch = 0
    best_average = 0
    best_average_epoch = 0

    for epoch in range(cfg.EPOCHS):
        scores = run_generation()
        if max(scores) > best_score:
            best_score = round(max(scores), 3)
            best_epoch = epoch + 1

        if st.mean(scores) > best_average:
            best_average = round(st.mean(scores), 3)
            best_average_epoch = epoch + 1

        print(f'Best score: {max(scores)}')
        print(f'Average score: {st.mean(scores):.2f}')
        
        print(f'Epoch: {epoch + 1} - Best epoch: {best_epoch} - Best average epoch: {best_average_epoch}')
        best = sort_best(scores)
        save_best(best)
        mating()

if __name__ == '__main__':
    preparation()
    train()
