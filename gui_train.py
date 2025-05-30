from cynes.windowed import WindowedNES
from cynes import * 
import os
import time
import random
from itertools import chain
import genetic_algorithm as ga
from genetic_algorithm import Brain
import torch
import config as cfg
import struct
import statistics as st

cfg.suppress_ctrl_c()
nes = WindowedNES("roms/tetris.nes")

# Global variables
actions = [NES_INPUT_A, NES_INPUT_B, NES_INPUT_DOWN, NES_INPUT_LEFT, NES_INPUT_RIGHT]
action_labels = ['A', 'B', 'Down', 'Left', 'Right']

def initialize():
    print("initializing...")
    while not nes[0x0048] or (nes[0x0058] and (nes[0x0048] == 10)):
        for i in range(4):
            nes.controller = NES_INPUT_START
            nes.step(frames=20)
            nes.controller = 0
        nes.step()
    print("initialized.")
    return nes

def run(mind_num, nes):
    brain = Brain() 
    brain.load_state_dict(torch.load('{}/{}.pt'.format(cfg.MINDS_DIR, mind_num)))

    score = None
    frames_survived = 0
    actable = False
    last_action = 0
    while not nes[0x0058] or nes[0x0058] == 20:
        nes.controller = 0

        # Inputs for neural network
        inputs = []
        piece_x = nes[0x0040]
        piece_y = nes[0x0041]
        piece_id = nes[0x0042]
        current_speed = nes[0x0044]
        seed = nes[0x0017] << 8 | nes[0x0018]
        next_piece = nes[0x0019]
        frame_number = nes[0x00B2] << 8 | nes[0x00B1]
        actable = not actable

        # One-liner to grab the board state
        board = [nes[0x0400 + i] for i in range(200)]

        # Change the board to 1s and 0s
        board = list(map(lambda x: 1 if x & 0b00010000 else 0, board))

        inputs.extend(board)
        inputs.extend([piece_x, piece_y, piece_id, current_speed, seed, next_piece, frame_number, last_action])

        # Run neural network
        # Only act on every other frame, idk if this fixes the inputs or not
        if actable:
            # TODO: Softmax outputs
            outputs = brain.activate(inputs)
            action = outputs.index(max(outputs))
            nes.controller = actions[action]
            last_action = actions[action]
        else:
            nes.controller = 0
            nes[0x00F5] = 0
            nes[0x00F7] = 0

        # Get the current score
        # Score is stored in two bytes, binary coded digits, little endian
        # This is how I convert it to an integer, but there is likely a better way
        byte1 = nes[0x0055].to_bytes(1, 'little')
        score2 = byte1[0] & 0b00001111
        score1 = byte1[0] >> 4 & 0b00001111

        byte2 = nes[0x0054].to_bytes(1, 'little')
        score4 = byte2[0] & 0b00001111
        score3 = byte2[0] >> 4 & 0b00001111

        byte3 = nes[0x0053].to_bytes(1, 'little')
        score6 = byte3[0] & 0b00001111
        score5 = byte3[0] >> 4 & 0b00001111

        score = int(str(score1) + str(score2) + str(score3) + str(score4) + str(score5) + str(score6))
        frame = nes.step()
        nes.controller = 0
        nes[0x00F5] = 0
        nes[0x00F7] = 0
        frames_survived += 1
    
    fitness = ga.fitness(board, score, frames_survived)
    print(f'Brain: {mind_num}; fitness: {fitness}')
    return fitness 

def run_generation():
    scores = []
    for i in range(cfg.POPULATION_SIZE):
        score = run(i, initialize())
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
        
        print('Epoch: {} - Best epoch: {} - Best average epoch: {}'.format(epoch + 1, best_epoch, best_average_epoch))
        best = ga.sort_best(scores)
        ga.save_best(best)
        ga.mating()
