from cynes import * 
import os
import time
import random
from itertools import chain
import torch
import config as cfg
import genetic_algorithm as ga
import random
import struct
import statistics as st

cfg.suppress_ctrl_c()
nes = NES("roms/tetris.nes")

# Global variables
actions = [NES_INPUT_A, NES_INPUT_B, NES_INPUT_DOWN, NES_INPUT_LEFT, NES_INPUT_RIGHT]
action_labels = ['A', 'B', 'Down', 'Left', 'Right']

def initialize():
    while not nes[0x0048] or (nes[0x0058] and (nes[0x0048] == 10)):
        for i in range(4):
            nes.controller = NES_INPUT_START
            nes.step(frames=20)
            nes.controller = 0
        nes.step()
    return nes

def run(nes):

    score = None
    while not nes[0x0058] or nes[0x0058] == 20:

        # Inputs for neural network
        inputs = []
        piece_x = nes[0x0040]
        piece_y = nes[0x0041]
        piece_id = nes[0x0042]
        current_speed = nes[0x0044]
        seed = nes[0x0017] << 8 | nes[0x0018]
        next_piece = nes[0x0019]
        frame = nes[0x00B2] << 8 | nes[0x00B1]

        # One-liner to grab the board state
        board = [nes[0x0400 + i] for i in range(200)]

        # Change the board to 1s and 0s
        board = list(map(lambda x: 1 if x & 0b00010000 else 0, board))

        inputs.extend(board)
        inputs.extend([piece_x, piece_y, piece_id, current_speed, seed, next_piece, frame])

        # Run neural network

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
        nes.controller = random.choice(actions)
        nes[0x00F5] = 0
        nes[0x00F7] = 0
        nes.step()
    
    fitness = ga.fitness(board, score)
    return fitness 

def main():
    nes = initialize()
    print(run(nes))

main()
