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

cfg.suppress_ctrl_c()
nes = NES("roms/tetris.nes")

# Global variables
actions = [NES_INPUT_A, NES_INPUT_B, NES_INPUT_DOWN, NES_INPUT_LEFT, NES_INPUT_RIGHT]
action_labels = ['A', 'B', 'Down', 'Left', 'Right']

def play(nes):

    score = None
    while not nes[0x0058] or nes[0x0058] == 20:
        # One-liner to grab the board state
        board = [nes[0x0400 + i] for i in range(200)]

        # Change the board to 1s and 0s
        board = list(map(lambda x: 1 if x & 0b00010000 else 0, board))

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
        print(nes.controller)
        frame = nes.step()
    
    fitness = ga.fitness(board, score)
    return fitness 
print(play(nes))
