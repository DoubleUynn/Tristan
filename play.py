from cynes.windowed import WindowedNES
from cynes import * # needed for the constants for NES_INPUT_*
from piece_maps import piece_maps
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
nes = WindowedNES("roms/tetris.nes")

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
        inputs = []
        piece_x = nes[0x0040]
        piece_y = nes[0x0041]
        piece_id = nes[0x0042]
        # Superimpose the current piece onto the board
        if piece_id in piece_maps:
            piece_shape = piece_maps[piece_id]
            for y_offset, row in enumerate(piece_shape):
                for x_offset, cell in enumerate(row):
                    if cell:
                        board_x = piece_x + x_offset
                        board_y = piece_y + y_offset
                        if 0 <= board_x < 10 and 0 <= board_y < 20:
                            board_index = board_y * 10 + board_x
                            if board_index < len(board):
                                board[board_index] = 1
        
        current_speed = nes[0x0044]
        seed = nes[0x0017] << 8 | nes[0x0018]
        next_piece = nes[0x0019]
        frame_number = nes[0x00B2] << 8 | nes[0x00B1]

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
        
        for i, block in enumerate(board):
            if i % 10 == 0:
                print("▒", end="")
            if block:
                print("█", end="")
            else:
                print(" ", end="")
            if (i + 1) % 10 == 0:
                print("▒")

        print("")

        print("Piece X:", piece_x, "Piece Y:", piece_y)

        time.sleep(0.027)
        os.system("clear")

        frame = nes.step()
    
    fitness = ga.fitness(board, score)
    return fitness 

play(nes)
