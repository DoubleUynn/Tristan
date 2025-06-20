import torch
import numpy as np

import utils.config as cfg
from cynes import *
from utils.piece_maps import piece_maps, next_piece_ids
from utils.genetic_algorithm import Brain, fitness, device  # Import device from genetic_algorithm
import utils.genetic_algorithm as ga

actions = [NES_INPUT_A, NES_INPUT_B, NES_INPUT_DOWN, NES_INPUT_LEFT, NES_INPUT_RIGHT]

def run(mind_num, initializer, elite=False):
    # Each process gets its own NES instance
    nes = initializer()
    brain = Brain().to("cpu")
    try:
        if elite:
            brain.load_state_dict(torch.load('{}/{}.pt'.format(cfg.ELITE_DIR, mind_num)))
        else:
            brain.load_state_dict(torch.load('{}/{}.pt'.format(cfg.MINDS_DIR, mind_num)))
    except Exception as e:
        print(f"Error loading brain {mind_num}: {e}")
        return 0

    score = None
    frames_survived = 0
    actable = False
    last_action = 0
    last_board = [0] * 200
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

        # Superimpose the current piece onto the board
        if piece_id in piece_maps:
            piece_shape = piece_maps[piece_id]
            for y_offset, row in enumerate(piece_shape):
                for x_offset, cell in enumerate(row):
                    if cell:
                        board_x = piece_x + x_offset - 2
                        board_y = piece_y + y_offset - 2
                        if 0 <= board_x < 10 and 0 <= board_y < 20:
                            board_index = board_y * 10 + board_x
                            if board_index < len(board):
                                board[board_index] = 1

        next_piece = next_piece_ids[next_piece]

        # Run neural network
        if actable:
            outputs = brain.activate(board, last_board, next_piece)
            logits = np.array(outputs)
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            if probs.sum() == 0:
               probs = np.ones(len(outputs)) / len(outputs)
               print("random action triggered")

            action = np.random.choice(len(outputs), p=probs)
            # action = outputs.index(max(outputs))
            nes.controller = actions[action]
            last_action = actions[action]
            last_board = board
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
    del nes
    return fitness
