from cynes import * 
import os
import time
import random

nes = NES("roms/tetris.nes")

def initialize():
    nes.reset()
    started = False
    while not started:
        print(nes[0x00C3])
        if nes[0x00C3] == 0:
            for i in range(4):
                nes.controller = NES_INPUT_START
                nes.step(frames=20)
                nes.controller = 0
            started = True
        nes.step()

initialize()


while not nes.should_close():
    nes.controller = 0
    # Inputs for neural network
    piece_x = nes[0x0040]
    piece_y = nes[0x0041]
    piece_id = nes[0x0042]
    current_speed = nes[0x0044]

    board = [
            [nes[0x0400 + i] for i in range(10)],
            [nes[0x040A + i] for i in range(10)],
            [nes[0x0414 + i] for i in range(10)],
            [nes[0x041E + i] for i in range(10)],
            [nes[0x0428 + i] for i in range(10)],
            [nes[0x0432 + i] for i in range(10)],
            [nes[0x043C + i] for i in range(10)],
            [nes[0x0446 + i] for i in range(10)],
            [nes[0x0450 + i] for i in range(10)],
            [nes[0x045A + i] for i in range(10)],
            [nes[0x0464 + i] for i in range(10)],
            [nes[0x046E + i] for i in range(10)],
            [nes[0x0478 + i] for i in range(10)],
            [nes[0x0482 + i] for i in range(10)],
            [nes[0x048C + i] for i in range(10)],
            [nes[0x0496 + i] for i in range(10)],
            [nes[0x04A0 + i] for i in range(10)],
            [nes[0x04AA + i] for i in range(10)],
            [nes[0x04B4 + i] for i in range(10)],
            [nes[0x04BE + i] for i in range(10)]
            ]
    
    seed = nes[0x0017] << 8 | nes[0x0018]
    next_piece = nes[0x0019]
    frame = nes[0x00B2] << 8 | nes[0x00B1]

    if frame % 100 == 0:
        print("Frame: ", frame)
        os.system("clear")
        for i in range(20):
            for j in range(10):
                if board[i][j] & 0b00010000:
                    print("[O]", end="")
                else:
                    print("[ ]", end="")
            print()
        print("Controller Initial:", format(nes[0x00B5], '08b'))
        print("Controller:", format(nes[0x00B6], '08b'))
        print("Seed:", seed)
        print(piece_x, piece_y, piece_id)



    frame = nes.step()
