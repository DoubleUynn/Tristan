from cynes.windowed import WindowedNES
import utils.config as cfg

cfg.suppress_ctrl_c()

def play(nes):
    while not nes[0x0058] or nes[0x0058] == 20:
        frame = nes.step()

if __name__ = "__main__":
    nes = WindowedNES("roms/tetris.nes")
    play(nes)
