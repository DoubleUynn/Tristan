import sys
import utils.config as cfg
from gui_train import initialize, run_generation
from utils.run import run

cfg.suppress_ctrl_c()

def run_brain(mind_num):
    score = run(mind_num, initialize, True)
    return score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(run_generation())
    else:
        try:
            mind_num = int(sys.argv[1])
        except:
            print("Usage: python test.py {mind_num}")
        print(run_brain(mind_num))
