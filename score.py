import utils.config as cfg
from utils.run import run
from headless_train import run_generation, initialize

cfg.suppress_ctrl_c()

def run_brain(mind_num):
    score = run(mind_num, initialize)
    return(score)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(run_generation())
    else:
        try:
            mind_num = int(sys.argv[1])
        except:
            print("Usage: python test.py {mind_num}")
        print(run_brain(mind_num, initialize)
