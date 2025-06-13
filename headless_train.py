from piece_maps import piece_maps, next_piece_ids
import psutil
import numpy as np
import gc
from cynes import NES
import time
from genetic_algorithm import Brain, sort_best, save_best, mating
import torch
import config as cfg
import statistics as st
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor as Pool
import multiprocessing

cfg.suppress_ctrl_c()

def initialize():
    # Create a new NES instance for each process
    nes = NES("roms/tetris.nes")
    while not nes[0x0048] or (nes[0x0058] and (nes[0x0048] == 10)):
        for i in range(4):
            nes.controller = NES_INPUT_START
            nes.step(frames=20)
            nes.controller = 0
        nes.step()
    return nes

def run_generation():
    scores = [0] * cfg.POPULATION_SIZE
    
    # Determine the number of workers to use
    if cfg.MAX_WORKERS <= 0:
        num_workers = min(multiprocessing.cpu_count(), cfg.POPULATION_SIZE)
    else:
        num_workers = min(cfg.MAX_WORKERS, cfg.POPULATION_SIZE)
        
    print(f"Using {num_workers} workers for parallel training")

    # Create a process pool and run the brains in parallel
    with Pool(max_workers=num_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        future_to_brain = {}

        for i in range(cfg.POPULATION_SIZE):
            future = executor.submit(run, i, initialize)
            future_to_brain[future] = i

        completed_futures = set()
        for future in concurrent.futures.as_completed(future_to_brain, timeout=600):
            if future in completed_futures:
                continue
            completed_futures.add(future)
            brain_index = future_to_brain[future]
            
            try:
                scores[brain_index] = future.result(timeout=120)
            except concurrent.futures.TimeoutError:
                print(f"Brain {brain_index} timed out after 120 seconds")
                scores[brain_index] = 0
            except Exception as e:
                print(f"Brain {brain_index} generated and exception: {e}")

    # Kill any remaining zombie processes
    print("Cleaning up any remaining processes...")
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                if 'python' in child.name().lower():
                    child.kill()

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except:
        pass
    
    return scores

def train():
    best_score = 0
    best_epoch = 0
    best_average = 0
    best_average_epoch = 0

    for epoch in range(cfg.EPOCHS):

        start_time = time.time()
        scores = run_generation()
        epoch_time = time.time() - start_time

        if max(scores) > best_score:
            best_score = round(max(scores), 3)
            best_epoch = epoch + 1

        if st.mean(scores) > best_average:
            best_average = round(st.mean(scores), 3)
            best_average_epoch = epoch + 1

        print(f'Best score: {max(scores)}')
        print(f'Average score: {st.mean(scores):.2f}')
        
        print(f'Epoch: {epoch + 1} - Best epoch: {best_epoch} - Best average epoch: {best_average_epoch}')
        print(f'Epoch completed in {epoch_time:.2f} seconds')

        print('Sorting by best scores...')
        start_time = time.time()
        best = sort_best(scores)
        sort_time = time.time() - start_time
        print(f'Sorted by best scores in {sort_time:.2f}')

        print('Saving best models...')
        start_time = time.time()
        save_best(best)
        save_time = time.time() - start_time
        print(f'Saved best models in {save_time:.2f}')
        
        print('Creating next generation...')
        start_time = time.time()
        mating()
        creation_time = time.time() - start_time
        print(f'Created new models in {creation_time:.2f}')
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

if __name__ == "__main__":
    preparation()
    print(f"Starting multi-threaded training with population size: {cfg.POPULATION_SIZE}")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    train()
