import torch
import torch.multiprocessing as mp

def worker(shared_counter, lock):
    with lock:
        shared_counter[0] += 1
        print("Counter:", shared_counter[0].item())

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    shared_counter = torch.zeros(1).share_memory_()
    lock = mp.Lock()

    processes = [
        mp.Process(target=worker, args=(shared_counter, lock))
        for _ in range(4)
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("Final Counter:", shared_counter.item())