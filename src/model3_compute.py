"""モデル3解析の計算: 多数シードを記録付きで回し results/model3/ に保存する
   使い方: python3 src/model3_compute.py [シード数(既定 16)] [ステップ数(既定 12000)]"""
import os, sys, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from sim import DEFAULT
from model3_tools import track_run

OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'model3')
os.makedirs(OUT, exist_ok=True)

if __name__ == '__main__':
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    t0 = time.time()
    p = dict(DEFAULT, T=T)
    with ProcessPoolExecutor(mp_context=mp.get_context('fork')) as ex:
        for s, r in enumerate(ex.map(track_run, [(400 + s, p) for s in range(n_seed)])):
            np.savez_compressed(f'{OUT}/run_{s:02d}.npz', **r)
            print('seed', s, 'saved', round(time.time() - t0), 's', flush=True)
    print('done', round(time.time() - t0), 's')
