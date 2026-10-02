"""モデル2解析の計算を行い results/model2/ に保存する
   使い方: python3 src/model2_compute.py [grid|learn|all] [シード数(既定 1)]"""
import os, sys, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from sim import DEFAULT
from model2_tools import simplex_grid, focal_run, learn_run

OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'model2')
os.makedirs(OUT, exist_ok=True)
HORIZONS = [50, 100, 200, 300, 600]   # 開始からの経過ステップ
TAUS = [20, 50, 100]                  # 雑音を測る評価窓 (定常区間が300ステップなので100まで)
BURN = 300                            # 定常とみなす開始ステップ

def summarize(r):
    d = r['deg']; st = d[BURN:]
    noise = [st[: len(st) // t * t].reshape(-1, t).mean(1).std() for t in TAUS]
    return [st.mean()] + [d[:h].mean() for h in HORIZONS] + noise + [r['pr_pct'], r['deaths'], r['pop_deg']]

def run_grid(n_seed=1, step=0.05, T=600):
    """注目ノードの配分を単体上の格子で総当たり (他の人は固定)"""
    g = simplex_grid(step); p = dict(DEFAULT, T=T)
    jobs = [(x, s, p) for x in g for s in range(n_seed)]
    with ProcessPoolExecutor(mp_context=mp.get_context('fork')) as ex:
        rows = [summarize(r) for r in ex.map(focal_run, jobs, chunksize=2)]
    A = np.array(rows).reshape(len(g), n_seed, -1)
    nh, nt = len(HORIZONS), len(TAUS)
    np.savez(f'{OUT}/grid.npz', grid=g, steady=A[..., 0], horizon=A[..., 1:1 + nh],
             noise=A[..., 1 + nh:1 + nh + nt], pr_pct=A[..., -3], deaths=A[..., -2],
             pop_deg=A[..., -1], horizons=HORIZONS, taus=TAUS)

# 学習の設定: (評価窓 tau, 一歩の大きさ eta) と初期値
LEARN_CONFIGS = [(50, 0.05), (50, 0.15), (200, 0.05), (200, 0.15)]
STARTS = {'center': (1/3, 1/3, 1/3), 'outreach': (0.15, 0.05, 0.80), 'research': (0.20, 0.75, 0.05)}

def run_learn(n_seed=1, T=6000):
    jobs, keys = [], []
    for tau, eta in LEARN_CONFIGS:
        for name, x0 in STARTS.items():
            for s in range(n_seed):
                jobs.append((x0, 1000 + s, dict(DEFAULT, T=T, tau=tau, eta=eta)))
                keys.append(f'{tau}_{eta}_{name}')
    with ProcessPoolExecutor(mp_context=mp.get_context('fork')) as ex:
        res = list(ex.map(learn_run, jobs))
    out = {}
    for k, (traj, deg) in zip(keys, res):
        out.setdefault(k + '_traj', []).append(traj)
        out.setdefault(k + '_deg', []).append(deg.reshape(-1, 100).mean(1))   # 100ステップ平均
    np.savez(f'{OUT}/learn.npz', **{k: np.array(v) for k, v in out.items()})

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    n_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if what in ('grid', 'all'):
        t = time.time(); run_grid(n_seed); print('grid done', round(time.time() - t), 's', flush=True)
    if what in ('learn', 'all'):
        t = time.time(); run_learn(n_seed); print('learn done', round(time.time() - t), 's', flush=True)
