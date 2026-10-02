"""モデル2(自分だけ学習)の最適戦略を解析するための道具
   - focal_run : 注目ノード0の配分を固定して回し、次数の時系列を返す
   - learn_run : 注目ノード0だけが学習する(モデル2そのもの)。軌跡と次数を返す
   - mean_field_power : 定常状態で自分が張るリンク数の理論値
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sim import DEFAULT, Sim, power

def simplex_grid(step=0.1):
    n = int(round(1 / step))
    return np.array([(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)], float) / n

def _setup(x0, seed, p, learner):
    rng = np.random.default_rng(seed)
    alloc = rng.dirichlet([1, 1, 1], size=p['N'])   # 他の人の配分 (seed が同じなら同じ環境)
    alloc[0] = x0
    L = np.zeros(p['N'], bool); L[0] = learner
    return Sim(alloc, L, p, rng)

def focal_run(args):
    """配分 x を固定した注目ノードの成績"""
    x, seed, p = args
    sim = _setup(x, seed, p, False)
    deg = np.empty(p['T'])
    for t in range(p['T']):
        sim.step(); deg[t] = len(sim.adj[0])
    pr = sim.centrality('pagerank')
    return dict(deg=deg, pr_pct=float((pr < pr[0]).mean()), deaths=int(sim.deaths[0]),
                pop_deg=float(sim.degree().mean()))

def learn_run(args):
    """モデル2: 注目ノードだけが学習する。sim.Sim.run と同じ学習を毎 tau ステップ行う"""
    x0, seed, p = args
    sim = _setup(np.asarray(x0, float), seed, p, True)
    deg = np.empty(p['T']); traj = [sim.x[0].copy()]
    for t in range(p['T']):
        sim.step(); d = sim.degree(); deg[t] = d[0]
        sim.win += d / (p['N'] - 1)
        if (t + 1) % p['tau'] == 0:
            sim.learn_step(sim.win / p['tau']); sim.win[:] = 0
            traj.append(sim.x[0].copy())
    return np.array(traj), deg

def mean_field_power(x, p=DEFAULT):
    """定常状態のストック S_k = R x_k / delta での張る力の合計 (候補不足や被リンクは無視)"""
    x = np.atleast_2d(x)
    return power(p['R'] * x / p['delta'], p).sum(1)
