"""3モデルを実行して results/ に図と数値を保存する
   使い方: python3 src/run_experiments.py [--quick]"""
import sys, os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from concurrent.futures import ProcessPoolExecutor
from sim import DEFAULT, model1, model2, model3, power

T_LEARN = 3000   # model 2, 3 は学習に時間がかかるので長く回す
def _m1(s): return model1(DEFAULT, s)
def _m2(s): return model2(dict(DEFAULT, T=T_LEARN), 100 + s)[0]
def _m3(s):
    hist, sim = model3(dict(DEFAULT, T=T_LEARN), 200 + s); return hist.mean(1), sim.x

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUT, exist_ok=True)
quick = '--quick' in sys.argv
p = dict(DEFAULT)

def tern(x):  # 単体 (x1,x2,x3) -> 正三角形の 2D 座標
    return x[:, 1] + 0.5 * x[:, 2], np.sqrt(3) / 2 * x[:, 2]

def tri_axes(ax, title):
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=0.8)
    ax.text(-0.05, -0.05, 'r1 acquaint.', ha='right'); ax.text(1.05, -0.05, 'r2 research')
    ax.text(0.5, 0.9, 'r3 outreach', ha='center'); ax.set_title(title)
    ax.set_aspect('equal'); ax.axis('off')

summary = {}

# ---- 関数形の図 ----
S = np.linspace(0, p['R'] / p['delta'], 200)
f = power(np.stack([S, S, S], 1), p)
plt.figure(figsize=(5, 3.5))
for k, n in enumerate(['r1 acquaintance (saturating)', 'r2 research (linear)', 'r3 outreach (threshold)']):
    plt.plot(S, f[:, k], label=n)
plt.xlabel('cumulative investment stock S'); plt.ylabel('expected new links / step')
plt.legend(); plt.tight_layout(); plt.savefig(f'{OUT}/power_functions.png', dpi=150); plt.close()

# ---- model 1 ----
seeds = range(4 if quick else 20)
X, D, PR, B, DT = [], [], [], [], []
if __name__ != '__main__': raise SystemExit
ex = ProcessPoolExecutor()
for r in ex.map(_m1, seeds):
    # 回ごとの差を消すため、各回の中で順位パーセンタイルに変換する
    X.append(r['x']); D.append(r['deg'].argsort().argsort() / (p['N'] - 1))
    PR.append(r['pr'].argsort().argsort() / (p['N'] - 1))
    B.append(r['btw'].argsort().argsort() / (p['N'] - 1)); DT.append(r['deaths'])
X, D, PR, B, DT = map(np.concatenate, (X, D, PR, B, DT))
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, v, n in zip(axs, (D, PR, B), ('degree', 'PageRank', 'betweenness')):
    u, w = tern(X)
    hb = ax.hexbin(u, w, C=v, gridsize=14, cmap='viridis', reduce_C_function=np.mean, vmin=0.2, vmax=0.8)
    tri_axes(ax, f'Model 1: mean rank percentile ({n})'); fig.colorbar(hb, ax=ax, shrink=0.7)
plt.tight_layout(); plt.savefig(f'{OUT}/model1_landscape.png', dpi=150); plt.close()
# 各中心性で最良の領域 (上位 5% の平均配分)
for v, n in ((D, 'degree'), (PR, 'pagerank'), (B, 'betweenness')):
    top = v >= np.quantile(v, 0.95)
    summary[f'model1_top5pct_mean_alloc_{n}'] = X[top].mean(0).round(3).tolist()
summary['model1_regression_degree_on_alloc'] = np.linalg.lstsq(
    np.c_[X[:, :2], X[:, :2] ** 2, X[:, 0] * X[:, 1], np.ones(len(X))], D, rcond=None)[0].round(3).tolist()
summary['model1_death_rate_by_dominant'] = [float(DT[X.argmax(1) == k].mean()) for k in range(3)]

# ---- model 2 ----
fig, ax = plt.subplots(figsize=(5, 4.5)); tri_axes(ax, 'Model 2: focal learner trajectories')
finals = []
for traj in ex.map(_m2, range(4 if quick else 16)):
    u, w = tern(traj); ax.plot(u, w, '-', lw=0.8, alpha=0.7); ax.plot(u[-1], w[-1], 'ko', ms=3)
    finals.append(traj[-1])
ax.plot(*tern(np.array([[1/3, 1/3, 1/3]])), 'r*', ms=12)
plt.tight_layout(); plt.savefig(f'{OUT}/model2_trajectories.png', dpi=150); plt.close()
summary['model2_final_alloc_mean'] = np.mean(finals, 0).round(3).tolist()
summary['model2_final_alloc_each'] = np.round(finals, 3).tolist()

# ---- model 3 ----
H, Xf = [], []
for h, x in ex.map(_m3, range(2 if quick else 8)):
    H.append(h); Xf.append(x)
H = np.mean(H, 0); Xf = np.concatenate(Xf)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
t = np.arange(len(H)) * p['tau']
for k, n in enumerate(['r1', 'r2', 'r3']):
    axs[0].plot(t, H[:, k], label=n)
axs[0].set_xlabel('step'); axs[0].set_ylabel('population mean allocation'); axs[0].legend()
axs[0].set_title('Model 3: everyone learns')
tri_axes(axs[1], 'Model 3: final allocations'); axs[1].plot(*tern(Xf), 'k.', ms=2, alpha=0.4)
plt.tight_layout(); plt.savefig(f'{OUT}/model3_population.png', dpi=150); plt.close()
summary['model3_final_mean_alloc'] = Xf.mean(0).round(3).tolist()

json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1))
