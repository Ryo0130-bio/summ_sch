"""各モデルの最終状態のネットワークを描く
   ノードの大きさ = 次数 / 色 = 最も配分が大きい戦略 (赤=知り合い, 緑=研究, 青=アウトリーチ)
   次数上位 5 人は黒枠で強調する
   使い方: python3 src/plot_network.py [seed]"""
import sys, os
import numpy as np
import networkx as nx
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from sim import DEFAULT, Sim, model2, model3

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUT, exist_ok=True)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
p = dict(DEFAULT)
COLORS = ['#e6550d', '#31a354', '#3182bd']   # 知り合い / 研究 / アウトリーチ

def graph_of(sim):
    G = nx.Graph(); G.add_nodes_from(range(sim.p['N']))
    G.add_edges_from((i, j) for i in range(sim.p['N']) for j in sim.adj[i] if i < j)
    return G

def draw(ax, sim, title, focal=None):
    G = graph_of(sim)
    pos = nx.spring_layout(G, seed=0, k=1.2 / np.sqrt(len(G)), iterations=200)
    deg = np.array([G.degree(i) for i in G])
    col = [COLORS[k] for k in sim.x.argmax(1)]          # 配分が最大の戦略で色分け
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, alpha=0.5, edge_color='#444444')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=8 + 14 * deg, node_color=col,
                           linewidths=0.3, edgecolors='k', alpha=0.55)
    top = np.argsort(-deg)[:5]
    ax.scatter(*np.array([pos[i] for i in top]).T, s=8 + 14 * deg[top] + 60, facecolors='none',
               edgecolors='k', linewidths=1.2, zorder=4)
    if focal is not None:
        ax.scatter(*pos[focal], s=8 + 14 * deg[focal] + 250, facecolors='none',
                   edgecolors='k', linewidths=2, zorder=5)
    ax.set_title(f'{title}\nmean degree {deg.mean():.1f}, max {deg.max()}, '
                 f'top-5 mean alloc {np.round(sim.x[top].mean(0), 2).tolist()}', fontsize=9)
    ax.axis('off')

# model 1
rng = np.random.default_rng(seed)
sim1 = Sim(rng.dirichlet([1, 1, 1], size=p['N']), np.zeros(p['N'], bool), p, rng); sim1.run()
# model 2 / 3 (学習があるので長く回す)
pl = dict(p, T=3000)
_, sim2 = model2(pl, 100 + seed)
_, sim3 = model3(dict(p, T=12000), 200 + seed)   # 模倣学習は長く回す

fig, axs = plt.subplots(1, 3, figsize=(18, 6.5))
draw(axs[0], sim1, 'Model 1: fixed random allocation')
draw(axs[1], sim2, 'Model 2: one learner (circled)', focal=0)
draw(axs[2], sim3, 'Model 3: everyone learns')
# 凡例: 混色の見方
from matplotlib.lines import Line2D
leg = [Line2D([], [], marker='o', ls='', mfc=c, mec='k', label=n)
       for c, n in zip(COLORS, ('acquaintance', 'research', 'outreach'))]
fig.legend(handles=leg, loc='lower center', ncol=3,
           title='node color = dominant strategy, size = degree, black ring = top-5 degree')
plt.tight_layout(rect=(0, 0.07, 1, 1))
plt.savefig(f'{OUT}/final_networks.png', dpi=150); plt.close()
print('saved', f'{OUT}/final_networks.png')
