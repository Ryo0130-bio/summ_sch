"""モデル3(模倣学習)の解析用: 配分・次数・戦略の系譜を時系列で記録する"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sim import DEFAULT, Sim

class TrackSim(Sim):
    """戦略に ID を付けて系譜を記録する Sim。
    模倣では相手の ID をコピーし、突然変異では親 ID を持つ新しい ID を作る。
    初期は全員が同じ配分なので、全員が ID 0 (根) から始まる。"""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        N = self.p['N']
        self.sid = np.zeros(N, np.int32)
        # 系譜表: parent, 誕生時刻, 誕生したノード, 配分
        self.s_parent, self.s_time, self.s_node, self.s_x = [-1], [0], [-1], [self.x[0].copy()]
        self.t = 0

    def _on_copy(self, i, j):
        # 模倣は更新前の配分を見て同時に行うので、j の更新前の ID を使う
        self.sid[i] = self._old_sid[j]

    def _on_mutate(self, idx):
        for i in idx:
            self.s_parent.append(int(self.sid[i])); self.s_time.append(self.t)
            self.s_node.append(int(i)); self.s_x.append(self.x[i].copy())
            self.sid[i] = len(self.s_parent) - 1

    def imitate_step(self, pay):
        self._old_sid = self.sid.copy()
        super().imitate_step(pay)

def track_run(args):
    """1 シード分を回し、評価窓ごとに 配分・窓平均次数・戦略 ID を記録する"""
    seed, p = args
    p = dict(p, learning='imitation')
    rng = np.random.default_rng(seed)
    N, tau = p['N'], p['tau']
    sim = TrackSim(np.tile([1/3, 1/3, 1/3], (N, 1)), np.ones(N, bool), p, rng)
    X, D, SID, DEATH = [sim.x.astype(np.float16)], [sim.degree().astype(np.float16)], [sim.sid.copy()], []
    for t in range(p['T']):
        sim.t = t + 1
        sim.step()
        d = sim.degree(); sim.win += d / (N - 1)
        if (t + 1) % tau == 0:
            pay = sim.win / tau
            D.append((pay * (N - 1)).astype(np.float16))   # この窓の平均次数 (模倣の判断に使った値)
            sim.imitate_step(pay); sim.win[:] = 0
            X.append(sim.x.astype(np.float16)); SID.append(sim.sid.copy())
            DEATH.append(sim.deaths.copy())
    return dict(X=np.array(X), D=np.array(D), SID=np.array(SID), deaths=np.array(DEATH, np.int16),
                label=sim.label, s_parent=np.array(sim.s_parent, np.int32),
                s_time=np.array(sim.s_time, np.int32), s_node=np.array(sim.s_node, np.int32),
                s_x=np.array(sim.s_x, np.float32),
                edges=np.array([(i, j) for i in range(N) for j in sim.adj[i] if i < j], np.int16))

def lineage(s_parent, sid):
    """戦略 sid から根までの祖先の ID 列 (根が先頭)"""
    chain = [int(sid)]
    while s_parent[chain[-1]] >= 0:
        chain.append(int(s_parent[chain[-1]]))
    return chain[::-1]
