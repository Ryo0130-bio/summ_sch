"""
学会で有名になる戦略: ネットワーク・シミュレーション参照実装
  戦略 k=1 知り合い / k=2 研究 / k=3 アウトリーチ
  有名さ = グラフ中心性
  model 1: 配分ランダム固定 / model 2: 自分だけ学習 / model 3: 全員学習
"""
import numpy as np
import networkx as nx

# ---------------- パラメータ ----------------
DEFAULT = dict(
    N=200,        # 研究者数
    K=5,          # 学術ラベル数
    R=1.0,        # 1ステップあたりの総投資量 r1+r2+r3
    T=600,        # ステップ数
    mu=0.02,      # 各リンクが1ステップで切れる確率
    delta=0.05,   # 投資ストックの減衰率 (0 なら純粋な累積)
    a1=0.08, K1=2.0,          # 知り合い: 飽和型  f1 = a1 S/(S+K1)
    a2=0.004,                 # 研究: 線形       f2 = a2 S
    a3=0.12, K3=14.0, n3=6,   # アウトリーチ: 閾値型 f3 = a3 S^n/(S^n+K3^n)
    m0=2,         # 初期に同ラベルと張るリンク数
    replace=True, # 死んだら同じ戦略で新人として再生する
    tau=50,       # 学習の評価窓 (ステップ)。反応の遅れ 1/delta+1/mu より長くする
    eta=0.05,     # 学習の1回の移動量 (単体上)
    measure='degree',  # 学習で使う中心性: degree / pagerank / eigenvector
    beta=0.3,     # 比較基準(過去の成績)の指数移動平均の重み。1 なら「直前の窓」とだけ比較
    # ---- 学習の方式 ----
    learning='hill',   # hill: 自分の試行錯誤 (モデル2) / imitation: 隣人の模倣 (モデル3)
    imit_K=0.5,        # フェルミ則の温度 (次数の単位)。小さいほど成績の差に敏感
    mut_rate=0.1,      # 評価のたびに配分が突然変異する確率
    mut_sigma=0.15,    # 突然変異の大きさ (単体上の移動量の標準偏差)
)

def power(S, p):
    """累積投資ストック S (N,3) -> 1ステップに張れるリンク数の期待値 (N,3)"""
    f = np.empty_like(S)
    f[:, 0] = p['a1'] * S[:, 0] / (S[:, 0] + p['K1'])
    f[:, 1] = p['a2'] * S[:, 1]
    s3 = S[:, 2] ** p['n3']
    f[:, 2] = p['a3'] * s3 / (s3 + p['K3'] ** p['n3'])
    return f

def project_simplex(x, eps=1e-3):
    x = np.clip(x, eps, None)
    return x / x.sum()

class Sim:
    def __init__(self, alloc, learners, p, rng):
        self.p, self.rng = p, rng
        N = p['N']
        self.x = alloc.copy()                 # 配分の割合 (N,3), 行和 1
        self.learn = learners.copy()          # 学習するか (N,) bool
        self.label = rng.integers(p['K'], size=N)
        self.S = np.zeros((N, 3))             # 累積投資ストック
        self.adj = [set() for _ in range(N)]
        self.deaths = np.zeros(N, int)
        # 学習の状態
        self.dir = self._rand_dir(N)
        self.prev_pay = np.full(N, -np.inf)
        self.prev_x = self.x.copy()
        self.win = np.zeros(N)                # 窓内の中心性の和
        for i in range(N):
            self._attach_same_label(i)

    # ---------- 基本操作 ----------
    def _rand_dir(self, n):
        d = self.rng.normal(size=(n, 3))
        d -= d.mean(axis=1, keepdims=True)    # 和 0 の方向 = 単体上の移動
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def _link(self, i, j):
        self.adj[i].add(j); self.adj[j].add(i)

    def _attach_same_label(self, i):
        pool = np.flatnonzero((self.label == self.label[i]) & (np.arange(self.p['N']) != i))
        for j in self.rng.choice(pool, size=min(self.p['m0'], len(pool)), replace=False):
            self._link(i, j)

    def _candidates(self, i, k):
        nb = self.adj[i]
        if k == 0:     # 知り合いの知り合い
            c = set().union(*[self.adj[j] for j in nb]) if nb else set()
        elif k == 1:   # 同じ学術ラベル
            c = set(np.flatnonzero(self.label == self.label[i]).tolist())
        else:          # 完全ランダム
            c = set(range(self.p['N']))
        c -= nb; c.discard(i)
        return list(c)

    # ---------- 1ステップ ----------
    def step(self):
        p, rng, N = self.p, self.rng, self.p['N']
        # 1. 投資 → ストック更新
        self.S = (1 - p['delta']) * self.S + p['R'] * self.x
        lam = power(self.S, p)
        # 2. リンク生成 (ランダム順)
        for i in rng.permutation(N):
            for k in range(3):
                m = rng.poisson(lam[i, k])
                if m == 0:
                    continue
                c = self._candidates(i, k)
                for j in rng.choice(c, size=min(m, len(c)), replace=False) if c else []:
                    self._link(i, j)
        # 3. リンク切断 (各リンク確率 mu)
        for i in range(N):
            for j in [j for j in self.adj[i] if j > i and rng.random() < p['mu']]:
                self.adj[i].discard(j); self.adj[j].discard(i)
        # 4. 死亡判定: 次数 0
        for i in range(N):
            if not self.adj[i]:
                self.deaths[i] += 1
                if p['replace']:
                    self.S[i] = 0.0
                    self._attach_same_label(i)

    def degree(self):
        return np.array([len(a) for a in self.adj], float)

    def centrality(self, kind):
        if kind == 'degree':
            return self.degree() / (self.p['N'] - 1)
        G = nx.Graph(); G.add_nodes_from(range(self.p['N']))
        G.add_edges_from((i, j) for i in range(self.p['N']) for j in self.adj[i] if i < j)
        if kind == 'pagerank':
            c = nx.pagerank(G)
        elif kind == 'eigenvector':
            c = nx.eigenvector_centrality_numpy(G)
        elif kind == 'betweenness':
            c = nx.betweenness_centrality(G)
        elif kind == 'closeness':
            c = nx.closeness_centrality(G)
        return np.array([c[i] for i in range(self.p['N'])])

    # ---------- 学習 (win-stay / lose-shift の山登り) ----------
    def learn_step(self, pay):
        L = self.learn
        worse = L & (pay < self.prev_pay)
        # 悪化: 前の配分に戻して新しい方向を引く / 改善: 同じ方向に進み続ける
        self.x[worse] = self.prev_x[worse]
        self.dir[worse] = self._rand_dir(worse.sum())
        # 比較基準 = 過去の成績の指数移動平均 (1回の幸運で固まらないようにする)
        b = self.p['beta']
        first = L & ~np.isfinite(self.prev_pay)
        self.prev_pay[first] = pay[first]
        self.prev_pay[L & ~first] = (1 - b) * self.prev_pay[L & ~first] + b * pay[L & ~first]
        # 学習者は今の配分を記録してから方向に一歩進む
        self.prev_x[L] = self.x[L]
        newx = self.x[L] + self.p['eta'] * self.dir[L]
        self.x[L] = np.apply_along_axis(project_simplex, 1, newx)

    # ---------- 模倣学習 (フェルミ則 + 突然変異) ----------
    def imitate_step(self, pay):
        """各学習者が隣人を1人ランダムに選び、確率 1/(1+exp(-(P_j-P_i)/K)) でその配分をまねる。
        全員が同時に、更新前の配分を見て判断する。その後、確率 mut_rate で配分を少しずらす。"""
        p, rng, N = self.p, self.rng, self.p['N']
        old = self.x.copy()
        scale = N - 1                                 # pay は次数/(N-1) なので次数の単位に戻す
        for i in np.flatnonzero(self.learn):
            nb = self.adj[i]
            if not nb:
                continue
            j = rng.choice(list(nb))
            dP = (pay[j] - pay[i]) * scale
            if rng.random() < 1 / (1 + np.exp(-dP / p['imit_K'])):
                self.x[i] = old[j]
                self._on_copy(i, j)
        mut = self.learn & (rng.random(N) < p['mut_rate'])
        if mut.any():
            d = self._rand_dir(mut.sum()) * np.abs(rng.normal(0, p['mut_sigma'], (mut.sum(), 1)))
            self.x[mut] = np.apply_along_axis(project_simplex, 1, self.x[mut] + d)
            self._on_mutate(np.flatnonzero(mut))

    # 解析用のフック (戦略の系譜を記録したいサブクラスが上書きする)
    def _on_copy(self, i, j):
        pass

    def _on_mutate(self, idx):
        pass

    def run(self, record_every=None):
        p = self.p
        hist = []
        for t in range(p['T']):
            self.step()
            if p['measure'] == 'degree':
                self.win += self.centrality('degree')
            if self.learn.any() and (t + 1) % p['tau'] == 0:
                pay = self.win / p['tau'] if p['measure'] == 'degree' else self.centrality(p['measure'])
                if p.get('learning', 'hill') == 'imitation':
                    self.imitate_step(pay)
                else:
                    self.learn_step(pay)
                self.win[:] = 0
            if record_every and t % record_every == 0:
                hist.append(self.x.copy())
        return hist

# ---------------- 3つのモデル ----------------
def model1(p, seed):
    """全員ランダム配分で固定"""
    rng = np.random.default_rng(seed)
    alloc = rng.dirichlet([1, 1, 1], size=p['N'])
    sim = Sim(alloc, np.zeros(p['N'], bool), p, rng)
    burn = p['T'] - 100
    p1 = dict(p, T=burn); sim.p = p1; sim.run()
    acc = np.zeros(p['N'])
    sim.p = dict(p, T=100); 
    for _ in range(100):
        sim.step(); acc += sim.degree()
    return dict(x=sim.x, deg=acc / 100, deaths=sim.deaths,
                pr=sim.centrality('pagerank'), btw=sim.centrality('betweenness'))

def model2(p, seed, x0=(1/3, 1/3, 1/3)):
    """ノード0だけ学習, 他はランダム配分で固定"""
    rng = np.random.default_rng(seed)
    alloc = rng.dirichlet([1, 1, 1], size=p['N'])
    alloc[0] = x0
    L = np.zeros(p['N'], bool); L[0] = True
    sim = Sim(alloc, L, p, rng)
    hist = sim.run(record_every=p['tau'])
    return np.array([h[0] for h in hist]), sim

def model3(p, seed, x0=(1/3, 1/3, 1/3)):
    """全員が同じ初期配分から、隣人の模倣で学習する"""
    p = dict(p, learning='imitation')
    rng = np.random.default_rng(seed)
    alloc = np.tile(x0, (p['N'], 1))
    sim = Sim(alloc, np.ones(p['N'], bool), p, rng)
    hist = sim.run(record_every=p['tau'])
    return np.array(hist), sim
