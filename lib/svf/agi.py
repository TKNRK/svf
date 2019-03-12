from functools import partial
from scipy import optimize as opt
import numpy as np
from pathlib import PurePath, Path
from OpenGL.raw.GL._types import GL_UNSIGNED_INT, GL_UNSIGNED_SHORT

from sn.io.graph.load import Loader
from .profile import Profile as P


# AGI に必要なデータをロードし，出力する．
def loadgraph(pf=0.5):
    name = P.name
    dataset_dir = PurePath(P.dataset_dir)
    g = Loader(dataset_dir, name)
    # グラフの頂点数，辺数
    num_of_nodes, num_of_edges = g.size()
    num_of_edges *= 2
    # グラフの次元数
    dim_hd = g.dim_hd()[1]
    # 固有値
    eigen_values, _ = g.eigens()
    projections = gen_projection(eigen_values, dim_hd, projection_factor=pf)
    # 高次元座標
    layout_hd = g.layout_hd()
    # グラフのエッジリスト
    edgelist = np.load(str(dataset_dir.joinpath(name,"graph","edgelist.npy")))
    # グラフの隣接リスト
    adjacency = np.load(str(dataset_dir.joinpath(name,"graph","adjacency.npy")))
    if (num_of_nodes < 1 << 16):
        edges = np.array(edgelist, dtype=np.uint32).flatten()
        edgetype = GL_UNSIGNED_INT
    else:
        edges = np.array(edgelist, dtype=np.uint32).flatten()
        edgetype = GL_UNSIGNED_INT
    # グラフのラベル
    try:
        labels = g.attribute("label")
    except:
        labels = list(map(str, range(num_of_nodes)))
        # AS.labels = list(map(str,(g.attribute("id"))))
    ids = np.arange(num_of_nodes)
    return num_of_edges, projections, layout_hd, edges, edgetype, labels, ids, adjacency


# generate projection vectors
def gen_projection(eigen_values, dim_hd, projection_factor=0.5):
    L = np.diag(eigen_values[0:dim_hd])
    base = np.zeros(dim_hd * 2).reshape(dim_hd, 2)
    for i in range(dim_hd):
        base[i][i % 2] = 1
    Proj = np.c_[gram_schmidt(L.dot(base))]
    return Proj


def gram_schmidt(e_new):
    e1 = e_new[:, 0]
    e2 = e_new[:, 1]
    E1 = e1 / np.linalg.norm(e1)
    E2 = e2 - E1.dot(e2) * E1
    E2 /= np.linalg.norm(E2)
    return np.array([E1,E2]).T


# 制約解消のためのメソッドたち

arr_init = np.array([1, 0, 0, 0, 1, 0, 1, 1])
constraints = lambda x_pre,y_pre,x_new,y_new,p_norm,a1,b1,c1,a2,b2,c2,t,s: \
    ((s*(a2 + c2*x_pre) + t*(b2 + c2*y_pre - 1))**2 + (s*(a1 + c1*x_pre - 1) + t*(b1 + c1*y_pre))**2 +
     (s**2 + t**2 - 1)**2 + (a1*x_pre + b1*y_pre + c1*p_norm - x_new)**2 +
     (a2*x_pre + b2*y_pre + c2*p_norm - y_new)**2 +
     (a1*a2 + b1*b2 + c1*c2*p_norm + x_pre*(a1*c2 + a2*c1) + y_pre*(b1*c2 + b2*c1))**2 +
     (a1**2 + 2*a1*c1*x_pre + b1**2 + 2*b1*c1*y_pre + c1**2*p_norm - 1)**2 +
     (a2**2 + 2*a2*c2*x_pre + b2**2 + 2*b2*c2*y_pre + c2**2*p_norm - 1)**2)


def get_new_projections(picked_hd, x2, y2, projections):
    picked_ld = picked_hd.dot(projections)
    norm_picked_hd = picked_hd.dot(picked_hd)
    Proj = np.c_[projections, picked_hd]
    f2 = partial(constraints, picked_ld[0],
                 picked_ld[1], x2, y2, norm_picked_hd)

    def g(args): return f2(*args)
    res = opt.minimize(g, arr_init, method='L-BFGS-B')

    if (res.success and res.fun < 1e-2):
        # print(res)
        Coefficient = res.x[0:6].reshape(2, 3)
        Proj_new = gram_schmidt(Proj.dot(Coefficient.T))
        # 新しい低次元座標が制約を満たすかチェック
        hd_norm = np.linalg.norm(picked_hd)
        ld_norm = np.linalg.norm(picked_ld)
        new_ld_norm = np.linalg.norm(picked_hd.dot(Proj_new))
        if (hd_norm * 0.95 > ld_norm and hd_norm * 0.95 > new_ld_norm):
            return Proj_new
        else:
            return projections
    else:
        return projections


# 高次元座標上の近傍を求めるメソッド．
# def mkNeighbor(self, r=1.8):
#     # AS.neighbor.clear()
#     # AS.rst.clear()
#     for ind in range(AS.num_of_nodes):
#         thisPos = AS.layout_HD[ind, :]
#         neighbor = []
#         rst = []
#         for i in range(AS.num_of_nodes):
#             d = np.linalg.norm(AS.layout_HD[i, :] - thisPos)
#             if i != ind:
#                 if d < r:
#                     neighbor.append(i)
#                 else:
#                     rst.append(i)
#         # AS.neighbor.append(neighbor)
#         # AS.rst.append(rst)


if __name__ == '__main__':
    print(0)
