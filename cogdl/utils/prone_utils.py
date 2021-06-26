import math
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.special import iv
from sklearn import preprocessing


class HeatKernel(object):
    def __init__(self, t=0.5, theta0=0.6, theta1=0.4):
        self.t = t
        self.theta0 = theta0
        self.theta1 = theta1

    def prop_adjacency(self, mx):
        mx_norm = preprocessing.normalize(mx.transpose(), "l1").transpose()
        adj = self.t * mx_norm
        adj.data = np.exp(adj.data)
        return adj / np.exp(self.t)

    def prop(self, mx, emb):
        adj = self.prop_adjacency(mx)
        return self.theta0 * emb + self.theta1 * adj.dot(emb)


class HeatKernelApproximation(object):
    def __init__(self, t=0.2, k=5):
        self.t = t
        self.k = k

    def taylor(self, mx, emb):
        mx_norm = preprocessing.normalize(mx, "l1")
        result = [math.exp(self.t) * emb]
        for i in range(self.k - 1):
            temp_mx = self.t * mx_norm.dot(result[-1]) / (i + 1)
            result.append(temp_mx)
        return sum(result)

    def chebyshev(self, mx, emb):
        mx = mx + sp.eye(emb.shape[0])
        mx = preprocessing.normalize(mx, "l1")
        conv = iv(0, self.t) * emb
        laplacian = sp.eye(emb.shape[0]) - mx
        Lx0 = emb
        Lx1 = laplacian.dot(emb)
        conv -= 2 * iv(1, self.t) * Lx1

        for i in range(2, self.k):
            Lx2 = 2 * laplacian.dot(Lx1) - Lx0
            conv += (-1) ** i * 2 * iv(i, self.t) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
        return conv

    def prop(self, mx, emb):
        return self.chebyshev(mx, emb)


class Gaussian(object):
    def __init__(self, mu=0.5, theta=1, rescale=False, k=3):
        self.theta = theta
        self.mu = mu
        self.k = k
        self.rescale = rescale
        self.coefs = [(-1) ** i * iv(i, self.theta) for i in range(k + 3)]
        self.coefs[0] = self.coefs[0] / 2

    # adj: 1 mul + 3 add,  emb: 2*k mul, 3*k add
    def prop(self, mx, emb):
        row_num, col_sum = mx.shape
        mx = mx + sp.eye(row_num)
        mx_norm = preprocessing.normalize(mx, "l1")
        mx_hat = (1 - self.mu) * sp.eye(row_num) - mx_norm

        Lx0 = emb
        Lx1 = mx_hat.dot(emb)
        Lx1 = 0.5 * mx_hat.dot(Lx1) - emb

        conv = iv(0, self.theta) * Lx0
        conv -= 2 * iv(1, self.theta) * Lx1
        for i in range(2, self.k):
            Lx2 = mx_hat.dot(Lx1)
            Lx2 = (mx_hat.dot(Lx2) - 2 * Lx1) - Lx0

            # Lx2 = 2 * mx_hat.dot(Lx1) - Lx0
            conv += (-1) ** i * 2 * iv(i, self.theta) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
        if self.rescale:
            conv = mx.dot(emb - conv)
        return conv


class PPR(object):
    """
    applying sparsification to accelerate computation
    """

    def __init__(self, alpha=0.5, k=10):
        self.alpha = alpha
        self.k = k
        self.alpha_list = [self.alpha * (1 - self.alpha) ** i for i in range(self.k)]
        self.epsilon = 1e-3

    def prop(self, mx, emb):
        mx_norm = preprocessing.normalize(mx, "l1")

        Lx = emb
        conv = self.alpha * Lx
        for i in range(1, self.k):
            Lx = (1 - self.alpha) * mx_norm.dot(Lx)
            conv += Lx
        return conv


class SignalRescaling(object):
    """
    - rescale signal of each node according to the degree of the node:
        - sigmoid(degree)
        - sigmoid(1/degree)
    """

    def __init__(self):
        pass

    def prop(self, mx, emb):
        mx = preprocessing.normalize(mx, "l1")
        degree = mx.sum(1).A.squeeze()

        degree_inv = 1.0 / degree
        signal_val = 1.0 / (1 + np.exp(-degree_inv))

        row_num, col_num = mx.shape
        q_ = sp.csc_matrix((signal_val, (np.arange(row_num), np.arange(col_num))), shape=(row_num, col_num))

        adj_norm = mx.dot(q_)
        adj_norm = preprocessing.normalize(adj_norm, "l1")
        conv = adj_norm.dot(emb)
        return conv


class ProNE(object):
    def __call__(self, A, a, order=10, mu=0.1, s=0.5):
        # NE Enhancement via Spectral Propagation
        print("Chebyshev Series -----------------")

        if order == 1:
            return a

        node_number = a.shape[0]

        A = sp.eye(node_number) + A
        DA = preprocessing.normalize(A, norm="l1")
        L = sp.eye(node_number) - DA

        M = L - mu * sp.eye(node_number)

        Lx0 = a
        Lx1 = M.dot(a)
        Lx1 = 0.5 * M.dot(Lx1) - a

        conv = iv(0, s) * Lx0
        conv -= 2 * iv(1, s) * Lx1
        for i in range(2, order):
            Lx2 = M.dot(Lx1)
            Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
            #         Lx2 = 2*L.dot(Lx1) - Lx0
            if i % 2 == 0:
                conv += 2 * iv(i, s) * Lx2
            else:
                conv -= 2 * iv(i, s) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
            del Lx2
        mm = A.dot(a - conv)
        return mm


class NodeAdaptiveEncoder(object):
    """
    - shrink negative values in signal/feature matrix
    - no learning
    """

    @staticmethod
    def prop(signal):
        mean_signal = signal.mean(1)
        mean_signal = 1.0 / (1 + np.exp(-mean_signal))
        sel_row, sel_col = np.where(signal < 0)
        mean_signal = mean_signal[sel_row]
        signal[sel_row, sel_col] = signal[sel_row, sel_col] * mean_signal
        return signal


def propagate(mx, emb, stype, space=None):
    if space is not None:
        if stype == "heat":
            heat_kernel = HeatKernelApproximation(t=space["t"])
            result = heat_kernel.prop(mx, emb)
        elif stype == "ppr":
            ppr = PPR(alpha=space["alpha"])
            result = ppr.prop(mx, emb)
        elif stype == "gaussian":
            gaussian = Gaussian(mu=space["mu"], theta=space["theta"])
            result = gaussian.prop(mx, emb)
        elif stype == "sc":
            signal_rs = SignalRescaling()
            result = signal_rs.prop(mx, emb)
        else:
            raise ValueError("please use filter in ['heat', 'ppr', 'gaussian', 'sc'], currently use {}".format(stype))
    else:
        if stype == "heat":
            heat_kernel = HeatKernelApproximation()
            result = heat_kernel.prop(mx, emb)
        elif stype == "ppr":
            ppr = PPR()
            result = ppr.prop(mx, emb)
        elif stype == "gaussian":
            gaussian = Gaussian()
            result = gaussian.prop(mx, emb)
        elif stype == "sc":
            signal_rs = SignalRescaling()
            result = signal_rs.prop(mx, emb)
        elif stype == "prone":
            signal_pro = ProNE()
            result = signal_pro(mx, emb)
        else:
            raise ValueError("please use filter in ['heat', 'ppr', 'gaussian', 'sc'], currently use {}".format(stype))
    return result


def get_embedding_dense(matrix, dimension):
    # get dense embedding via SVD
    U, s, Vh = scipy.linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
    U = np.array(U)
    U = U[:, :dimension]
    s = s[:dimension]
    s = np.sqrt(s)
    U = U * s
    U = preprocessing.normalize(U, "l2")
    return U
