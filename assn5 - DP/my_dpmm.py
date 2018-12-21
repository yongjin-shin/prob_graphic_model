import numpy as np
import pandas as pd
from scipy.special import psi, gammaln
from sklearn.cluster import KMeans


class my_dpmm:
    def __init__(self, data, alpha, base, t=10, n_iter=100):
        self.data = data
        self.N, self.D = data.shape
        self.alpha = alpha
        self.T = t
        self.G0 = base
        self.n_iter = n_iter
        self.phi = np.zeros(shape=(self.T, self.N))
        self.gamma = np.zeros(shape=(self.T-1, 2))
        self.tau = np.zeros(shape=(self.T, self.D))

    def initializing(self):
        print("Initializing")
        self.phi = np.random.uniform(size=(self.T, self.N))
        self.phi = np.divide(self.phi, np.sum(self.phi, axis=0))

    def m_step(self):
        print("Mstep")
        for iter in range(self.n_iter):

            # gamma_{k,1} and gamma_{k,2}
            self.gamma[:, 0] = 1 + np.sum(self.phi[:self.T-1, :], axis=1)
            phi_cum = np.cumsum(self.phi[:0:-1, :], axis=0)[::-1, :]
            self.gamma[:, 1] = self.alpha + np.sum(phi_cum, axis=1)

            # tau(atom)
            self.tau = self.G0 + np.matmul(self.phi, self.data)

            # phi
            log_v = psi(self.gamma[:, 0]) - psi(self.gamma[:, 0] + self.gamma[:, 1])
            log_1_v = psi(self.gamma[:, 1]) - psi(self.gamma[:, 0] + self.gamma[:, 1])
            log_v = np.vstack((np.reshape(log_v, newshape=(self.T-1, -1)), 0.))
            log_1_v = np.cumsum(np.vstack((0., np.reshape(log_1_v, newshape=(self.T-1, -1)))), axis=0)
            e_theta = psi(self.tau) - np.expand_dims(psi(np.sum(self.tau, axis=1)), axis=1)
            s_nt = log_v + log_1_v + np.matmul(e_theta, np.transpose(self.data))
            s_nt = s_nt - self.log_sum(s_nt, axis=0)
            self.phi = np.exp(s_nt)

    def solver(self):
        self.initializing()
        self.m_step()
        return self.gamma, self.tau, self.phi

    @staticmethod
    def log_sum(a, axis=None):
        a_max = np.max(a, axis=axis)
        try:
            return a_max + np.log(np.sum(np.exp(a - a_max), axis=axis))
        except:
            return a_max + np.log(np.sum(np.exp(a - a_max[:, np.newaxis]), axis=axis))


if __name__ == '__main__':
    df = pd.read_csv('iris.csv')
    dd = df.values
    gamma, tau, phi = my_dpmm(data=dd, base=np.ones(dd.shape[1]), alpha=1).solver()
    assigned_clusters = np.argmax(phi, axis=0)
    print(assigned_clusters)

