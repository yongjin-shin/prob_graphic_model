import numpy as np
from scipy.special import digamma
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt


class VIMOG():
    def __init__(self, data, k=3, beta_0=1, w_0=10, max_iter=100):
        self.x = data
        self.K = k
        self.D = data.ndim
        self.N = len(data)
        self.max_iter = max_iter
        # self.ELBO = np.zeros(self.max_iter)

        self.alpha_0 = 1/self.K
        self.beta_0 = beta_0
        self.m_0 = np.zeros(self.D)  # np.mean(self.x, axis=0)
        self.nu_0 = self.D + 10
        self.W_0 = np.eye(self.D, self.D) * w_0
        self.W_0_inv = np.linalg.inv(self.W_0)

        self.alpha = np.ones(self.K) * self.alpha_0  # Dirichlet Prior
        self.beta = np.ones(self.K) * self.beta_0  # scaling of precision matrix
        self.m = np.zeros((self.D, self.K))  # mean of gaussian
        self.nu = np.ones(self.K) * self.nu_0  # Wishart Distribution
        self.W = np.zeros((self.D, self.D, self.K))

        self.log_pi = np.zeros(self.K)  # expectation of pi
        self.log_lambda = np.zeros(self.K)  # expectation of lambda

        self.log_rho_nk = np.zeros((self.N, self.K))
        self.log_r_nk = np.zeros((self.N, self.K))  # log of responsibility
        self.r_nk = np.zeros((self.N, self.K))  # responsibility

        self.x_bar = np.zeros((self.D, self.K))
        self.Nk = np.zeros(self.K)
        self.S = np.zeros((self.D, self.D, self.K))

    def initial(self):
        kmeans = KMeans(n_clusters=self.K).fit(self.x)
        self.m = np.transpose(kmeans.cluster_centers_)

        alpha_hat = np.sum(self.alpha)
        for k in range(self.K):
            # self.m[:, k] = self.m_0
            self.W[:, :, k] = self.W_0
            self.log_pi[k] = digamma(self.alpha[k]) - digamma(alpha_hat)

            tmp = 0
            for i in range(self.D):
                tmp += digamma((self.nu[k] + 1 - i) / 2)
            self.log_lambda[k] = tmp + self.D * np.log(2) + np.log(np.linalg.det(self.W[:, :, k]))

        # print("Initializing Done")

    def estep(self):
        for k in range(self.K):
            diff = self.x - self.m[:, k]
            for n in range(self.N):
                exp_quad = self.D/(self.beta[k]) + np.dot(np.matmul(np.transpose(diff[n, :]), self.W[:, :, k]), diff[n, :]) * self.nu[k]
                self.log_rho_nk[n, k] = self.log_pi[k] + 0.5*self.log_lambda[k] - 0.5*self.D*np.log(2*np.pi) - 0.5*exp_quad

        for n in range(self.N):
            self.log_r_nk[n, :] = self.log_rho_nk[n, :] - self.log_sum(self.log_rho_nk[n, :])
            self.r_nk[n, :] = np.exp(self.log_r_nk[n, :])

        # print("Estep Done")

    def mstep(self):
        self.Nk = np.sum(self.r_nk, axis=0) + 1e-5
        for k in range(self.K):
            self.x_bar[:, k] = 1/self.Nk[k] * np.sum(self.r_nk[:, k].reshape((-1, 1))*self.x, axis=0)
            diff = self.x - self.x_bar[:, k]
            for n in range(self.N):
                array = diff[n, :].reshape((self.D, -1))
                self.S[:, :, k] += self.r_nk[n, k]*np.matmul(array, np.transpose(array))
            self.S[:, :, k] = self.S[:, :, k] * 1/self.Nk[k]

        self.alpha = self.alpha_0 + self.Nk
        self.beta = self.beta_0 + self.Nk
        self.nu = self.nu_0 + self.Nk

        for k in range(self.K):
            self.m[:, k] = 1/self.beta[k] * (self.beta_0*self.m_0 + self.Nk[k]*self.x_bar[:, k])
            diff2 = (self.x_bar[:, k] - self.m_0).reshape(self.D, -1)
            self.W[:, :, k] = self.W_0_inv + self.Nk[k]*self.S[:, :, k] \
                              + np.matmul(diff2, np.transpose(diff2))*self.beta_0*self.Nk[k]/(self.beta_0+self.Nk[k])
            self.W[:, :, k] = np.linalg.inv(self.W[:, :, k])

        alpha_hat = np.sum(self.alpha)
        for k in range(self.K):
            self.log_pi[k] = digamma(self.alpha[k]) - digamma(alpha_hat)

            tmp = 0
            for i in range(self.D):
                tmp += digamma((self.nu[k] + 1 - i) / 2)
            self.log_lambda[k] = tmp + self.D * np.log(2) + np.log(np.linalg.det(self.W[:, :, k]))

        # print("Mstep Done")

    def log_sum(self, rho):
        a = np.max(rho)
        y = a + np.log(np.sum(np.exp(rho - a)))
        return y

    def solver(self):
        self.initial()
        for num_it in range(self.max_iter):
            print('{} th iter'.format(num_it+1))
            self.estep()
            self.mstep()
            if (num_it+1) % 10 == 0:
                self.plot(num_it)
                self.bar_plot(num_it)
        print(self.m)
        print(self.Nk)

    def eigsorted(self, cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    def create_cov_ellipse(self, cov, pos, nstd=1, **kwargs):
        vals, vecs = self.eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
        return ellip

    def bar_plot(self, num_it):
        plt.bar(np.arange(self.K), self.Nk)
        plt.title('iteration: {}'.format(num_it))
        plt.savefig('bar_{}'.format(num_it))
        plt.close()

    def plot(self, num_it):
        x1 = self.x[:, 0]
        x2 = self.x[:, 1]
        plt.scatter(x1, x2)
        ells = []
        color = []
        for k in range(self.K):
            if self.Nk[k] > 0.5:
                ells.append(self.create_cov_ellipse(cov=np.linalg.inv(self.nu[k] * self.W[:, :, k]), pos=self.m[:, k]))
                color.append(self.Nk[k])

        color = np.array(color)
        color = color/np.sum(color)

        a = plt.subplot(111)
        for k, e in enumerate(ells):
            a.add_artist(e)
            e.set_facecolor((1-color[k], 1-color[k], 1-color[k]))

        a.set_xlim(np.min(x1)-np.min(x1)/10, np.max(x1)+np.min(x1)/10)
        a.set_ylim(np.min(x2)-np.min(x2)/10, np.max(x2)+np.min(x2)/10)
        plt.scatter(x1, x2)
        plt.title('iteration: {}'.format(num_it))
        plt.savefig('plot_{}'.format(num_it))
        plt.close()


if __name__ == "__main__":
    df = pd.read_csv('faithful.csv')
    dd = df.values
    my_mog = VIMOG(dd, k=6, beta_0=1, w_0=10, max_iter=61)
    my_mog.solver()

