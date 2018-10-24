import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import random
np.random.seed(7)


def initialization_kmeans(X, p, q, variance_level=None):
    """
    X : dataset
    p : number of clusters
    q : dimension of the latent space
    variance_level
    pi : proportions of clusters
    mu : centers of the clusters in the observation space
    W : latent to observation matricies
    sigma2 : noise
    """

    N, d = X.shape

    # initialization
    init_centers = np.random.randint(0, N, p)
    while (len(np.unique(init_centers)) != p):
        init_centers = np.random.randint(0, N, p)

    mu = X[init_centers, :]
    distance_square = np.zeros((N, p))
    clusters = np.zeros(N, dtype=np.int32)

    D_old = -2
    D = -1

    while(D_old != D):
        D_old = D

        # assign clusters
        for c in range(p):
            distance_square[:, c] = np.power(X - mu[c, :], 2).sum(1)
        clusters = np.argmin(distance_square, axis=1)

        # compute distortion
        distmin = distance_square[range(N), clusters]
        D = distmin.sum()

        # compute new centers
        for c in range(p):
            mu[c, :] = X[clusters == c, :].mean(0)

    #for c in range(p):
    #    plt.scatter(X[clusters == c, 0], X[clusters == c, 1], c=np.random.rand(3,1))

    # parameter initialization
    pi = np.zeros(p)
    W = np.zeros((p, d, q))
    sigma2 = np.zeros(p)
    for c in range(p):
        if variance_level:
            W[c, :, :] = variance_level * np.random.randn(d, q)
        else:
            W[c, :, :] = np.random.randn(d, q)

        pi[c] = (clusters == c).sum() / N
        if variance_level:
            sigma2[c] = np.abs((variance_level/10) * np.random.randn())
        else:
            sigma2[c] = (distmin[clusters == c]).mean() / d

    return pi, mu, W, sigma2, clusters


def generateData(family, N=300):
    assert family in ['sine', 'elipse']

    if family == 'sine':
        x = np.linspace(-1, 1, N) + 0.3 * np.random.random(N)
        y = np.sin(5 * x) + 0.3 * np.random.randn(N)
    elif family == 'elipse':
        angle = np.linspace(0, 2 * np.pi, N)
        x = 1.5 * np.cos(angle) + 0.05 * np.random.randn(N)
        y = np.sin(angle) + 0.05 * np.random.randn(N)
    x = x.reshape([-1, 1])
    y = y.reshape([-1, 1])
    data = np.concatenate((x, y), axis=1)
    plotData(data)
    plt.show()
    return data


def plot_fig(X, clusters, W, mu, sigma):
    for c in np.unique(clusters):
        plt.scatter(X[clusters == c, 0], X[clusters == c, 1])
        C = sigma[c]*np.eye(2) + np.dot(W[c, :, :], W[c, :, :].T)
        sample = np.random.multivariate_normal(mu[c, :].reshape((-1)), C, 100)
        plt.scatter(sample[:, 0], sample[:, 1], c='b', s=3)

    plt.scatter(mu[:, 0], mu[:, 1], c='k', s=100)
    plt.show()


def mixture_ppca(X, n_clusters, n_latent=1):
    N, D = X.shape  # N: number of samples, D: number of dims in X

    # initializing
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    # mu = kmeans.cluster_centers_
    # pi = np.ones([n_clusters]) * 1/n_clusters
    # W = np.random.rand(n_clusters, D, n_latent)
    # sigma = np.ones(n_clusters)
    pi, mu, W, sigma, clusters = initialization_kmeans(X, n_clusters, n_latent)
    plot_fig(X, clusters, W, mu, sigma)


    # variables
    _, _, d = W.shape  # d: dims of latent variables (d << D), K: number of clusters
    M = np.zeros((n_clusters, d, d))  # M_k shape : d x d
    C = np.zeros((n_clusters, D, D))  # C_k shape : D x D
    Cinvs = np.zeros((n_clusters, D, D))  # C_k shape : D x D
    Minvs = np.zeros((n_clusters, d, d))
    R = np.zeros((N, n_clusters))
    ES = np.zeros((n_clusters, d, 1, N))
    ESS = np.zeros((n_clusters, N, d, d))

    # EM algorithm
    MAX_iter = 500
    for i in range(MAX_iter):
        # print("{}th iteration".format(i))

        # E step ==================
        for k in range(n_clusters):
            C[k, :, :] = sigma[k] * np.identity(D) + np.matmul(W[k, :, :], W[k, :, :].T)

        for k in range(n_clusters):  # update r_ni
            Y = multivariate_normal.pdf(X, mean=mu[k, :], cov=C[k, :, :])
            R[:, k] = pi[k]*Y

        R_sum = np.sum(R, axis=1).reshape((-1, 1))
        R[:, :] = R[:, :]/R_sum

        for k in range(n_clusters):
            # M_k : W_{k}^{T}W_{k} + sigma_{k}^{2}I
            M[k, :, :] = np.dot(W[k, :, :].T, W[k, :, :]) + sigma[k]*np.eye(d)
            Minvs[k, :, :] = np.linalg.inv(M[k, :, :])

            # C inverse
            Cinvs[k, :, :] = (np.eye(D) - np.dot(np.dot(W[k, :, :], Minvs[k, :, :]), W[k, :, :].T)) / sigma[k]

            # E(s)
            MW = np.dot(Minvs[k, :, :], W[k, :, :].T)
            sigM = sigma[k]*Minvs[k, :, :]
            cnt = X-mu[k, :]
            for n in range(N):
                es = np.dot(MW, cnt[n, :].reshape((D, -1)))
                ES[k, :, :, n] = es
                ESS[k, n, :, :] = sigM + np.dot(es, es.T)

        # M step ==================
        for k in range(n_clusters):
            # pi_new
            pi[k] = np.sum(R[:, k], 0)/N

            # mu_new
            WE = np.zeros((D, 1, N))
            for n in range(N):
                WE[:, :, n] = np.dot(W[k, :, :], ES[k, :, :, n])
            mu[k, :] = np.sum(np.multiply(R[:, k].reshape(-1, 1), (X - np.transpose(np.squeeze(WE)))), 0)\
                       /np.sum(R[:, k].reshape(-1, 1), 0)

            # W_new
            cnt = X - mu[k, :]
            W_front = np.zeros((N, D, d))
            W_back = np.zeros((N, d, d))
            for n in range(N):
                W_front[n, :, :] = R[n, k] * np.dot(cnt[n, :].reshape(-1, 1), ES[k, :, :, n].T)
                W_back[n, :, :] = np.multiply(R[n, k], ESS[k, n, :, :])
            W[k, :, :] = np.dot(np.sum(W_front, 0), np.linalg.inv(np.sum(W_back, 0)))

            # Sigma_new
            wtw = np.dot(W[k, :, :].T, W[k, :, :])
            sig_front = np.zeros((N, 1))
            sig_mid = np.zeros((N, 1))
            sig_last = np.zeros((N, 1))
            for n in range(N):
                sig_front[n, :] = R[n, k] * np.dot(cnt[n, :], cnt[n, :])
                sig_mid[n, :] = R[n, k] * 2*np.dot(np.dot(ES[k, :, :, n].T, W[k, :, :].T), cnt[n, :])
                sig_last[n, :] = R[n, k] * np.trace(np.dot(ESS[k, n, :, :], wtw))

            sigma[k] = (np.sum(sig_front, 0) - np.sum(sig_mid, 0) + np.sum(sig_last, 0)) / (D * np.sum(R[:, k], 0))

    return pi, mu, sigma, W, R


def plotContour(A, mu, sigma2):
    K = len(mu)
    delta = 0.1
    x1_min = x2_min = -2
    x1_max = x2_max = 2
    x1 = np.arange(x1_min, x1_max, delta)
    x2 = np.arange(x2_min, x2_max, delta)
    X1, X2 = np.meshgrid(x1, x2)
    for k in range(K):
        C = np.matmul(A[k], A[k].T) + sigma2[k]*np.identity(2)
        Z = np.zeros([len(x1), len(x2)])
        for i in range(len(x1)):
            for j in range(len(x2)):
                x = np.array([x1[i], x2[j]])
                Z[i, j] = multivariate_normal.pdf(x, mu[k, :], C)
        contour = plt.axes().contour(X1, X2, Z.T)


def plotData(data):
    plt.scatter(data[:,0],data[:,1])
    return


def main():
    # sample = 300
    # x = np.arange(sample)
    # noise = 0.08 * np.asarray(random.sample(range(100, 500), sample))
    # y = 100*np.sin(2 * np.pi * x/100) + noise
    # sin_data = np.transpose(np.array([x.T, y.T]))
    # plt.scatter(x, y)
    # plt.show()

    # sin_data = np.transpose([x, 5*x, 10*x, 2*x])
    sin_data = generateData('sine')
    pi, mu, sigma, W, R = mixture_ppca(sin_data, 6, 1)
    print(pi)
    pi, mu, sigma, W, R = mixture_ppca(sin_data, 4, 1)
    print(pi)
    plotData(sin_data)
    plotContour(W, mu, sigma)
    plt.show()
    plot_fig(sin_data, R.argmax(axis=1), W, mu, sigma)


if __name__ == '__main__':
    main()

