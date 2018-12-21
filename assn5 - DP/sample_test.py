import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm

# np.set_printoptions(precision=2)
# def stats(scale_factor, G0=[.2, .2, .6], N=10000):
#     samples = dirichlet(alpha=scale_factor * np.array(G0)).rvs(N)
#     print("                          alpha:", scale_factor)
#     print("              element-wise mean:", samples.mean(axis=0))
#     print("element-wise standard deviation:", samples.std(axis=0))
#     print()
# for scale in [0.1, 1, 10, 100, 1000]:
#     stats(scale)


def dirichlet_sample_approximation(base_measure, alpha, tol=0.01):
    betas = []
    pis = []
    betas.append(beta(1, alpha).rvs())
    pis.append(betas[0])
    while sum(pis) < (1.-tol):
        s = np.sum([np.log(1 - b) for b in betas])
        new_beta = beta(1, alpha).rvs()
        betas.append(new_beta)
        pis.append(new_beta * np.exp(s))
    pis = np.array(pis)
    thetas = np.array([base_measure() for _ in pis])
    return pis, thetas


def plot_normal_dp_approximation(alpha):
    plt.figure()
    plt.title("Dirichlet Process Sample with N(0,1) Base Measure")
    plt.suptitle("alpha: %s" % alpha)
    pis, thetas = dirichlet_sample_approximation(lambda: norm().rvs(), alpha)
    pis = pis * (norm.pdf(0) / pis.max())
    plt.vlines(thetas, 0, pis, )
    X = np.linspace(-4,4,100)
    plt.plot(X, norm.pdf(X))
    plt.show()


plot_normal_dp_approximation(.1)
plot_normal_dp_approximation(1)
plot_normal_dp_approximation(10)
plot_normal_dp_approximation(1000)
