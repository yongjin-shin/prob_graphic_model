import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.mixture import BayesianGaussianMixture
from matplotlib.patches import Ellipse
import scipy.sparse as sp
import pandas as pd


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def create_cov_ellipse(cov, pos, nstd=1, **kwargs):
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    return ellip


def bar_plot(K, w, iteration):
    plt.ylim(0,1)
    plt.bar(np.arange(K), w)
    plt.savefig('bar_{}.png'.format(iteration))
    plt.show()
    plt.close()


def plot(estimator, X, title, iteration, plot_title=False):
    x1 = X[:, 0]
    x2 = X[:, 1]
    plt.scatter(x1, x2)
    ells = []
    w = estimator.weights_
    mu = estimator.means_
    cov = estimator.covariances_
    for k in range(len(w)):
        ells.append(create_cov_ellipse(cov=cov[k], pos=mu[k, :]))

    a = plt.subplot(111)
    for k, e in enumerate(ells):
        a.add_artist(e)
        e.set_facecolor((1 - w[k], 1 - w[k], 1 - w[k]))

    a.set_xlim(np.min(x1) - np.min(x1) / 10, np.max(x1) + np.min(x1) / 10)
    a.set_ylim(np.min(x2) - np.min(x2) / 10, np.max(x2) + np.min(x2) / 10)
    plt.scatter(x1, x2)
    plt.savefig('plot_{}.png'.format(iteration))
    plt.show()
    plt.close()

estimators = [
    ("Infinite mixture with a Dirichlet process\n prior and" r"$\gamma_0=$",
     BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_process",
        n_components=2 * 5, reg_covar=0, init_params='random',
        max_iter=10, mean_precision_prior=.8,
        random_state=3), [1], 10),
    ("Infinite mixture with a Dirichlet process\n prior and" r"$\gamma_0=$",
     BayesianGaussianMixture(
         weight_concentration_prior_type="dirichlet_process",
         n_components=2 * 5, reg_covar=0, init_params='random',
         max_iter=15, mean_precision_prior=.8,
         random_state=3), [1], 15),
    ("Infinite mixture with a Dirichlet process\n prior and" r"$\gamma_0=$",
         BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_process",
            n_components=2 * 5, reg_covar=0, init_params='random',
            max_iter=20, mean_precision_prior=.8,
            random_state=3), [1], 20),
    ("Infinite mixture with a Dirichlet process\n prior and" r"$\gamma_0=$",
         BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_process",
            n_components=2 * 5, reg_covar=0, init_params='random',
            max_iter=50, mean_precision_prior=.8,
            random_state=3), [1], 50),
    ("Infinite mixture with a Dirichlet process\n prior and" r"$\gamma_0=$",
         BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_process",
            n_components=2 * 5, reg_covar=0, init_params='random',
            max_iter=100, mean_precision_prior=.8,
            random_state=3), [1], 100)]

# Generate data
# df = pd.read_csv("faithful.csv")
# X = df.values
def load_ap_data(ap_data, ap_vocab):
    n = len(open(ap_data).readlines())
    m = len(open(ap_vocab).readlines())

    X = sp.lil_matrix((n, m))

    for i, line in enumerate(open(ap_data)):
        words = line.split()
        idxs = []
        vals = []
        for w in words[1:]:
            idx, val = map(int, w.split(':'))
            idxs.append(idx)
            vals.append(val)

        X[i, idxs] = vals

    X = X.tocsr()
    return X

data_file = './ap/ap.dat'
vocab_file = './ap/vocab.txt'
X = load_ap_data(data_file, vocab_file)
X = X.toarray()
X = X[1:100, :]

for (title, estimator, concentrations_prior, iteration) in estimators:
    plt.figure(figsize=(4.7 * 3, 8))
    plt.subplots_adjust(bottom=.04, top=0.90, hspace=.05, wspace=.05,
                        left=.03, right=.99)

    gs = gridspec.GridSpec(3, len(concentrations_prior))
    for k, concentration in enumerate(concentrations_prior):
        estimator.weight_concentration_prior = concentration
        estimator.fit(X)
        plot(estimator, X, r"%s$%.1e$" % (title, concentration), iteration, plot_title=k == 0)
        bar_plot(len(estimator.weights_), estimator.weights_, iteration)
