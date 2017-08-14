"""1-D Gaussian Processes for Regression and Bayesian Optimization"""

# Author: Charles Franzen
# License: MIT
from functools import partial

import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import dillinger.kernel_functions as kern


# main Gaussian Process class, with sample and fit methods.
class GaussianProcess:
    '''Gaussian Processes for Regression.

    Args:
        domain (ndarray): Points in the domain.

        kernel_function (string or Kernel):
            Kernel to be used for covariance matrices. Can be a string
            specifying a kernel function or a Kernel instance.

        kernel_args (dict):
            Dictionary of keyword arguments to pass the the kernel function.

        noise (float): Gaussan noise assumed in observations.
    '''
    def __init__(self, domain: np.ndarray, kernel_function,
                 kernel_args=None, noise=1.):
        self.domain = domain
        self.domain.shape = -1, 1
        self.n = np.prod(domain.shape)
        self.noise = noise

        if type(kernel_function) is str:
            kernel_function = kern._kernel_dict[kernel_function]
            self.kernel_obj = None
            if kernel_args:
                self.kernel_args = kernel_args
                self.kernel = partial(kernel_function, **kernel_args)
            else:
                self.kernel = kernel_function
        else:  # for Kernel objects
            if kernel_args:
                self.kernel_args = kernel_args
                self.kernel_obj = kernel_function(**kernel_args)
            else:
                self.kernel_obj = kernel_function()
            self.kernel = self.kernel_obj.covariance

        self.K = kern.cov_mat(self.kernel, self.domain, self.domain)
        self.μ = np.zeros(self.n)
        self.μ.shape = -1, 1
        self.obs = None

    def expected_improvement(self):
        '''Expected improvement over the domain.

        Returns:
            ei (ndarray):
                Expected improvement values at each point in the domain.
        '''
        obs_df = pd.DataFrame(self.obs, columns=['x', 'y'])
        # mean observation at each point in the domain
        mean_obs = obs_df.groupby('x').mean()
        best_val = mean_obs.y.max()
        # EI calculation
        sigma = np.sqrt(np.diag(self.K))
        gamma = (self.μ.flatten() - best_val) / sigma
        ei = sigma * (gamma * stats.norm.cdf(gamma) + stats.norm.pdf(gamma))
        return ei

    def fit(self, x, y, clear_obs=False, **optimizer_args):
        '''Fit the GP to observed data

        Args:
            x (ndarray): domain values

            y (ndarray): noisy observations

            clear_obs (bool): if True, clears previously stored observations

            optimizer_args: args to be passed to Kernel.optimize_params()
        '''
        if clear_obs:
            self.obs = None
        if self.obs is not None:
            self.obs = np.concatenate((self.obs, np.concatenate((x, y),
                                      axis=1)))
        else:
            self.obs = np.concatenate((x, y), axis=1)
        x = self.obs[:, 0]
        x.shape = -1, 1
        y = self.obs[:, 1]
        y.shape = -1, 1
        n_obs = x.shape[0]

        if self.kernel_obj:
            self.kernel_obj.optimize_params(x, y, **optimizer_args)
            self.set_kernel_args(self.kernel_obj.params)

        # create block matrix
        K_X_X = kern.cov_mat(self.kernel, x, x)
        K_X_Xt = kern.cov_mat(self.kernel, x, self.domain)
        K_Xt_X = kern.cov_mat(self.kernel, self.domain, x)
        K_Xt_Xt = kern.cov_mat(self.kernel, self.domain, self.domain)

        # get means
        shared_term = K_Xt_X.dot(LA.inv(K_X_X + self.noise**2 * np.eye(n_obs)))

        self.μ = shared_term.dot(y)  # see equation (4)

        # get covariances
        self.K = K_Xt_Xt - shared_term.dot(K_X_Xt)  # see equation (3)

    def plot(self, n_samples=0):
        # get sigmas
        sigmas = np.sqrt(np.diag(self.K))
        sigmas.shape = -1, 1
        # get confidence intervals
        upper_ci = self.μ + 1.96 * sigmas
        lower_ci = self.μ - 1.96 * sigmas
        x = self.domain.flatten()
        upper_ci = upper_ci.flatten()
        lower_ci = lower_ci.flatten()
        samples = self.sample(n_samples)
        for sample in samples:
            plt.plot(x, sample, alpha=.5)
        plt.plot(x, self.μ, color='k', label='GP mean estimate')
        plt.fill_between(x, lower_ci, upper_ci, alpha=.5, color='m')
        if self.obs is not None:
            plt.scatter(self.obs[:, 0],
                        self.obs[:, 1],
                        color='r',
                        marker='.',
                        linewidths=2,
                        s=100,
                        label='Observations')
        plt.legend()

    def plot_expected_improvement(self, objective=None):
        # get sigmas
        sigmas = np.sqrt(np.diag(self.K))
        sigmas.shape = -1, 1
        # get confidence intervals
        upper_ci = self.μ + 1.96 * sigmas
        lower_ci = self.μ - 1.96 * sigmas
        x = self.domain.flatten()
        upper_ci = upper_ci.flatten()
        lower_ci = lower_ci.flatten()

        # get expected improvement
        ei = self.expected_improvement()
        ei_max = ei.max()
        ei_arg = np.argmax(ei)

        # plotting
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)

        # plot the GP
        ax1.plot(x, self.μ, color='k', label='GP mean estimate')
        ax1.fill_between(x, lower_ci, upper_ci, alpha=.5, color='m')
        ax1.scatter(self.obs[:, 0],
                    self.obs[:, 1],
                    color='r',
                    marker='.',
                    linewidths=2,
                    s=100,
                    label='Observations')
        if objective is not None:
            if type(objective) == np.ndarray:
                ax1.plot(self.domain,
                         objective,
                         linestyle='dashed',
                         linewidth=3,
                         c='y',
                         label='True objective')
            else:
                obj_x = np.linspace(np.min(self.domain),
                                    np.max(self.domain),
                                    100)
                obj_y = objective(obj_x)
                ax1.plot(obj_x,
                         obj_y,
                         linestyle='dashed',
                         linewidth=3,
                         c='y',
                         label='True objective')
        ax1.set_title('GP')
        ax1.legend(bbox_to_anchor=(1.1, 1.05))

        # plot EI
        ax2.plot(x, ei, linewidth=3)
        ax2.vlines(x[ei_arg], 0, ei_max,
                   color='r', linestyles='dashed', label='max EI')
        ax2.set_title('Expected improvement')
        ax2.set_xlabel('$x$')
        ax2.set_ylabel('$a(x)$')
        ax2.legend()

    def sample(self, n_samples):
        # uses the cholesky decomp of the covariance matrix to draw samples
        samples = np.zeros((n_samples, self.n))
        try:
            L = LA.cholesky(self.K + 1e-5 * np.eye(self.n))
        except LA.LinAlgError:
            # attempt rank 1 update if not positive definite
            print('attempting rank 1 update')
            e, v = LA.eig(self.K)
            v1 = v[:, 0]
            v1.shape = -1, 1
            e1 = e[0]
            print(f'negative eigenvalue: {e1:.4f}')
            perturbed_K = self.K + v1.dot(v1.T).dot(np.spacing(e1) - e1)
            L = LA.cholesky(perturbed_K)
        for i in range(n_samples):
            # draw samples
            u = np.random.randn(self.n, 1)
            z = L.dot(u)
            z += self.μ
            z = z.reshape(1, self.n)
            samples[i] = z
        return samples

    def set_kernel_args(self, kernel_args):
        self.kernel_args = kernel_args
        if self.kernel_obj:
            self.kernel_obj.set_params(**kernel_args)
            self.kernel = self.kernel_obj.covariance
        else:
            self.kernel = partial(self.kernel, **kernel_args)

    def x_next(self):
        # get next point to sample
        ei = self.expected_improvement()
        return np.argmax(ei)
