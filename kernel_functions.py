"""Kernel Functions for Gaussian Processes"""

# Author: Charles Franzen
# License: MIT

import numpy as np
import numpy.linalg as LA


def cov_mat(kernel, x, x_prime, **kernel_args):
    '''Creates covariance matrices.

    Args:
        kernel (callable or Kernel instance):
            The kernel function to be used for covariance calculations.

        x (ndarray): Axis 0 domain.

        x_prime (ndarray): Axis 1 domain.

        kernel_args: Params to be passed to kernel.

    Returns:
        C (ndarray): n by m covariance matrix.
    '''
    n = len(x)
    m = len(x_prime)
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i, j] = kernel(x[i], x_prime[j], **kernel_args)
    return C


# basic kernel definitions
def rational_quadratic_kernel(x, y, sigma=1, ell=1, alpha=1):
    return sigma**2 * (1 + ((x - y)**2) / (2 * alpha * ell**2))**-alpha


def sq_exp_kernel(x, y, sigma=1, ell=1):
    return sigma**2 * np.exp(-(x - y)**2 / (2 * ell**2))


def linear_kernel(x, y, sigma_intercept=0, sigma_slope=1, c=0):
    return sigma_intercept**2 + sigma_slope**2 * (x - c) * (y - c)


def brownian_kernel(x, y):
    return np.minimum(x, y)


def abs_exp_kernel(x, y):
    return np.exp(-1 * np.abs(x - y))


def periodic_kernel(x, y, sigma=1, ell=1, p=1):
    return sigma**2 * \
        np.exp(-2 * np.sin(np.pi * np.abs(x - y) / p)**2 / ell**2)


def locally_periodic_kernel(x, y, sigma=1, ell=1, p=1):
    return sigma**2 * \
        np.exp(-2 * np.sin(np.pi * np.abs(x - y) / p)**2 / ell**2) * \
        np.exp(-(x - y)**2 / (2 * ell**2))


def symmetric_kernel(x, y):
    return np.exp(-min(np.abs(x - y), np.abs(x + y))**2)


def constant_kernel(x, y):
    return .5


def polynomial_kernel(x, y, d=3):
    return (np.dot(x, y))**d


def matern_32(x, y, ell=1):
    r = np.sqrt(np.sum((x / ell - y / ell)**2))
    term1 = 1 + np.sqrt(3) * r
    term2 = np.exp(-np.sqrt(3) * r)
    return term1 * term2


def matern_52(x, y, ell=1):
    r = np.sqrt(np.sum((x - y)**2))
    term1 = 1 + np.sqrt(5) * r / ell + 5 * r**2 / (3 * ell**2)
    term2 = np.exp(-np.sqrt(5) * r / ell)
    return term1 * term2


def ornstein_uhlenbeck(x, y, ell=1):
    r = np.sqrt(np.sum((x - y)**2))
    return np.exp(-r / ell)


def greens_kernel(x, y):
    return np.minimum(x, y) - x * y


_kernel_dict = {
    'rational_quadratic': rational_quadratic_kernel,
    'sq_exp': sq_exp_kernel,
    'linear': linear_kernel,
    'brownian': brownian_kernel,
    'abs_exp': abs_exp_kernel,
    'periodic': periodic_kernel,
    'locally_periodic': locally_periodic_kernel,
    'symmetric': symmetric_kernel,
    'constant': constant_kernel,
    'polynomial': polynomial_kernel,
    'matern_32': matern_32,
    'matern_52': matern_52,
    'ornstein_uhlenbeck': ornstein_uhlenbeck,
    'greens': greens_kernel
}


class Kernel(object):
    '''Base class for kernel functions.

    Args:
        theta: kwargs specifying parameter values for the Kernel
    '''
    def __init__(self, **theta):
        self.params = dict(theta)

    def covariance(self, x, x_prime):
        pass

    def log_marginal_likelihood(self, x, y, noise=.01):
        '''Log marginal likelihood of a kernel function and observations.

        Args:
            x (ndarray): Data points x.

            y (ndarray): Noisy targets y.

            noise (float): Gaussian noise in observations.

        Returns:
            lml (float): Log marginal likelihood.
        '''
        K = cov_mat(self.covariance, x, x)
        n = len(y)
        Ky = K + np.eye(n) * noise**2
        lml = (-1 / 2) * y.T.dot(LA.inv(Ky)).dot(y) - (1 / 2) * \
            np.log(LA.det(Ky)) - (n / 2) * np.log(2 * np.pi)
        return lml

    def grad_log_marginal_likelihood(self):
        pass

    def optimize_params(self, x, y):
        pass

    def set_params(self, **theta):
        self.params = dict(theta)


class PeriodicKernel(Kernel):
    '''Periodic Kernel function for GP covariance matrices.

    Args:
        sigma: variance scale

        ell: lengthscale

        p: period length
    '''
    def __init__(self, sigma=1., ell=1., p=1.):
        self.params = {'sigma': sigma, 'ell': ell, 'p': p}

    def covariance(self, x, x_prime):
        '''Covariance between two points'''
        sigma = self.params['sigma']
        ell = self.params['ell']
        p = self.params['p']

        r = LA.norm(x - x_prime)
        return sigma**2 * np.exp(-2 * (np.sin(np.pi * r / p) / ell)**2)

    def grad_log_marginal_likelihood(self, x, y):
        '''Gradient of the marginal likelihood

        Args:
            x (ndarray): Data points x.

            y (ndarray): Noisy targets y.

        Returns:
            grad (ndarray): Gradient of the log marginal likelihood.
        '''
        sigma = self.params['sigma']
        ell = self.params['ell']
        p = self.params['p']

        d_K_d_sigma = cov_mat(lambda u, v: 2 * sigma *
                              np.exp((-2 * np.sin(np.pi * np.abs(u - v) /
                                                  p)**2) / ell**2),
                              x, x)
        d_K_d_p = cov_mat(lambda u, v: (sigma**2) *
                          np.exp((-2 * np.sin(np.pi * np.abs(u - v) /
                                              p)**2) / ell**2) *
                          (2 * np.pi * np.abs(u - v) *
                           np.cos(np.pi * np.abs(u - v) / p)) /
                          (ell**2 * p**2), x, x)
        d_K_d_l = cov_mat(lambda u, v: (sigma**2) *
                          np.exp((-2 * np.sin(np.pi * np.abs(u - v) /
                                              p)**2) / ell**2) *
                          ((4 * np.sin(np.pi * np.abs(u - v) / p)**2) /
                          ell**3), x, x)

        K = cov_mat(self.covariance, x, x)
        K = K + .1 * np.eye(len(y))
        K_inv = LA.inv(K)
        alpha = K_inv.dot(y)
        term1 = alpha.dot(alpha.T) - K_inv

        d_d_sigma = (1 / 2) * np.trace(term1.dot(d_K_d_sigma))
        d_d_p = (1 / 2) * np.trace(term1.dot(d_K_d_p))
        d_d_l = (1 / 2) * np.trace(term1.dot(d_K_d_l))
        grad = np.array([d_d_sigma, d_d_p, d_d_l])
        return grad

    def optimize_params(self, x, y,
                        n_steps=100,
                        learning_rate=.001,
                        momentum=.8,
                        n_restarts=0,
                        return_vals=False):
        '''Gradient ascent to optimize kernel parameters.

        Args:
            x (ndarray): Data points x.

            y (ndarray): Noisy targets y.

            n_steps (int): Number of iterations to run gradient ascent.

            learning_rate (float): Step size.

            n_restarts (int): Number of random restarts for gradient ascent.

            return_vals (bool):
                If True, returns the best params as a dict and trace of log
                likelihoods.

        Returns:
            best_trace (list):
                Log marginal likelihood values at each step of the best random
                restart.

            best_params (ndarray): Best parameters found by the optimizer.
        '''
        best_log_likelihood = -np.inf
        best_params = None
        best_trace = []
        for j in range(n_restarts + 1):
            # random start for params
            params = np.random.rand(3)
            trace = []
            update = 0
            for i in range(n_steps):
                # gradient ascent loop
                if -1 in np.sign(params):
                    params = np.random.rand(3)
                    continue
                grad = self.grad_log_marginal_likelihood(x, y)
                update = learning_rate * grad + momentum * update
                params += update
                self.set_params(sigma=params[0],
                                ell=params[1],
                                p=params[2])
                trace.append(self.log_marginal_likelihood(x, y))
            if trace[-1] > best_log_likelihood:
                best_params = params
                best_trace = trace
        best_params = {'sigma': best_params[0],
                       'p': best_params[1],
                       'ell': best_params[2]
                       }
        self.set_params(**best_params)
        if return_vals:
            return best_trace, best_params


class SqExpKernel(Kernel):
    '''Squared Exponential Kernel function for GP covariance matrices.

    Args:
        sigma: variance scale

        ell: lengthscale
    '''
    def __init__(self, sigma=1., ell=1.):
        self.params = {'sigma': sigma, 'ell': ell}

    def covariance(self, x, y):
        '''Covariance between two points'''
        sigma = self.params['sigma']
        ell = self.params['ell']

        r = LA.norm(x - y)
        return sigma**2 * np.exp(-.5 * r**2 / ell**2)

    def grad_log_marginal_likelihood(self, x, y):
        '''Gradient of the marginal likelihood

        Args:
            x (ndarray): Data points x.

            y (ndarray): Noisy targets y.

        Returns:
            grad (ndarray): Gradient of the log marginal likelihood.
        '''
        sigma = self.params['sigma']
        ell = self.params['ell']

        d_K_d_sigma = cov_mat(lambda u, v: 2 * sigma *
                              np.exp((-LA.norm(u - v)**2) /
                                     (2 * ell**2)), x, x)
        d_K_d_l = cov_mat(lambda u, v: (sigma**2) *
                          np.exp((-LA.norm(u - v)**2) / (2 * ell**2)) *
                          ((LA.norm(u - v)**2) / ell**3), x, x)
        K = cov_mat(self.covariance, x, x)
        K = K + .01 * np.eye(len(y))
        K_inv = LA.inv(K)
        alpha = K_inv.dot(y)
        term1 = alpha.dot(alpha.T) - K_inv

        d_d_sigma = (1 / 2) * np.trace(term1.dot(d_K_d_sigma))
        d_d_l = (1 / 2) * np.trace(term1.dot(d_K_d_l))

        return np.array([d_d_sigma, d_d_l])

    def optimize_params(self, x, y,
                        n_steps=100,
                        learning_rate=.001,
                        momentum=.8,
                        n_restarts=0,
                        return_vals=False):
        '''Gradient ascent to optimize kernel parameters.

        Args:
            x (ndarray): Data points x.

            y (ndarray): Noisy targets y.

            n_steps (int): Number of iterations to run gradient ascent.

            learning_rate (float): Step size.

            n_restarts (int): Number of random restarts for gradient ascent.

            return_vals (bool):
                If True, returns the best params as a dict and trace of log
                likelihoods.

        Returns:
            best_trace (list):
                Log marginal likelihood values at each step of the best random
                restart.

            best_params (ndarray): Best parameters found by the optimizer.
        '''
        best_log_likelihood = -np.inf
        best_params = None
        best_trace = []
        for j in range(n_restarts + 1):
            # random parameter initialization
            params = np.random.rand(2)
            trace = []
            # gradient update
            update = 0
            for i in range(n_steps):
                # gradient ascent
                if -1 in np.sign(params):
                    # reset if params go negative
                    params = np.random.rand(2)
                    continue
                grad = self.grad_log_marginal_likelihood(x, y)
                update = learning_rate * grad + momentum * update
                params += update
                self.set_params(sigma=params[0], ell=params[1])
                trace.append(self.log_marginal_likelihood(x, y, noise=.1))
            if trace[-1] > best_log_likelihood:
                best_params = params
                best_trace = trace
        best_params = {'sigma': best_params[0], 'ell': best_params[1]}
        self.set_params(**best_params)
        if return_vals:
            return best_trace, best_params
