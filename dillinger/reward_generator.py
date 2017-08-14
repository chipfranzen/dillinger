"""Generate samples of synthetic data set"""

# Author: C. Franzen
# License: MIT

import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy import stats


class RewardGenerator(object):
    '''Generates rewards for multi-armed bandit algorithms.

    Args:
        min_price (float): The lowest price to be offered.

        max_price (float): The highest price to be offered.

        n_price_points (int): The number of price points to be offered.

        seed (int): Seed to set the random state.
    '''
    def __init__(self,
                 min_price: float,
                 max_price: float,
                 n_price_points: int,
                 seed=None):
        self.max_price = max_price
        self.min_price = min_price
        self.n_price_points = n_price_points
        # create the domain
        self.x = np.linspace(self.min_price,
                             self.max_price,
                             self.n_price_points)
        if seed:
            np.random.seed(seed)

    def __getitem__(self, key: int) -> float:
        return self.pull_arm(key)

    def create_spend_distributions(self, variance_level=1000):
        '''Creates high variance gamma distributions for when customers decide
           to spend.'''
        self.dists = []
        for exp, prob in zip(self.expected_value, self.p):
            theta = stats.gamma(variance_level).rvs()
            k = exp / (theta * prob)
            dist = stats.gamma(a=k, scale=theta)
            self.dists.append(dist)

    def plot_conversion_rates(self):
        plt.plot(self.x, self.p, label='chance to spend')
        plt.xlabel('price')
        plt.legend()

    def plot_expected_values(self):
        plt.plot(self.x,
                 self.expected_value,
                 label='expected value per customer')
        plt.xlabel('price')
        plt.legend()

    def pull_arm(self, arm_index: int) -> float:
        '''Returns a pull from a multi-armed bandit.

        Args:
            arm_index: index of the arm to be pulled

        Returns:
            observed reward
        '''
        if stats.bernoulli(self.p[arm_index]).rvs():
            return self.dists[arm_index].rvs()
        else:
            return 0

    def sample(self, n_draws: int):
        '''Draws random samples from the reward generator by pulling arms
           uniformly.'''
        draws = np.zeros((n_draws, 2))
        for i in range(n_draws):
            price = np.random.randint(self.n_price_points)
            draw = self.pull_arm(price)
            draws[i] = price, draw
        return draws

    def set_conversion_rates(self, x0=5, k=.7, max_conversion_rate=.1):
        '''Sets spend probabilities and calculates expected values.'''
        self.x0 = x0  # x-value of the midpoint
        self.k = k  # steepness of the curve
        self.max_conversion_rate = max_conversion_rate
        self.p = max_conversion_rate * \
            special.expit(-(self.k * (self.x - self.x0))) + .01
        # create the objective function
        self.expected_value = self.x * self.p
