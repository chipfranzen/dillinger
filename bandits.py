"""Multi-armed bandits"""

# Author: C. Franzen
# License: MIT

import numpy as np


class SoftMax(object):
    '''The softmax bandit algorithm.'''
    def __init__(self, counts=None, values=None, tau_scale=1e3):
        self._t = 1
        self.tau_scale = tau_scale
        if counts is not None:
            self._counts = np.array(counts, dtype=int)
            self._values = np.array(values, dtype=float)
            self.n_arms = len(self._counts)
        else:
            self._counts = counts
            self._values = values
            self.n_arms = 0

    def get_probs(self):
        '''Returns probabilites according to the softmax function.'''
        # anneal the tau parameter
        tau = self.tau_scale / self._t
        z = np.sum([np.exp(v / tau) for v in self._values])
        probs = np.array([np.exp(v / tau) / z for v in self._values])
        return probs

    def initialize(self, n_arms: int):
        '''Initializes a blank bandit.'''
        self._counts = np.zeros(n_arms, dtype=int)
        self._values = np.ones(n_arms, dtype=float)
        self.n_arms = n_arms

    def state_report(self, action_labels=None):
        '''Prettily prints bandit state.'''
        print('Softmax Results\n' + '-' * 20)
        print('{} total observations over {} actions'.format(sum(self._counts),
                                                             self.n_arms))
        if action_labels is None:
            action_labels = ['action' + str(i) for i in range(self.n_arms)]
        price_pt_str = '\nprice point: {:>19}' + ' {:>8}' * (self.n_arms - 1)
        print(price_pt_str.format(*action_labels))
        prob_str = 'probabilities: {:>17.4f}' + \
                   ' {:>8.4f}' * (self.n_arms - 1)
        print(prob_str.format(*self.get_probs()))
        ev_str = 'expected value estimate: {:>7.4f}' + \
                 ' {:>8.4f}' * (self.n_arms - 1)
        print(ev_str.format(*self._values))
        obs_str = 'observations: {:>18}' + ' {:>8}' * (self.n_arms - 1)
        print(obs_str.format(*self._counts))
        best_arm = action_labels[np.argmax(self._values)]
        print('BEST ACTION: {}'.format(best_arm))

    def select_arm(self):
        '''Selects an arm based upon softmax probabilties.'''
        probs = self.get_probs()
        self._t += 1
        return np.random.choice(np.arange(self.n_arms), p=probs)

    def update(self, chosen_arm: int, reward: float):
        '''Given a pulled arm and observed reward,
           updates softmax counts and values.'''
        self._counts[chosen_arm] = self._counts[chosen_arm] + 1
        n = self._counts[chosen_arm]
        value = self._values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self._values[chosen_arm] = new_value
