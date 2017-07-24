#!\usr\bin\env python3

'''
Classes and methods for multi-armed bandits
'''

# Author: C. Franzen

import numpy as np


class SoftMax:
    '''
    The softmax bandit algorithm
    '''
    def __init__(self, counts=None, values=None, tau_scale=1e3):
        self.t = 1
        self.tau_scale = tau_scale
        if counts is not None:
            self.counts = np.array(counts, dtype=int)
            self.values = np.array(values, dtype=float)
            self.num_arms = len(self.counts)
        else:
            self.counts = counts
            self.values = values
            self.num_arms = 0

    def get_probs(self):
        '''
        returns probabilites according to the softmax function
        '''
        # anneal the tau parameter
        tau = self.tau_scale / self.t
        z = np.sum([np.exp(v / tau) for v in self.values])
        probs = np.array([np.exp(v / tau) / z for v in self.values])
        return probs

    def initialize(self, n_arms: int):
        '''
        initializes a blank bandit
        '''
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.ones(n_arms, dtype=float)
        self.num_arms = n_arms

    def results_report(self, action_labels=None):
        print('Softmax Results\n' + '-' * 20)
        print('{} total observations over {} actions'.format(sum(self.counts),
                                                             self.num_arms))
        if action_labels is None:
            action_labels = ['action' + str(i) for i in range(self.num_arms)]
        price_pt_str = '\nprice point: {:>19}' + ' {:>8}' * (self.num_arms - 1)
        print(price_pt_str.format(*action_labels))
        prob_str = 'probabilities: {:>17.4f}' + \
                   ' {:>8.4f}' * (self.num_arms - 1)
        print(prob_str.format(*self.get_probs()))
        ev_str = 'expected value estimate: {:>7.4f}' + \
                 ' {:>8.4f}' * (self.num_arms - 1)
        print(ev_str.format(*self.values))
        obs_str = 'observations: {:>18}' + ' {:>8}' * (self.num_arms - 1)
        print(obs_str.format(*self.counts))
        best_arm = action_labels[np.argmax(self.values)]
        print('BEST ACTION: {}'.format(best_arm))


    def select_arm(self):
        '''
        selects an arm based upon softmax probabilties
        '''
        probs = self.get_probs()
        self.t += 1
        return np.random.choice(np.arange(self.num_arms), p=probs)

    def update(self, chosen_arm: int, reward: float):
        '''
        given a reward for a pulled arm, updates softmax counts and values
        '''
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
