"""Multi-armed bandits"""

# Author: C. Franzen
# License: MIT

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class SoftMax(object):
    '''The softmax bandit algorithm.

    Args:
        n_arms (int): Number of arms in the bandit

        counts (ndarray):
            Number of times each arm has been pulled. Useful for starting a new
            bandit with observations from a previous bandit.

        values (ndarray):
            Average reward of each arm. Also useful for starting a new bandit
            with previous observations.

        tau_scale (int):
            Initial value for the temperature parameter. Larger values mean
            that the bandit will act more randomly for longer.
    '''
    def __init__(self, n_arms: int, counts=None, values=None, tau_scale=1e3):
        self._t = 1
        self.tau_scale = tau_scale
        self.action_trace = None
        self.regret_trace = None
        if counts is not None:
            self._counts = np.array(counts, dtype=int)
            self._values = np.array(values, dtype=float)
            self.n_arms = len(self._counts)
        else:
            self._counts = np.zeros(n_arms, dtype=int)
            self._values = np.ones(n_arms, dtype=float)
            self.n_arms = n_arms

    def get_probs(self):
        '''Returns probabilites according to the softmax function.'''
        # anneal the tau parameter
        tau = self.tau_scale / self._t
        z = np.sum([np.exp(v / tau) for v in self._values])
        probs = np.array([np.exp(v / tau) / z for v in self._values])
        return probs

    def plot_action_trace(self, best_action=None, action_labels=None):
        '''Plots action allocation over time'''
        if action_labels is None:
            action_labels = ['action' + str(i) for i in range(self.n_arms)]
        if self.action_trace is None:
            raise RuntimeError('The bandit has not been run.')
        trace_count = self.action_trace.shape[0]
        sns.set_palette('cubehelix', len(self._values))
        cum_trace = self.action_trace.cumsum(axis=1)
        plt.fill_between(np.arange(trace_count), 0, cum_trace[:, 0],
                         label=action_labels[0])
        for i in range(len(self._values) - 1):
            if i == best_action:
                plt.fill_between(np.arange(trace_count),
                                 cum_trace[:, i],
                                 cum_trace[:, i + 1],
                                 color='r',
                                 label='best action')
            else:
                plt.fill_between(np.arange(trace_count),
                                 cum_trace[:, i],
                                 cum_trace[:, i + 1],
                                 label=action_labels[i + 1])
        plt.legend(bbox_to_anchor=(1.2, .5))
        plt.xlabel('$t$')
        plt.tick_params(axis='x', labelbottom='off')
        plt.ylabel('$P(action = a)$')

    def run(self,
            actions: list,
            reward_generator,
            n_steps: int,
            track_regret=False,
            best_action=None,
            trace_count=20):
        '''Bandits over the given actions on the given reward generator.

        Args:
            actions (list):
                A list of actions to be considered. Entries should be indices
                for actions in the reward generator.

            reward_generator: A RewardGenerator instance.

            n_steps (int): Number of time steps to run the bandit.

            regret (bool):
                If True, tracks the regret of the bandit. If True, a best
                action must be provided.

            best_action (int): Index of the best action for regret tracking.

            trace_count (int): Number of entries in the trace.
        '''
        if track_regret and (best_action is None):
            raise RuntimeError('best_action must be provided to track regret')
        trace_interval = n_steps // trace_count
        trace = np.zeros((trace_count, len(actions)))
        regret_trace = np.zeros(1)

        # run bandit
        print('running bandit...')
        for step in range(n_steps):
            chosen_arm = self.select_arm()
            reward = reward_generator[actions[chosen_arm]]
            if track_regret:
                regret = reward_generator[best_action] - reward
                regret_trace = np.append(regret_trace, regret)
            self.update(chosen_arm, reward)
            if step % trace_interval == 0:
                print('step {} complete'.format(step))
                trace[int(step / trace_interval), :] = self.get_probs()
        self.action_trace = trace
        self.regret_trace = np.cumsum(regret_trace)
        print('bandit run complete')

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
