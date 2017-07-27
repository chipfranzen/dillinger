#Dillinger: Deadly accurate multi-armed bandits

Dillinger is a guide to using Bayesian optimization to select new actions for multi-armed bandits. The core of the project is a **Gaussian Process** class that can be fit to observations from multi-armed bandits. To facilitate demonstration, the package also has the following features: a data generator that simulates LTV of customers based on a price sensitivity curve, an implementation of the Softmax bandit algorithm.

This project is still very much under construction, as I'm adapting an existing project to make it more useable and accessible to those interested in applying Bayesian optimization to A/B tests or multi-armed bandit experiments.