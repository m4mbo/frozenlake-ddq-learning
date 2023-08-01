# Double_Deep_Q-Learning_for_FrozenLake_Env
This repository showcases the implementation of a Double Deep Q-Learnig algorithm for the FrozenLake environment from Open AI's gym library.

The main idea behind Q-learning is that if we had a function $Q^*: State \times Action \rightarrow \mathbb{R} $, that could tell us what our return would be, if we were to take an action in a given state, then we could easily construct a policy that maximizes our rewards: 

$ \pi ^ * (s) = \arg\!\max_a \ Q^*(s, a) $

But is not scalable. Must compute $Q(s,a)$ for every state-action pair. If state is e.g. current game state pixels, computationally infeasible to compute for entire state space! But, since neural networks are universal function approximators, we can simply create one and train it to resemble $Q^*$.

For our training update rule, we'll use a fact that every $Q$ function for some policy obeys the Bellman equation:

$Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))$

The difference between the two sides of the equality is known as the temporal difference error, $\delta$:

$\delta = Q(s, a) - (r + \gamma \max_a Q(s', a))$

To minimise this error, we will use the `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large - this makes it more robust to outliers when the estimates of $Q$ are very noisy. We calculate this over a batch of transitions, $B$, sampled from the replaymemory.

# Double Deep Q-Learning

We will implement Double Deep Q-Learning here. Double Deep Q-Learning is used to reduce the maximaztion bias in Q-Learning. This entails using two separate $Q$-value estimators, each of which is used to update the other. The target values are calculated using a target Q-network. The target Q-network's parameters are updated to the current networks every $C$ time steps.

![image](https://github.com/M4mbo/Double_Deep_Q-Learning_for_FrozenLake_Env/assets/115642529/a044aa38-dc09-45c4-96f1-7688e795b1a2)

Frozen Lake:

![descarga (1)](https://github.com/M4mbo/Double_Deep_Q-Learning_for_FrozenLake_Env/assets/115642529/1cca1f96-a6fd-4a88-a2d5-aded5ae8ba86)


