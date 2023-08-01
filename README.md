# Double_Deep_Q-Learning_for_FrozenLake_Env
This repository showcases the implementation of a Double Deep Q-Learnig algorithm for the FrozenLake environment from Open AI's gym library.

The main idea behind Q-learning is that if we had a function $Q^*: State \times Action \rightarrow \mathbb{R}$, that could tell us what our return would be, if we were to take an action in a given state, then we could easily construct a policy that maximizes our rewards:

${\pi}^*(s) = \arg\!\max_a \ Q^*(s, a)$

But this is not scalable. Must compute $Q(s,a)$ for every state-action pair. If state is e.g. current game state pixels, computationally infeasible to compute for entire state space! But, since neural networks are universal function approximators, we can simply create one and train it to resemble $Q^*$.

For our training update rule, we'll use a fact that every $Q$ function for some policy obeys the Bellman equation:

$Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))$

The difference between the two sides of the equality is known as the temporal difference error, $\delta$:

$\delta = Q(s, a) - (r + \gamma \max_a Q(s', a))$

To minimise this error, we will use the `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large - this makes it more robust to outliers when the estimates of $Q$ are very noisy. We calculate this over a batch of transitions, $B$, sampled from the replaymemory:
 
\begin{align}\mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)\end{align}

\begin{align}\text{where} \quad \mathcal{L}(\delta) = \begin{cases}
     \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
     |\delta| - \frac{1}{2} & \text{otherwise.}
   \end{cases}\end{align}


