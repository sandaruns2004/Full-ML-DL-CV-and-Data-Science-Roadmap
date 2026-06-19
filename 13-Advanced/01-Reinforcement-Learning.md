# 🤖 Deep Reinforcement Learning (DRL)

> **Prerequisites**: Neural Networks Fundamentals, Deep Learning | **Difficulty**: ⭐⭐⭐⭐⭐ Expert

---

## 📋 Table of Contents
1. [The Reinforcement Learning Framework](#1-the-reinforcement-learning-framework)
2. [Markov Decision Processes (MDPs)](#2-markov-decision-processes-mdps)
3. [Q-Learning & The Bellman Equation](#3-q-learning--the-bellman-equation)
4. [Deep Q-Networks (DQN)](#4-deep-q-networks-dqn)
5. [Policy Gradient Methods](#5-policy-gradient-methods)
6. [Proximal Policy Optimization (PPO)](#6-proximal-policy-optimization-ppo)
7. [Library Implementation (Gymnasium & Stable Baselines3)](#7-library-implementation-gymnasium--stable-baselines3)
8. [Project Ideas & What's Next](#8-project-ideas--whats-next)

---

## 1. The Reinforcement Learning Framework

Machine Learning is historically divided into three branches:
1. **Supervised Learning**: Learning from labeled data (Classification/Regression).
2. **Unsupervised Learning**: Finding hidden structures in unlabeled data (Clustering/PCA).
3. **Reinforcement Learning (RL)**: Learning to make decisions by interacting with an environment to maximize cumulative reward.

**The RL Loop:**
- An **Agent** exists in an **Environment** (e.g., Mario in a video game).
- At time $t$, the Agent observes the current **State** $S_t$ (e.g., the screen pixels).
- The Agent takes an **Action** $A_t$ (e.g., "Jump").
- The Environment responds by giving the Agent a **Reward** $R_{t+1}$ (e.g., +100 points for hitting a coin) and transitions to a new **State** $S_{t+1}$.

The Agent's goal is to learn a **Policy** $\pi(a|s)$—a mapping from states to actions—that maximizes the total reward over time.

---

## 2. Markov Decision Processes (MDPs)

RL is mathematically formalized as a **Markov Decision Process (MDP)**.
An MDP is a 5-tuple $(S, A, P, R, \gamma)$:
- $S$: A set of all possible States.
- $A$: A set of all possible Actions.
- $P(s' | s, a)$: The Transition Probability (probability of ending up in state $s'$ if you take action $a$ in state $s$).
- $R(s, a)$: The Reward function.
- $\gamma$: The **Discount Factor** ($0 \le \gamma \le 1$).

### The Discount Factor ($\gamma$)
A reward of +10 today is better than a reward of +10 next year. We discount future rewards using $\gamma$. 
The cumulative Return $G_t$ at time $t$ is:
$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} $$

- If $\gamma = 0$, the Agent is purely short-sighted (only cares about immediate reward).
- If $\gamma = 0.99$, the Agent is far-sighted (cares about long-term success).

---

## 3. Q-Learning & The Bellman Equation

How does an Agent know if a state is "good"? We define the **Action-Value Function**, or **Q-Function**, $Q(s, a)$.
$Q(s, a)$ is the *expected total return* starting from state $s$, taking action $a$, and then following the optimal policy forever.

To find the optimal $Q$-values, we use the famous **Bellman Optimality Equation**:
$$ Q^*(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \right] $$

In words: The value of taking action $a$ right now is the immediate reward you get, plus the discounted value of whatever the *best possible action* is in the *next* state.

**Tabular Q-Learning**: For tiny games (like Tic-Tac-Toe), we can just build a lookup table (a 2D array) with a row for every state and a column for every action, and update it iteratively.

---

## 4. Deep Q-Networks (DQN)

What if the state is a $256 \times 256$ RGB image (like playing Atari)? The number of possible states is $256^{256 \times 256 \times 3}$, which is larger than the number of atoms in the universe. We cannot use a lookup table.

In 2013, DeepMind solved this by combining Q-Learning with Deep Learning, creating the **DQN**.
Instead of a table, they used a Convolutional Neural Network. The CNN takes the screen pixels as input, and outputs the Q-values for all possible joystick actions (e.g., Up, Down, Left, Right).

### Key Innovations of DQN:
1. **Experience Replay**: Instead of training on consecutive frames (which are highly correlated and destabilize the network), the Agent stores all its experiences $(S, A, R, S')$ in a massive memory buffer. During training, it samples random mini-batches from this buffer.
2. **Target Network**: The Q-learning target $R + \gamma \max Q(S', a')$ uses the same network we are updating, creating a moving target that causes catastrophic oscillation. DeepMind fixed this by using a separate, frozen "Target Network" to calculate the target, updating its weights only once every 10,000 steps.

---

## 5. Policy Gradient Methods

DQN is a **Value-Based** method (it learns the value of states, and derives the policy by just picking the max value). 
However, DQN fails in environments with continuous actions (like steering a steering wheel). 

**Policy Gradient** methods are **Policy-Based**. They don't care about Q-values. They use a Neural Network to directly output the probabilities of actions: $\pi_\theta(a|s)$.

We run an episode, collect the rewards, and then calculate the gradient to update the policy weights $\theta$:
$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) \cdot G_t \right] $$

In words: If an action resulted in a high return $G_t$, increase the probability of taking that action in the future. If it resulted in a low return, decrease the probability.

*(Problem: Pure Policy Gradients suffer from incredibly high variance and slow training).*

---

## 6. Proximal Policy Optimization (PPO)

To fix the variance of Policy Gradients, researchers combined Value-Based and Policy-Based methods into **Actor-Critic** architectures:
- The **Actor** network outputs the policy $\pi(a|s)$.
- The **Critic** network estimates the Value function $V(s)$.

The Actor uses the Critic's estimate as a baseline to reduce variance.

In 2017, OpenAI released **PPO (Proximal Policy Optimization)**. It is currently the default RL algorithm used across the industry (including the RLHF used to train ChatGPT).

PPO prevents the Actor from updating its policy too drastically in a single step. It does this by calculating the ratio between the new policy and the old policy:
$$ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} $$

It then clips this ratio into a small range $[1-\epsilon, 1+\epsilon]$ (where $\epsilon=0.2$). This clipping acts as a "trust region", ensuring that a single lucky/unlucky episode doesn't completely destroy the Neural Network's weights. 

PPO is highly sample-efficient, robust, and much easier to tune than previous algorithms like DDPG or TRPO.

---

## 7. Library Implementation (Gymnasium & Stable Baselines3)

Writing PPO from scratch requires thousands of lines of highly optimized PyTorch code. In practice, we use **Stable Baselines3** on top of **Gymnasium** (the modern standard for RL environments).

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# 1. Create the Environment
# Let's teach an AI to land a lunar module!
env = gym.make("LunarLander-v2", render_mode="rgb_array")

# 2. Instantiate the PPO Agent
# MlpPolicy means we use a standard Feed-Forward Neural Network
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# 3. Train the Agent
print("Starting training for 100,000 timesteps...")
model.learn(total_timesteps=100000)

# 4. Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# 5. Save the model
model.save("ppo_lunar_lander")
env.close()
```

---

## 8. Project Ideas & What's Next

### Project Ideas
- 🟢 **Atari DQN**: Use `Stable Baselines3` to train a `DQN` model on `ALE/Pong-v5`. Watch the agent start by randomly twitching the paddle, and slowly learn to flawlessly defeat the hardcoded opponent.
- 🟡 **Custom Stock Trading Env**: Read the `Gymnasium` documentation on creating Custom Environments. Build a class where the "State" is the last 30 days of Apple stock prices, the "Actions" are [Buy, Hold, Sell], and the "Reward" is the portfolio profit. Train a PPO agent on it!

### What's Next
| Next | Why |
|------|-----|
| [Graph Neural Networks](./02-Graph-Neural-Networks.md) | We've looked at text (1D sequences) and images (2D grids). But what if your data is a highly complex web of relationships (like a Social Network, or the molecular structure of a drug)? We need Graph Neural Networks. |

---

[← Modern NLP With Transformers](../12-NLP/07-Modern-NLP-With-Transformers.md) | [Back to Index](../README.md) | [Next: Graph Neural Networks →](./02-Graph-Neural-Networks.md)
