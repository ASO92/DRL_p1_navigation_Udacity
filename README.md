# Project 1: Navigation
## 1. Introduction
This project is the first sumbission task of the
[Deep Reinforcement Learning Nanodegree program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), by Udacity, and in collaboration with Unity and Nvidia.

The main goal of this project is to train an agent to navigate and collect yellow bananas and avoid blue bananas in a large ,square world. This agent is trained under DRL techniques concretely following the


Here it is possible to see the agent collectin bananas and taking its own decisions.
![Example trained agent](./assets/collecting_bananas.gif)

#### About Deep Reinforcement learning

> [Reinforcement learning](https://pathmind.com/wiki/deep-reinforcement-learning) refers to goal-oriented algorithms, which learn how to attain a complex objective (goal) or maximize along a particular dimension over many steps, always through the interaction with the environment; for instance, maximize the points won in a game over many moves.
These algorithms can start their learning from scratch, and under the right learning pattern, they can achieve performances over the human possibilities. Like a dog is incentivized by a biscuit-bone, these algorithms are rewarded when they make the right decisiones and penalized when the make the wrong ones - this is the Reinforcement part.
On the other hand, the part that helps to the agent to take decisions based on its own experience is based on [Neural Network Algorithms](https://pathmind.com/wiki/neural-network) - this is the deep learning part.

This project implement a Value Based method called [Deep Q-Networks](https://deepmind.com/research/dqn/)

## 2. First steps in the environment
All the project has been built and developed using the **Unity Machine Learning Agents Toolkit (ML-Agents)**, which is an open-source Unity plugin that enables games and simulations to serve as envirnments for training intelligent agents. You can read further about ML-agents by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents).

### Environment details

Note: The project environment provided by Udacity is similar to, but not indentical to the [Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) environment.

> The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent. Agents can be trained using reinforcement learning, imitation learning, neuroevolution, or other machine learning methods through a simple-to-use Python API.


A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.

Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- `0` - move forward
- `1` - move backward
- `2` - turn left
- `3` - turn right

#### Solving the environment
To solve the environment, the Agent must obtain an average score of **+13** over 100 consecutive episodes.

## 3. Included in this repository
* The code used to train the Agent
  * NAvigation.ipynb
