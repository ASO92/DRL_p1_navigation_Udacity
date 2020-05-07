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
* The code used to create and train the Agent
  * Navigation.ipynb
  * dqn_agent.py
  * model.py
* The trained model weights
  * checkpoint.pth
* A Report.md file describing the development process and the learning algorithm, along with ideas for future work
* This README.md file to explain the environment configuration characteristics

## 4. Setting up the environment

This section describes how to get the code for this project and configure the environment.

### Getting the code
You have two options to get the code contained in this repository:
##### Option 1. Download it as a zip file

* [Click here](https://github.com/ASO92/DRL_p1_navigation_Udacity/archive/master.zip) to download all the content of this repository as a zip file
* Uncompress the downloaded file into a folder of your choice

##### Option 2. Clone this repository using Git version control system
If you are not sure about having Git installed in your system, run the following command to verify that:

```
$ git --version
```
If you need to install it, follow [this link](https://git-scm.com/downloads) to do so.

Having Git installed in your system, you can clone this repository by running the following command:

```
$ git clone https://github.com/ASO92/DRL_p1_navigation_Udacity.git
```
### Installing Miniconda / Anaconda
You can skip this step in case that you have installed Miniconda or Anaconda version.

Miniconda is a free minimal installer for conda. It is a small, bootstrap version of Anaconda that includes only conda, Python, the packages they depend on, and a small number of other useful packages, including pip, zlib, and a few others.  

If you would like to know more about Anaconda, visit [this link](https://www.anaconda.com/).

In the following links, you find all the information to install **Miniconda** (*recommended*)

* Download the installer: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
* Installation Guide: [https://conda.io/projects/conda/en/latest/user-guide/install/index.html](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

Alternatively, you can install the complete Anaconda Platform.

* Download the installer: [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)
* Installation Guide: [https://docs.anaconda.com/anaconda/install/](https://docs.anaconda.com/anaconda/install/)

### Environment installation requirements
This step is intended to be done in order to install the dependencies (packages) necessary to run the environment in your machine.

- You first need to configure a Python 3.6 / PyTorch 0.4.0 environment with the needed requirements as described in the [Udacity repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)
- Of course you have to clone this project and have it accessible in your Python environment
- Then you have to install the Unity environment as described in the [Getting Started section](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md) (The Unity ML-agant environment is already configured by Udacity)

  - Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.


- Finally, unzip the environment archive in the 'project's environment' directory and eventually adjust thr path to the UnityEnvironment in the code.

### Uninstall
If you wish to revert all the modifies in your system, and remove all the code, dependencies and programs installed in the steps above, you will want to follow the next steps.

#### Uninstall Miniconda or Anaconda
To do so, please refer to [this link](https://docs.anaconda.com/anaconda/install/uninstall/).


#### Remove the code
Simply delete the entire folder containing the code you downloaded in the step "Getting the code"


## 5. Train a agent

Execute the notebook Navigation.ipynb included in the repository. Inside there will be specific cells with instructions for:
- Importing the necessary packages
- Define the path to the specific executable of the collecting_bananas environments
- Examine the State and Action Spaces
- Take Random actions in the environment
- The training agent part
- The different tests created to evaluate performance characteristics of the algorithm. From hyperparameters evaulation to enhanced algorithm evaluation (DDQN, Duelling).

Note :
- Manually playing with the environment has not been implemented as it is not available with Udacity Online Worspace (No Virtual Screen)    
- Watching the trained agent playing in the environment has not been implemented neither, as it is not available with Udacity Online Worspace (No Virtual Screen) and not compatible with my personal setup (see Misc : Configuration used  section)
- To do so, it is necessary to play with the agent in your local machine, configuring the environment, with the steps described above.
