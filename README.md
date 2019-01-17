# Udacity Deep Reinforcement learning Nanodegree
## Problem statement
**Project Collaboration and Competition**: In this project, we have to make two agents learn to play the game of Tennis. The environment is provided by [Unity-ML agents](https://github.com/Unity-Technologies/ml-agents). The aim of this project is to demonstrate how two or more agents can collaborate with each other to learn to achieve the goal efficiently (in this case to learn to play tennis). 
![Tennis environment](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png)

**NOTE:**
1. This project was completed in the Udacity Workspace, but the project can also be completed on a local Machine. Instructions on how to download and setup Unity ML environments can be found in [Unity ML-Agents Github repo](https://github.com/Unity-Technologies/ml-agents). 
2. The project environment is similar to, but not identical to the Tennis environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md).

## Environment
In this environment, two agents (players) control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of `+0.1`. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of `-0.01`. Thus, the goal of each agent is to keep the ball in play.
The **observation space** consists of **8** variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

### Solving the environment
The task is episodic, and in order to solve the environment, the agents must get an average score of `+0.5` (over **100**` consecutive episodes, after taking the maximum over both agents). Specifically, 
1. After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields **2** (potentially different) scores. We then take the maximum of these **2** scores.
2. This yields a single score for each episode.

The environment is considered solved, when the average (over **100** episodes) of those scores is at least `+0.5`.

## Getting started

1. Download the environment from one of the links below. You need to only select the environment that matches your operating sytem: 
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
   - Max OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
 (*For Windows users*) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
 
 (*For AWS*) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), the please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

## Dependencies
1. Python 3.6
2. Pytorch
3. Unity ML-Agents

## Solution
The code for this project is mostly taken from my implementation for the Udacity Project on [Continuous Control](https://github.com/thedatamonk/Unity-Continuous-Control). Following are the key features of the implementation:
1. The agents learnt from the experience tuples every **20** timesteps and at very update step, the agents learnt **10** times. 
2. Gradient clipping helped improved the training.
```
self.critic_optimizer.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
self.critic_optimizer.step()
```
3. Also, to add a bit of exploration while choosing actions, as suggested in the [DDPG](https://arxiv.org/abs/1509.02971) paper, **Ornstein-Uhlenbeck process** was used to add noise to the chosen actions. 
4. Batch Normalization after all the hidden layers.
5. The score for each episode is calculated by taking the maximum over the individual scores of both the players.

## Running the code
1. After installing all dependcies, clone this repository in your local system.
2. Make sure you have **Jupyter** installed. To install Jupyter:
```
python3 -m pip install --upgrade pip
python3 -m pip install jupyter
```
3. Code structure:
   - `Tennis.ipynb` : Main notebook containing the training function
   - `maddpg.py` : code for **MADDPG** agent
   - `model.py` : code for **Actor** and **Critic** networks
   - `workspace_utils.py` : code to keep the Udacity workspace awake during training
   

## Results

![DDPG score](https://github.com/thedatamonk/Unity-Tennis-Competition/blob/master/plots/score_maddpg.png)
![DDPG score over 100 episodes](https://github.com/thedatamonk/Unity-Tennis-Competition/blob/master/plots/score100_maddpg.png)

The implementation was able to solve the environment in approximately **2050** episodes.
