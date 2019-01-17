import numpy as np
import copy
from collections import namedtuple, deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import (Actor, Critic)

# training hyperparameters
lr_actor = 1e-3                    # learning rate of Actor
lr_critic = 1e-3                   # learning rate of Critic
weight_decay = 0.                  # L2 weight decay 
gamma = 0.99                       # discount factor
tau = 1e-3                        # soft update parameter
batch_size = 512                   # batch size to sample from replay buffer
buffer_size = int(1e5)             # max size (capacity) of the replay buffer
n_agents = 2                       # number of agents
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
update_every = 20
update_freq = 10

class MADDPGAgent(object):
    def __init__(self, state_size, action_size, seed):
        super(MADDPGAgent, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        
        # initialise local and target Actor networks
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr = lr_actor)
        
        # initialise local and target Critic networks
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr = lr_critic, weight_decay = weight_decay)
        
        
        # copying the network weights of local model to target model
        # self.hard_update(self.actor_local, self.actor_target)
        # self.hard_update(self.critic_local, self.critic_target)
        
        # initialise the Ornstein-Uhlenbeck noise process
        self.noise = OUNoise((n_agents, action_size), seed)        
        
        
        # Shared Replay Buffer
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)
        
        self.t_step = 0
        
    def hard_update(self, local_model, target_model):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
            
            
    def step(self, states, actions, rewards, next_states, dones):
        
        # each agent adding their experience tuples in the replay buffer
        for i in range(n_agents):
            self.memory.add(states[i, :], actions[i, :], rewards[i], next_states[i, :], dones[i])
            
            
        self.t_step = (self.t_step + 1) % update_every
        if self.t_step == 0:
            # if enough samples are there then learn
            if len(self.memory) > batch_size:
                for i in range(update_freq):
                    experiences = self.memory.sample()
                    self.learn(experiences, gamma)
        
        
    def act(self, states, add_noise=True):
            """Returns actions for given state as per current policy"""
            states = torch.from_numpy(states).float().to(device)
            actions = np.zeros((n_agents, self.action_size))
                               
            self.actor_local.eval()

            # get the actions for each agent
            with torch.no_grad():
                for i in range(n_agents):
                    action_i = self.actor_local(states[i]).cpu().data.numpy()
                    actions[i, :] = action_i
                               
            self.actor_local.train()

            # Ornstein-Uhlenbeck noise process           
            if add_noise:
                actions += self.noise.sample()

            return np.clip(actions, -1, 1)

    
    def reset(self):
        self.noise.reset()
        
        
    def learn(self, experiences, gamma):
        
        states, actions, rewards, next_states, dones = experiences
        
        # update critic
        
        # Get the actions corresponding to next states and then their Q-values
        # from target critic network
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Now minimize this loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        
        # gradient clipping as suggested
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        self.critic_optim.step()
        
        
        # Update Actor
        # Compute Actor loss
        actions_pred = self.actor_local(states)
        
        # -ve sign because we want to maximise this value
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # minimizing the loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        
        # update target networks
        self.soft_update(self.critic_local, self.critic_target, tau)
        self.soft_update(self.actor_local, self.actor_target, tau)
        
    def soft_update(self, local_model, target_model, tau):
        """
            Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target
            Params
            ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

        
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
        
class OUNoise(object):
    """Ornstein-Uhlenbeck process"""
    
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """ initialise noise parameters """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = 1e-2
        self.seed = torch.manual_seed(seed)
        self.reset()
        
    def reset(self):
        """reset the internal state to mean (mu)"""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update internal state and return it as a noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.array([np.random.normal() for i in range(len(x))])
        self.state = x + dx
        return self.state
    
    
    
class ReplayBuffer(object):
    """Replay buffer to store experience tuples"""
    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    

