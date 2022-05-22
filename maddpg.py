from ddpg_agent import Agent
import numpy as np
import torch
import random
from collections import namedtuple, deque

from params import *

device = 'cpu'

class MADDPG:
    """A wrapper object encapsulating mutliple agents interacting in a mixed cooperative competetive environment."""
    
    def __init__(self, state_size, action_size, num_agents):
        super(MADDPG, self).__init__()
        """Initializes a Multi-Agent object.
        Params
        ======
            state_size (int):
            action_size (int):
            num_agents (int):
        """
        # Initialize agents
        self.random_seed = SEED
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.agents = [Agent(self.state_size, self.action_size, self.random_seed,\
                            fc1_units=FC1_UNITS, fc2_units=FC2_UNITS, fcs1_units=FCS1_UNITS, fcs2_units=FCS2_UNITS, \
                            num_agents=num_agents) for _ in range(num_agents)]
        
        # Initialize a shared replay buffer
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.random_seed)

    def reset(self):
        """Resets each agent's noise."""
        for agent in self.agents:
            agent.reset()

    def act(self, states, add_noise=True, noise_decay=1.0):
        """Returns actions for given state as per current policy."""
        return [np.reshape(agent.act(state, add_noise, noise_decay), -1) for agent, state in zip(self.agents, states)]
        
    def add_to_buffer(self, states, actions, rewards, next_states, dones):
        """Save experiences of multiple agents in a shared replay buffer"""
        
        # concatenate the states of each agent to a shared state of all agents
        full_states = np.reshape(states, newshape=(-1))
        next_full_states = np.reshape(next_states, newshape=(-1))
                
        self.buffer.add(states, full_states, actions, rewards, next_states, next_full_states, dones)

    def train(self):
        """Updates each actor and the shared critic."""
        if len(self.buffer) >= self.batch_size:
            for (agent, agent_id) in zip(self.agents, range(self.num_agents)):
                experiences = self.buffer.sample()
                agent.learn(experiences, agent_id)

    def save(self, dir='ckpts/', identifier=''):
        """Saves all networks."""
        torch.save(self.agents[0].actor_local.state_dict(), dir + 'checkpoint_agent_0_actor'+ identifier +'.pth')
        torch.save(self.agents[0].critic_local.state_dict(), dir + 'checkpoint_agent_0_critic'+ identifier +'.pth')

        torch.save(self.agents[1].actor_local.state_dict(), dir + 'checkpoint_agent_1_actor'+ identifier +'.pth')
        torch.save(self.agents[1].critic_local.state_dict(), dir + 'checkpoint_agent_1_critic'+ identifier +'.pth')

    def load(self, agent_0_actor, agent_0_critic, agent_1_actor, agent_1_critic):
        """Restores all agents."""
        self.agents[0].actor_local.load_state_dict(torch.load(agent_0_actor, map_location='cpu'))
        self.agents[0].critic_local.load_state_dict(torch.load(agent_0_critic, map_location='cpu'))
        
        self.agents[1].actor_local.load_state_dict(torch.load(agent_1_actor, map_location='cpu'))
        self.agents[1].critic_local.load_state_dict(torch.load(agent_1_critic, map_location='cpu'))

            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "full_state", "action", "reward", "next_state", "next_full_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, states, full_states, actions, rewards, next_states, next_full_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, full_states, actions, rewards, next_states, next_full_states, dones)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        full_states = torch.from_numpy(np.vstack([e.full_state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_full_states = torch.from_numpy(np.vstack([e.next_full_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, full_states, actions, rewards, next_states, next_full_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)