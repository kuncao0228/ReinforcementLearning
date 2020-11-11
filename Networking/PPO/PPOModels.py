 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np

class ContinuousActor(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(ContinuousActor, self).__init__()
		self.fc = nn.Linear(input_dim, 32)
		self.mu = nn.Linear(32, output_dim)
		self.log_std = nn.Linear(32, output_dim)


	def forward(self, state):
		x = F.relu(self.hidden(state))
		mu = torch.tanh(self.mu(x))
		log_std = torch.tanh(self.log_std(x))

		std = torch.exp(log_std)
		
		dist = Normal(mu, std)
		action = dist.sample()
		return action, dist

class Critic(nn.Module):
	def __init(self, input_dim):
		super(Critic, self).__init__()
		self.fc = nn.Linear(input_dim, 64)
		self.out = nn.Linear(64, 1)


	def forward(self, state):
		x = F.relu(self.hidden(state))
		value = self.out(x)

		return value

def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.0)

