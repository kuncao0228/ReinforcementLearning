 
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
		x = F.relu(self.fc(state))
		mu = torch.tanh(self.mu(x))
		log_std = torch.tanh(self.log_std(x))

		std = torch.exp(log_std)
		
		dist = Normal(mu, std)
		action = dist.sample()
		return action, dist


