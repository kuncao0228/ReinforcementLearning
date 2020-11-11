import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np



class Critic(nn.Module):
	def __init__(self, input_dim):
		super(Critic, self).__init__()
		self.fc = nn.Linear(input_dim, 64)
		self.out = nn.Linear(64, 1)


	def forward(self, state):
		x = F.relu(self.fc(state))
		value = self.out(x)

		return value


