 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical

import matplotlib.pyplot as plt
from IPython.display import clear_output

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, "PPO/")


class Memory:
	def __init__(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.is_terminals = []
		self.log_probs = []
		self.values = []

		self.states = []
		self.actions = []
		self.rewards = []
		self.is_terminals = []
		self.log_probs = []
		self.values = []

class PPOAgent(object):
	def __init__(self, env, observ_dim, actor_lr=1e-4, critic_lr=5e-4):
		self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
		print("device:", self.device)
		self.env = env

		self.observ_dim = observ_dim

        
