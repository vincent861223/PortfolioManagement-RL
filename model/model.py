import torch.nn as nn

class Actor(nn.Module):
	def __init__(self, obs_space):
		super(Actor, self).__init__()


	def forward(self, x):
		close_price = x[0]
		