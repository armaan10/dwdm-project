import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import numpy as np 

class RnnModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
		super(RNNModel, self).__init__()
		# Hidden dimensions
		self.hidden_dim = hidden_dim

		# Number of hidden layers
		self.layer_dim = layer_dim

		# Building your RNN

		# (batch_dim, seq_dim, input_dim) o/p dim
		# batch_dim = number of samples per batch
		self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

		# last layer
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):

		# (layer_dim, batch_size, hidden_dim)
		h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()


		out, hn = self.rnn(x, h0.detach())

		#only last time step??        
		out = self.fc(out[:, -1, :]) 
		# out.size() --> 100, 10
		return out


critetrion=nn.MSELoss()
lr=0.01
optimizer=torch.