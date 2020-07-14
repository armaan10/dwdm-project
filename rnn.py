import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#################################################Model and Method Definitions##################################################
class LSTMModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
		super(LSTMModel, self).__init__()
		# Hidden dimensions
		self.hidden_dim = hidden_dim
		# Number of hidden layers
		self.num_layers = num_layers
		# Building your LSTM
		# (batch_dim, seq_dim, input_dim) o/p dim
		# batch_dim = number of samples per batch
		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
		# last layer
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):

		# (num_layers, batch_size, hidden_dim)
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
		lstm_out, (ht,ct) = self.lstm(x, (h0,c0))
		#only last time step??        
		out = self.fc(lstm_out) 
		# out.size() --> 100, 10
		return out

def train_model(model, epochs=10, lr = 0.01):
	loss_values = []
	val_loss_values = []
	parameters = filter(lambda p: p.requires_grad, model.parameters())
	optimiser = torch.optim.Adam(parameters, lr, betas=(0.9,0.99))
	for e in range(epochs):
		model.train()
		running_loss = 0.0
		total = 0
		for x, y, in train_dataloader:
			x = x.float()
			y = y.float() 
			y_pred = model(x)
			optimiser.zero_grad()
			loss = criterion(y_pred, y)
			loss.backward()
			optimiser.step()
			running_loss = loss.item()*y.size(0)
			total+=y.size(0)
		epoch_loss = running_loss/total
		loss_values.append(epoch_loss)
		val_loss = validation_metrics(model, val_dataloader)
		val_loss_values.append(val_loss)
		if e%5 == 0:
			print("Train mse: %.3f Val mse %.3f" %(epoch_loss, val_loss))
			#torch.save({
			#	'epoch': e,
			#	'model_state_dict': model.state_dict(),
			#	'optimizer_state_dict': optimiser.state_dict(),
			#	'loss': epoch_loss,
			#}, '<path>'+str(e)+'.pth.tar')
	plt.plot(epoch_loss)
	plt.plot(val_loss_values, color = 'orange')
	plt.show()
		
def validation_metrics(model, val_dataloader):
	model.eval()
	running_loss = 0.0
	total = 0
	for x, y, in val_dataloader:
		x = x.float()
		y = y.float()
		y_pred = model(x)
		loss = criterion(y_pred, y)
		total+=y.size(0)
		running_loss = loss.item()*y.size(0)
		return running_loss/total

###############################################################################################################################
if __name__ == "__main__":

	##########################Input pipeline##################################
	#Join CSV files to create feature vectors for training
	df_ips=[]
	df_op=1
	path="/home/sidharth/dwdm-project-irdc/dwdm-project-irdc"
	ind_df=np.zeros((4494,1))
	for i in range(0,4494):
		ind_df[i,0]=i
	ind_df=pd.DataFrame({'Index':ind_df[:,0]})
	for i in os.listdir(path):
		if ".csv" in i:
			if i=="Control_Link_Angles.csv":
				df_op=pd.read_csv(os.path.join(path,i))
			else:
				df_ips.append(pd.read_csv(os.path.join(path,i)).drop(['Time (sec)'],axis=1).join(ind_df))

			
	ip_set=df_ips[0]
	print(ind_df)


	for i in range (len(df_ips)):
		if i==0:
			continue
		ip_set=pd.merge(ip_set,df_ips[i],on='Index',how='inner')

		#print (ip_set)
	#ip_set=ip_set.drop(['Time (sec)'],axis=1)
	df_op=df_op.drop(['Time (sec)'],axis=1)
	ip_set=ip_set.drop(['Index'],axis=1)
	#print(ip_set.iloc[0])
	#print(df_op)

	#Preprocess the data by scaling inputs
	dataset = ip_set.to_numpy()
	scaler = StandardScaler()
	scaler.fit(dataset)
	dataset = scaler.transform(dataset)
	dataset = dataset.reshape((107,42,11))

	df_op = df_op.to_numpy()
	df_op = df_op.reshape((107,42,4))

	train_data, test_data_temp, train_labels, test_labels_temp = train_test_split(dataset, df_op, test_size = 0.3)
	test_data, val_data, test_labels, val_labels = train_test_split(test_data_temp, test_labels_temp, test_size = 0.25)

	#Create pytorch dataloader objects
	train=TensorDataset(torch.from_numpy(train_data),torch.from_numpy(train_labels))
	val=TensorDataset(torch.from_numpy(val_data),torch.from_numpy(val_labels))
	train_dataloader = DataLoader(train, batch_size = 32, shuffle = False)
	val_dataloader = DataLoader(val, batch_size = 32, shuffle = False) 

	print(train_data.shape,train_labels.shape)
	#Define loss and pass model to device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = LSTMModel(11, 20, 1, 4).to(device)
	criterion = nn.MSELoss()
	train_model(model)