import cv2
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout,BCEWithLogitsLoss
from matplotlib import pyplot as plt
from torch.optim import Adam, SGD
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h=50
w=50
c=3
images=[]
labels=[]
positive_data_dir="./8863/1"
neg_data_dir="./8863/0"
for img in os.listdir(neg_data_dir):
	img_arr=cv2.imread(os.path.join(neg_data_dir,img))/255.0
	if img_arr.shape ==(50,50,3):
		img_arr = img_arr.astype('float32')
		images.append(img_arr)
		labels.append(0)


for img in os.listdir(positive_data_dir):
	img_arr=cv2.imread(os.path.join(positive_data_dir,img))/255
	img_arr = img_arr.astype('float32')
	labels.append(1)
	images.append(img_arr)
images=np.array(images)
labels=np.array(labels)

x_train,x_val,y_train,y_val = train_test_split(images,labels,test_size=0.15)
#train
x_train=np.reshape(x_train, (len(x_train),3,50,50))
x_train=torch.from_numpy(x_train)

y_train=y_train.astype('int')
y_train=torch.from_numpy(y_train)

#val

x_val=np.reshape(x_val, (len(x_val),3,50,50))
x_val=torch.from_numpy(x_val)

y_val=y_val.astype('int')
y_val=torch.from_numpy(y_val)
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
class Net(Module):
	
	def __init__(self):
		super(Net, self).__init__()
		
		self.cnnlayers=Sequential(
			Conv2d(3, 64, kernel_size=3, stride=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, stride=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 128, kernel_size=3, stride=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=3, stride=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            )
		self.linear_layers= Sequential(
			Linear(10368, 256),
			ReLU(inplace=True),
			Linear(256,1),
			)
	def forward(self,x):
		x=self.cnnlayers(x)
		x=x.view(x.size(0),-1)
		print(x.size())
		x=self.linear_layers(x)
		return x
model = Net()
model.to(device)
print(model)
criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.001)
model.train()
acc=1
train_loss=[]
for i in range(1,100):
	
	x_train,y_train=x_train.to(device),y_train.to(device)
	optimizer.zero_grad()
	y_pred=model(x_train)
	loss=criterion(y_pred.float(), y_train.unsqueeze(1).float())
  #acc = binary_acc(y_pred, y_batch.unsqueeze(1))
	loss.backward()
	optimizer.step()

	epoch_loss = loss.item()
	train_loss.append(epoch_loss)
	#epoch_acc += acc.item()
	if i%2 == 0:
	# printing the train loss
		print('Epoch : ',i, '\t', 'loss :', epoch_loss)
#print("Accuracy",acc.item())
#plt.plot(train_loss, label='Training loss')
#plt.legend()
#plt.show()
model.eval()
x_val=x_val.to(device)
y_val=y_val.to(device)
preds=model(x_val)
print("Binary acc",binary_acc(preds,y_val.unsqueeze(1)).item())
preds=preds.cpu()
preds=preds.detach().numpy()>0.5
print(accuracy_score(y_val.cpu(),preds))