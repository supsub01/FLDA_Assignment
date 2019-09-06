import matplotlib
import matplotlib.pyplot as mp
import numpy as np

#data extraction ####################### datasets in the same directory ########################################
with open ("dataset_3.csv",'r') as target:
	data = target.read().splitlines()
for i in range(len(data)):
	data[i] = ['1']+ data[i].split(',')
	data[i] = [data[i][0]]+data[i][2:]

data = data[500:]+data[:500]
train_features, train_labels = [line[:3] for line in data], [line[-1] for line in data]
train_features, train_labels = np.array(train_features).astype(float),np.array(train_labels).astype(float)
################################################################################################################

#initializion of perceptron weights and biases
weights = np.random.rand(3) # 2 weights and a bias

#perceptron function ###########################################################################################
def perceptron_predict(feaures,weights):
	a = np.dot(features,weights)
	if a > 0:
		output = 1
	if a <= 0:
		output = 0
	return output
################################################################################################################

#training
lr = 0.1 #learning rate

for iteration in range(25):  #No. of epochs
	count = 0
	for features, label in zip(train_features, train_labels):
		pred = perceptron_predict(features,weights)
		if label == pred:
			count += 1
		weights += lr*(label-pred)*features  #weights are updated only if incorrectly classified

#   plots
	colors = ['red','blue']	
	mp.scatter(train_features[:,1],train_features[:,2], c=train_labels, cmap=matplotlib.colors.ListedColormap(colors))		
	mp.plot(train_features[:,1],-(weights[1]/weights[2])*train_features[:,1] - (weights[0]/weights[2]))
	mp.savefig('IMG_'+str(iteration)+'.png')
	mp.close()
print(count)   #print the number of correctly classified inputs
print(weights) #print weight vector
################################################################################################################