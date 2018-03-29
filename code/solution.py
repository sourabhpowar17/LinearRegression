import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from helper import *

def sign(x):
	return 1 if x > 0 else -1

#-------------- Implement your code Below -------------#

def show_images(data):
        fig = plt.figure()
        first_image = data[0]
        first_image = np.array(first_image, dtype='float')
        pixelOne = first_image.reshape((16, 16))
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(pixelOne, cmap = 'gray')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        second_image = data[1]
        second_image = np.array(second_image, dtype='float')
        pixelTwo = second_image.reshape((16, 16))
        ax = fig.add_subplot(1, 2, 2)
        ax.matshow(pixelTwo, cmap = 'gray')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.savefig('Image.png')
        plt.show()

def show_features(data, label):
        for index in range(len(data)):
                if label[index] == 1.0:
                        plt.scatter(data[index][0], data[index][1],c='r',marker='*')
                else:
                        plt.scatter(data[index][0], data[index][1],c='b',marker='+')
        plt.xlabel('Symmetry')
        plt.ylabel('Intensity')
        plt.savefig('Features.png')
        plt.show()

def perceptron(data, label, max_iter, learning_rate):
        w = np.zeros((1,len(data[0])))
        eta = learning_rate
        epochs = max_iter
        
        for epoch in range(epochs):
                for i, x in enumerate(data):
                        if (sign(np.dot(data[i,:],np.transpose(w))) != sign(label[i])) :
                                w = w + eta*data[i]*label[i]
        return w

def show_result(data, label, w):
        for index in range(len(data)):
                if label[index] == 1.0:
                        plt.scatter(data[index][0], data[index][1],c='r',marker='*')
                else:
                        plt.scatter(data[index][0], data[index][1],c='b',marker='+')
        plt.xlabel('Symmetry')
        plt.ylabel('Intensity')
        p1 = [0,-w[0][0]/w[0][2]]
        p2 = [-w[0][0]/w[0][1],0]
        ax = plt.gca()
        xmin, xmax = ax.get_xbound()
        if(p2[0] == p1[0]):
                xmin = xmax = p1[0]
                ymin, ymax = ax.get_ybound()
        else:
                ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
                ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

        l = mlines.Line2D([xmin,xmax], [ymin,ymax])
        ax.add_line(l)
        plt.savefig('Result.png')
        plt.show() 
#-------------- Implement your code above ------------#
def accuracy_perceptron(data, label, w):
	n, _ = data.shape
	mistakes = 0
	for i in range(n):
		if sign(np.dot(data[i,:],np.transpose(w))) != label[i]:
			mistakes += 1
	#print("number of records",n)
	#print("mistake",mistakes)
	return (n-mistakes)/n


def test_perceptron(max_iter, learning_rate):
	#get data
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	#train perceptron
	w = perceptron(train_data, train_label, max_iter, learning_rate)
	train_acc = accuracy_perceptron(train_data, train_label, w)	
	#test perceptron model
	test_acc = accuracy_perceptron(test_data, test_label, w)
	return w, train_acc, test_acc


