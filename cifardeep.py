import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.cifar10 as cifar
import numpy as np 

(X,Y),(test_x,test_y)=cifar.load_data(one_hot=True)
#X=np.array(X)
#test_x=np.array(test_x)
#test_y=np.array(test_y)
#Y=np.array(Y)

X=X.reshape([-1,32,32,3])
test_x=test_x.reshape([-1,32,32,3])

convnet=input_data(shape=[None,32,32,3],name='input')

convnet=conv_2d(convnet,32,3,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,3,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,3,activation='relu')
convnet=conv_2d(convnet,128,3,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,512,activation='relu')
convnet=fully_connected(convnet,512,activation='relu')
convnet=dropout(convnet,0.8)


convnet=fully_connected(convnet,10,activation='softmax')
convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy')

model=tflearn.DNN(convnet)
'''
model.fit(X,Y,n_epoch=10,validation_set=(test_x,test_y),batch_size=100,snapshot_step=1000,show_metric=True)

model.save('tflearn.model')
'''
model.load('tflearn.model')


from PIL import Image

img = Image.open('test.jpg')
img = img.resize([32,32]) 
img=np.array(img)
ans=model.predict([img])
print "*************\n"
if(ans[0][0]==1):
	print "Airplane"
if(ans[0][1]==1):
	print "Automobile"
if(ans[0][2]==1):
	print "Bird"
if(ans[0][3]==1):
	print "Cat"
if(ans[0][4]==1):
	print "Deer"
if(ans[0][5]==1):
	print "Dog"
if(ans[0][6]==1):
	print "Frog"
if(ans[0][7]==1):
	print "Horse"
if(ans[0][8]==1):
	print "Ship"
if(ans[0][9]==1):
	print "Truck"
print "\n",
print "*************"
print "Above predictions are subjected to 75 percent accuracy\n"

