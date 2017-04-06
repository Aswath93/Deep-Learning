#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

import time
import cv2, numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import keras.backend as K
K.set_image_dim_ordering('th')
import theano

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False


batch_size = 50

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
conv1 = Convolution2D(64, 3, 3, activation='relu')
model.add(conv1)
model.add(ZeroPadding2D((1,1)))
conv2 = Convolution2D(64, 3, 3, activation='relu')
model.add(conv2)
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
conv3 = Convolution2D(128, 3, 3, activation='relu')
model.add(conv3)
model.add(ZeroPadding2D((1,1)))
conv4 = Convolution2D(128, 3, 3, activation='relu')
model.add(conv4)
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
conv5 = Convolution2D(256, 3, 3, activation='relu')
model.add(conv5)
model.add(ZeroPadding2D((1,1)))
conv6 = Convolution2D(256, 3, 3, activation='relu')
model.add(conv6)
model.add(ZeroPadding2D((1,1)))
conv7 = Convolution2D(256, 3, 3, activation='relu')
model.add(conv7)
model.add(ZeroPadding2D((1,1)))
conv8 = Convolution2D(256, 3, 3, activation='relu')
model.add(conv8)
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
conv9 = Convolution2D(512, 3, 3, activation='relu')
model.add(conv9)
model.add(ZeroPadding2D((1,1)))
conv10 = Convolution2D(512, 3, 3, activation='relu')
model.add(conv10)
model.add(ZeroPadding2D((1,1)))
conv11 = Convolution2D(512, 3, 3, activation='relu')
model.add(conv11)
model.add(ZeroPadding2D((1,1)))
conv12 = Convolution2D(512, 3, 3, activation='relu')
model.add(conv12)
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
conv13 = Convolution2D(512, 3, 3, activation='relu')
model.add(conv13)
model.add(ZeroPadding2D((1,1)))
conv14 = Convolution2D(512, 3, 3, activation='relu')
model.add(conv14)
model.add(ZeroPadding2D((1,1)))
conv15 = Convolution2D(512, 3, 3, activation='relu')
model.add(conv15)
model.add(ZeroPadding2D((1,1)))
conv16 = Convolution2D(512, 3, 3, activation='relu')
model.add(conv16)
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))

model.load_weights('./weights_no_rot.hdf5')

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
print "model compiled"

class drone:
	def __init__(self):
		self.vel = 0.3;
		self.command = Twist()
		self.vel_pub = rospy.Publisher("/cmd_vel",Twist,queue_size = 10)
		self.land_pub = rospy.Publisher("ardrone/land",Empty,queue_size = 1)
		self.image_sub = rospy.Subscriber("ardrone/front/image_raw",Image,self.get_image)
	
	def set_vel(self,x,y,z,az):
		self.command.linear.x = x
		self.command.linear.y = y
		self.command.linear.z = z
		self.command.angular.x = 0
		self.command.angular.y = 0	
		self.command.angular.z = az
	
	def get_image(self,data):
		try:
			self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

	def send_command(self):
		# send the current twist commands and then set them to zero
		# print "sending command"
		# print self.command
		self.vel_pub.publish(self.command)
		self.set_vel(0,0,0,0)
	
	def go_straight(self):
		print "going straight"
		self.set_vel(self.vel,0,0,0)
		self.send_command()
	
	def turn_left(self):
		self.set_vel(0,0,0,-self.vel)
		self.send_command()

	def turn_right(self):
		self.set_vel(0,0,0,self.vel)
		self.send_command()

	def slide_left(self):
		self.set_vel(0,-self.vel,0,0)
		self.send_command()

	def slide_right(self):
		self.set_vel(0,self.vel,0,0)
		self.send_command()
	
	def hover(self):
		self.send_command()

	def predict(self):
		im = cv2.resize(self.cv_image, (224, 224)).astype(np.float32)
		im = im.transpose((2,0,1))
		im = np.expand_dims(im, axis=0)
		out = model.predict(im)
		return np.argmax(out)
		# return 2

def main():
	ardrone = drone()
	rospy.init_node('ardrone_deep', anonymous=True)
	# while not rospy.is_shutdown():
	# rospy.spinonce()
	rospy.Rate(10)
	print "spinning"
	pred = 0
	while not rospy.is_shutdown():
		pred = ardrone.predict()
		print "predicting",pred
		if pred == 1:
			ardrone.go_straight()
		elif pred == 2:
			ardrone.slide_left()
		elif pred == 3:
			ardrone.slide_right()
		elif pred == 4:
			ardrone.turn_left()
		elif pred == 5:
			ardrone.turn_right()
		else:
			ardrone.hover()
	pred = 0

if __name__ == '__main__':
    main()