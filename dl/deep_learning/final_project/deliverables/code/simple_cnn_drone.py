# Carlos Morato, PhD.
# cwmorato@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Simple CNN for the MNIST Dataset
import numpy
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
import keras.backend as K
K.set_image_dim_ordering('th')
import theano
from keras.callbacks import ModelCheckpoint

num_classes = 5
batch_size = 50
# added image enhancing
train_datagen = ImageDataGenerator(rotation_range=5.)
				             
#train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory('./train1',target_size = (224,224),batch_size = batch_size)
test_generator = test_datagen.flow_from_directory('./test1',target_size = (224,224),batch_size = batch_size)

def baseline_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape=(3, 224, 224), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# build the model
model = baseline_model()
checkpoint = ModelCheckpoint('simple_cnn_drone.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Fit the model
history = model.fit_generator(train_generator,samples_per_epoch = 3460,nb_epoch = 25,validation_data = test_generator,nb_val_samples = 600,verbose = 1 , callbacks=[checkpoint])
print ('accuracy is',history.history['acc'])
# Final evaluation of the model


