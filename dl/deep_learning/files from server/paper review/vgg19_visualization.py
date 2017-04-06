from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import keras.backend as K
import theano
K.set_image_dim_ordering('th')

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
model.add(Dense(1000, activation='softmax'))

model.load_weights('vgg19_weights.h5')

#####if __name__ == "__main__":
with open('labelout.txt','rb') as f:
  my_list = pickle.load(f)
im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
im[:,:,0] -= 103.939
im[:,:,1] -= 116.779
im[:,:,2] -= 123.68
im = im.transpose((2,0,1))
im = np.expand_dims(im, axis=0)

# Test pretrained model
#####model = VGG_19('vgg19_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
out = model.predict(im)
print my_list[np.argmax(out)]

conv_layers = [conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8,conv9,conv10,conv11,conv12,conv13,conv14,conv15,conv16]
#maxp_layers = [maxp1,maxp2,maxp3,maxp4,maxp5]

fig = plt.figure()

l=0
for layer in conv_layers:
  l += 1
  out_f = K.function([model.layers[0].input], [layer.output])
  conv_layer_out = out_f([im])
  
  #  for i in range(conv_layer_out[0][0].shape[0]):
  for i in range(1,26):
    tmp = conv_layer_out[0][0,i,:,:]/conv_layer_out[0][0,i,:,:].max()
    fig.add_subplot(5,5,i)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.imshow(tmp, cmap=matplotlib.cm.binary)
  plt.savefig('vgg19_%02d.jpg'%(l))
  plt.clf()  

