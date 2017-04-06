import os
import numpy as np
import cv2 as cv
import pickle


class_path = '/home/aswath/Desktop/deeplearning_hw2/VOCdevkit/VOC2012/image_class'
pickle_file = '/home/aswath/Desktop/deeplearning_hw2/pickle'
for filename in os.listdir(class_path):
    img_array = []
    img_array = np.array(img_array)
    image_path = os.path.join(class_path,filename)
    # grp = image_file.create_group(filename)
    pickle_path=os.path.join(pickle_file,filename+'.pkl')
    # if not os.path.isfile(pickle_path):
    #    	os.mknod(filename+'.pkl')
    file = open(pickle_path,'wb')

    for imagename in os.listdir(image_path):
	    if not imagename.endswith('.jpg'): continue
	    img_name = os.path.splitext(imagename)[0]
	    image = cv.imread(image_path + '/' + img_name+'.jpg')
	    # print image
	    img_array = np.append(img_array,image)

    img_array = np.array(img_array)
    pickle.dump(img_array,file)
    print 'dumping'
    img_array=[]
    file.close