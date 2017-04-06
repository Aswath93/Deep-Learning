import os
import cv2
import xml.etree.ElementTree as ep
import random
# parsing throgh all the files in the directort
path = '/home/aswath/Desktop/deeplearning_hw2/VOCdevkit/VOC2012/Annotations'
imagepath = '/home/aswath/Desktop/deeplearning_hw2/VOCdevkit/VOC2012/JPEGImages'
class_path = '/home/aswath/Desktop/deeplearning_hw2/VOCdevkit/VOC2012/image_class'
for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    img_name = os.path.splitext(filename)[0]
    tree = ep.parse(fullname)
    root = tree.getroot() 
    image = cv2.imread(imagepath + '/' + img_name+'.jpg')

    for child in root.iter('object'):
        # getting the name of the class
        class_name = child.find('name').text
        # finding the values for the bounding box
        for box in child.iter('bndbox'):
        	xmin = int(float(box.find('xmin').text))
        	ymin = int(float(box.find('ymin').text))
        	xmax = int(float(box.find('xmax').text))
        	ymax = int(float(box.find('ymax').text))

        cv2.imshow("image",image)
        crop_image = image[ymin:ymax,xmin:xmax]
        resize_image = cv2.resize(crop_image,(224,224),None,fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
        final_path = os.path.join(class_path,class_name)
        # saving the images to the file
        if not os.path.isdir(final_path):
        	os.mkdir(final_path)
        cv2.imwrite(final_path+'/'+img_name+'_'+str(random.random())+'.jpg',resize_image)
        print final_path , img_name
        

        



