#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty

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
		return 2

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