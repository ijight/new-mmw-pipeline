#!/usr/bin/env python

import rospy
import torch
import numpy as np
from sensor_msgs.msg import Imu, PointCloud2
import ros_numpy  # For converting PointCloud2 to numpy array

# Define the model class if you have it
# from your_model_file import YourModelClass  # Import your model class if not using a standard model

# Load the PyTorch model
# model = YourModelClass()  # Replace with your model class if necessary
# model = torch.load('model.pth', map_location=torch.device('cpu'))  # Ensure the model is loaded on the correct device
# model.eval()  # Set the model to evaluation mode

def imu_callback(data):
    print(data)

def pointcloud_callback(data):
    print(data)

def listener():
    rospy.init_node('listener', anonymous=True)
    
    # Subscribe to IMU and PointCloud2 topics
    rospy.Subscriber("imu_topic", Imu, imu_callback)  # Change "imu_topic" to your IMU topic name
    rospy.Subscriber("pointcloud_topic", PointCloud2, pointcloud_callback)  # Change "pointcloud_topic" to your PointCloud2 topic name
    
    rospy.spin()

if __name__ == '__main__':
    listener()
