import cv2
import numpy as np
from PIL import Image
import os
import time
#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
count =0
dataset_path = 'knn_examples/train/'
file_name = input("Enter the name of the person : ")
dataset_path=dataset_path+file_name
os.mkdir(dataset_path)
while count<10:
    ret,frame = cap.read()
    temp = dataset_path + "/frame"+str(count)+".jpg"
    cv2.imwrite(temp, frame)
    print('Read a new frame: ',ret)
    count += 1
    time.sleep()
cap.release()
cv2.destroyAllWindows()
