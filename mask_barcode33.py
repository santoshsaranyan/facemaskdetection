# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 21:02:47 2021

@author: Santosh Saranyan 
"""


# Importing the needed packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import pandas as pd
from datetime import datetime
from pyzbar.pyzbar import decode

# Function to decode the QR Code
def qrcode(frame):
    img = frame
    result = decode(img)
    if (result!=[]):
        name2 = result[0].data.decode("utf-8")
        return name2
    else:
        name2="Unknown Person"
        return name2

    
# Function to store the results of the classification
def arraycall(name3):
    print(mask_count, ppl, nomask_count, inc_count)
    if (mask_count==max(mask_count, nomask_count, inc_count)):
        # If the classifier has predicting that in most of the frames in the video, the person is wearing a mask
                small_array=["Mask",1, dt_string, name3]
                big_array.append(small_array)
                small_array=[0,0,0,0]
                print("Mask")
    elif (nomask_count==max(mask_count, nomask_count, inc_count)):
        # If the classifier has predicting that in most of the frames in the video, the person is not wearing a mask
                small_array=["Without Mask",1, dt_string, name3]
                big_array.append(small_array)
                small_array=[0,0, 0, 0]
                print("No mask")
    else:
        # If the classifier has predicting that in most of the frames in the video, the person is wearing a mask incorrectly
                small_array=["incorrectly Worn",1, dt_string, name3]
                big_array.append(small_array)
                small_array=[0,0, 0, 0]
                print("Incorrectly worn")

def detect_and_predict_mask(frame, faceNet, maskNet):
    # Grabs the dimensions of the frame and then constructs a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # Passes the blob through the network and obtains the face detections (using the SSD-RESNET Model)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    #print(detections.shape)

    # Initialize the list of faces, their corresponding locations, and the list of predictions from the facemask network
    faces = []
    locs = []
    preds = []

    # Loop over all the detections
    for i in range(0, detections.shape[3]):
        # Extract the confidence of the model (the probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # Computes the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensurse the bounding boxes fall within the dimensions of the input video's frame 
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extracts the face ROI, convert it from BGR to RGB channel
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # Resizes to 224x224, and preprocess it to fit with MobileNet's input standards
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Adds the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Only make a predictions if at least one face was detected
    if len(faces) > 0:
        # Batch Prediction used for faster inference
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # Returns a tuple of the face locations and their corresponding locations
    return (locs, preds)

# Loads the face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Loads the facemask detector model from disk
maskNet = load_model("mask_detector5.model")

# Initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# Intialize the various counts used
mask_count=0
mask2_count=0
nomask_count=0
nomask2_count=0
inc_count=0
inc2_count=0
frame_count=0
zero_count=0
big_array=[]
final_array=[]
small_array=[0,0,0,0]
dt_string=""
name="Unknown Person"
# loop over the frames from the video stream
while True:
    frame_count+=1
    # Get the frame from the video stream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Detect faces in the frame and determine if they are wearing a face mask or not, or incorrectly wearing one
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    
    ppl=len(locs)
    #print(frame)
    #print(ppl)
    
    # If there is no one/face in the video frame
    if ppl==0:
        #print(mask_count, ppl, nomask_count, inc_count)
        if(mask_count!=0 or nomask_count!=0 or inc_count!=0):
            print(mask_count, ppl, nomask_count, inc_count)
            now = datetime.now()
            # Save the current time
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
            arraycall(name)
        mask_count=0
        nomask_count=0
        inc_count=0
        name="Unknown Person"
        #if(frame_count%45==0):
        small_array=[0,0,0,0]
        big_array.append(small_array)
        small_array=[0,0,0,0]
        
    # Loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        # Unpacks the bounding box and the predictions
        #print(box)
        (startX, startY, endX, endY) = box
        #print(pred)
        (partialMask, mask, withoutMask) = pred
        # Determines the class label and the respective colour used for the box
        #label = "Mask" if mask == max(mask, withoutMask, partialMask) else "No Mask"
        #label = "Incorrectly worn" if partialMask == max(mask, withoutMask, partialMask) else "No Mask"
        if (mask > withoutMask and mask>partialMask):
            label = "Mask"
            mask_count+=1
            mask2_count+=1
        elif ( withoutMask > partialMask and withoutMask > mask):
            label = "No Mask Found, Wear a Mask"
            nomask_count+=1
            nomask2_count+=1
        else:
            label = "Wear Mask Properly"
            inc_count+=1
            inc2_count+=1
        if label == "Mask":
            color = (0, 255, 0)
        elif label=="No Mask Found, Wear a Mask":
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
            
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask, partialMask) * 100)
        name2=qrcode(frame)
        
        if (name2!="Unknown Person"):
            name=name2
        if(frame_count%45==0 and ppl==1):
            print(mask_count, ppl, nomask_count, inc_count)
            now = datetime.now()
            # Save the current time
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
            #print(dt_string)
            #name=qrcode(frame)
            arraycall(name)
            mask_count=0
            nomask_count=0
            inc_count=0
            name="Unknown Person"
    
            
        # Displays the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    

    # Shows the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    
    # Exit the program if the 'q' key is pressed
    if key == ord("q"):
        if(mask_count!=0 or nomask_count!=0 or inc_count!=0):
          arraycall(name)
        break
big_array.append(small_array)
#print("Printing Big Array....")
#print(big_array)

# Getting all the results into an array for future storage
qrarray=[]
for i in range(0,len(big_array)-1):
    if(big_array[i][1]==1 and big_array[i+1][1]==0):
        final_array.append(big_array[i])
print("Printing Final Array....")
print(final_array)

small_array=[0,0,0,"Unknown"]
final_array.append(small_array)

# QR array
for i in range(0,len(final_array)-1):
    if(final_array[i][3]=="Unknown Person"):
        qrarray.append(final_array[i])
    else:
        if(final_array[i][3]!=final_array[i+1][3]):
            qrarray.append(final_array[i])
        if(final_array[i][3]==final_array[i+1][3]):
            if(final_array[i][0]!=final_array[i+1][0]):
                qrarray.append(final_array[i])

print("\n \n \n")

print("Printing Qr Array...... \n")
for i in qrarray:
    print(i, end="\n")

# Storing the data into a dataframe and saving it as a csv file
df=pd.DataFrame(qrarray, columns=["Category", "Ppl", "Date", "Name"])
#df.head()
df.drop("Ppl", inplace=True, axis=1)
df.index.set_names("Person Number", inplace=True)
df.index+=1
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %H:%M:%S")
#df3=pd.read_csv('temp2.csv', index_col="Person Number")
#df4=pd.concat([df3,df], ignore_index=True)
#df4.index.set_names("Person Number", inplace=True)
#df4.index+=1
#df4.to_csv('temp2.csv')
df.to_csv('temp4.csv')


'''def analytics(mask2_count, nomask2_count, inc2_count):
    mask2_count = mask2_count/2
    nomask2_count = nomask2_count/2
    inc2_count = inc2_count/2
    total_count = mask2_count + nomask2_count + inc2_count
    percent_mask=((mask2_count/total_count)*100)
    percent_nomask=((nomask2_count/total_count)*100)
    percent_inc=((inc2_count/total_count)*100)
    print(percent_mask, percent_nomask, percent_inc)
    ratio_array=[percent_mask, percent_nomask, percent_inc]
    df2=pd.DataFrame(columns=["Mask %", "Without Mask %", "Incorrectly Worn %"])
    df2.loc[0]=ratio_array
    df2.to_csv('file2.csv')

analytics(mask2_count, nomask2_count, inc2_count)'''  

#Cleanup
cv2.destroyAllWindows()
vs.stop()