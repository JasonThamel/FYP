''' 
Title: Live Demonstration of the Smart Pedestrian System
Coding Language: Python 

Author: Jason Thamel
Student ID: 2057941
University of Birmingham 

Description:
This script is the main script of this project, integrating the data from the Yolov8 object detection model and the Heimdall FMCW Radar.
It allows to make predictions from the Decision Tree Algorithm and provide a decision with an allocated crossing time.
'''


import os
import cv2
import serial

import time
import numpy as np
import pandas as pd
from threading import Thread

from ultralytics import YOLO
import supervision as sv
from supervision import PolygonZone as pz
from polygone_zone import * 

import joblib
import tkinter as tk
from collections import Counter

# Load the decision-making algorithm model from the joblib file
decision_algorithm = joblib.load('v3_Decision_Tree_model_20.joblib')

ZONE_POLYGON = ALL          #Define the Zone of Interest
CONFIDENCE = 70             #Define the Confidence Treshold
RESOLUTION = [1280, 720]    #Define the resolution of the display

SHUT = 0

# Initialise the variables
timestamp = 0 
predictions = "No prediction yet"
mostCommonPred = "No Decision"
brightness = 0  # Initialise brightness
disability = 0  # Initialise disability
personCount = 0 # Initialise persson_count

i = 0 # Initialise the iteration for the timestamp

#Define the model used for the object detection
model = YOLO('customdetection4.pt')

#Connect to the USB Port the Heimdall Radar - 19200 baud - 8 bit - No parity - 1 stop bit
heimdall = serial.Serial('COM4', 19200)

#Verify if the COM is open on the serial
if heimdall.isOpen():
    print(heimdall.name +  ' is open')

#Class for the GUI simulation of the traffic light using tkinter
class TrafficLightSimulator(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Traffic Light Simulator")

        # Title of the Simulation
        title_label = tk.Label(self, text="Traffic Light Indicator", font=("Helvetica", 16))
        title_label.pack(side="top", pady=10)

        # Initialise variables for the state of the traffic light
        self.redLED = False
        self.yellowLED = False
        self.greenLED = False

        # Create canvas to display LEDs
        self.canvas = tk.Canvas(self, width=100, height=300)
        self.canvas.pack()

        # Label to display the disability message
        self.disabilityLabel = tk.Label(self, text="", font=("Helvetica", 14))
        self.disabilityLabel.pack(side="bottom", pady=10)

        # Label to display the decision of the algorithm
        self.decisionLabel = tk.Label(self, text=f"Decision: {mostCommonPred}", font=("Helvetica", 14, "bold"))
        self.decisionLabel.pack(side="bottom", pady=10)

        # Label to display the brightness intensity
        self.brightnessLabel = tk.Label(self, text=f"Brightness: {brightness}", font=("Helvetica", 14, "bold"))
        self.brightnessLabel.pack(side="bottom", pady=10)

        # Label to display the predictions of the algorithm
        self.predictionsLabel = tk.Label(self, text=f"Prediction: {predictions}", font=("Helvetica", 14, "bold"))
        self.predictionsLabel.pack(side="bottom", pady=10)

    def start_simulation(self):
        # Change the state of LEDs to simulate the traffic light
        self.greenLED = True
        self.updateTrafficLight()

    def yellowLight(self):
        self.redLED = False
        self.yellowLED = True
        self.greenLED = False
        self.updateTrafficLight()

        app.decisionLabel.config(text=f"Decision: {mostCommonPred}")

        self.after(2000, self.redLight)

    def redLight(self):
        self.redLED = True
        self.yellowLED = False
        self.greenLED = False
        self.updateTrafficLight()

        if mostCommonPred == "Camera1" or mostCommonPred == "Radar1":
            self.after(5000, self.greenLight)

        elif mostCommonPred == "Camera2":
            self.after(5750, self.greenLight)

        elif mostCommonPred == "Camera3" or mostCommonPred == "Radar2" or mostCommonPred == "Disability":
            self.after(6500, self.greenLight)

        elif mostCommonPred == "Camera4" or mostCommonPred == "Radar3":
            self.after(7500, self.greenLight)


    def greenLight(self):
        global STATE
        self.redLED = False
        self.yellowLED = False
        self.greenLED = True
        self.updateTrafficLight()
        app.decisionLabel.config(text=f"Decision: No Decision")
        STATE = 0

        # Clear the message label when switching back to green
        self.disabilityLabel.config(text="")

    def updateTrafficLight(self):
        # Update the display based on the state of LEDs
        self.canvas.delete("all")

        # Draw red LED
        self.canvas.led(30, 20, 70, 60, fill="red" if self.redLED else "gray")

        # Draw yellow LED
        self.canvas.led(30, 105, 70, 145, fill="yellow" if self.yellowLED else "gray")

        # Draw green LED
        self.canvas.led(30, 205, 70, 245, fill="green" if self.greenLED else "gray")

def GuiThread():
    global app
    print("Thread App")

    app = TrafficLightSimulator()
    app.mainloop()

def RadarThread():
    global personCount, brightness, STATE, mostCommonPred

    radarList = [0,0,0,0,0]      # List to store the last 5 values of radarCount
    personList = [0,0,0,0,0,0]  # List to store the last 5 values of personCount
    brightnessList = [0,0,0,0,0,0]    # List to store the last 5 values of brightness
    disabilityList = [0,0,0,0,0,0]    # List to store the last 5 values of disability

    # List to store the predictions
    last_mostCommonPred = None
    counter = 0
    predictionsList = []

    while True:

        if SHUT == 1:
            print(" ------------------------------------- ")
            print("----> Radar Closed !!")
            print(" ------------------------------------- ")
            break

        #Send the VOD command to the Radar
        command = b'vod\r'
        heimdall.write(command)
        # Read the message received from the Radar at a rate of 200ms
        receive = heimdall.read(12).decode('utf-8')

        # Check every byte in the message received and only append the number displayed 
        for byte in receive:
            if byte.isdigit():
                radarCount = byte


        # Get the last 5 values of radarCount, personCount, brightness and disability
        radarList = radarList[-5:] + [radarCount]
        personList = personList[-5:] + [personCount]
        brightnessList = brightnessList[-5:] + [brightness] 
        disabilityList = disabilityList[-5:] + [disability] 
        
        # Combine radarCount values, personCount, disability and brightness into a single list
        fullList =  radarList + personList + brightnessList + disabilityList
        output = [fullList]

        # Given the output list, provide a prediction from the algorithm
        predictions = decision_algorithm.predict(output)
        app.predictionsLabel.config(text=f"Prediction: {predictions}")

        # Save predictions to the list
        predictionsList.append(predictions[0])  

        # Check if the predictions list is full
        if len(predictionsList) == 5:

            #Take the most common prediction from the list
            mostCommonPred = Counter(predictionsList).most_common(1)[0][0]

            #This if statement is to make sure the predictions are the same if they have the same crossing time
            if mostCommonPred == "Radar1":
                mostCommonPred = "Camera1"
            
            elif mostCommonPred == "Radar2":
                mostCommonPred = "Camera3"
            
            elif mostCommonPred == "Radar3":
                mostCommonPred = "Camera4"

            #This section is making sure that the most common prediction is the same for 10 times in a row
            #It is adjustable to provide more or less certainty 
            elif mostCommonPred == last_mostCommonPred:
                counter += 1

                if counter == 10 and mostCommonPred != "Ignore":
                    #Start the traffic light simulation
                    app.yellowLight()

                    if mostCommonPred == "Camera1":
                        time.sleep(7) #This sleep time value is for demonstration
                    
                    if mostCommonPred == "Camera2":
                        time.sleep(7.75) #This sleep time value is for demonstration
                    
                    if mostCommonPred == "Camera3" or mostCommonPred == "Disability":
                        time.sleep(8.5) #This sleep time value is for demonstration

                    if mostCommonPred == "Camera4":
                        time.sleep(9.5) #This sleep time value is for demonstration

                    #STATE 1 means the Traffic LIght Thread cannot rerun again until the sequence is finished 
                    STATE = 1

                    # Reset the counter
                    counter = 0
                    last_mostCommonPred = None
            else:
                # Reset the counter and update the last most common prediction
                counter = 1
                last_mostCommonPred = mostCommonPred

            # Empty the predictions list for the next 10 values
            predictionsList = []

    
xgui = Thread(target=GuiThread)
xradar = Thread(target=RadarThread)

def main():
    # Initialize the global variables
    global personCount, timestamp, SHUT, brightness, disability

    # Capture the Webcam feed
    cap = cv2.VideoCapture(0)

    #Get the fps of the Webcam feed
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #Bounding boxes annotation using supervision 
    boxAnnotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=0.5
    )
    #Zone annotation using supervision
    zone = sv.PolygonZone(polygon=ZONE_POLYGON, frame_resolution_wh=tuple([1280, 720]))

    zoneAnnotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    #Start the Traffic Light Thread
    xgui.start()
    time.sleep(1)

    #Start the Radar Thread
    xradar.start()
    time.sleep(0.5)

    #Initiate the Traffic simulation 
    app.start_simulation()

    while True:
        #Read the frames of the Camera Feed 
        ret, frame = cap.read()

        if not ret:
            SHUT = 1
            break
        
        #Convert the frame to grayscale and provide a value of the avergae brightness of the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        brightness = round(np.mean(gray),0)
        app.brightnessLabel.config(text=f"Brightness: {brightness}")

        #Run the object detection of the frame using YOLOv8 and Supervision
        result = model(frame, agnostic_nms=True)[0]

        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence >= CONFIDENCE/100] # Set the confidence threshold 

        #Filter the detections to only include the person and umbrella and have the disability detection separate
        person = detections[(detections.class_id == 0)|(detections.class_id == 1)] #person + umbrella
        disability = detections[(detections.class_id == 2)] #disability

        #If a person with disability is detected its variable to set to 1, to be used in the decision algorithm
        if any(detections.class_id == 2):
            disability = 1
            app.disabilityLabel.config(text="Disability", fg="red")
        else:
            disability = 0
            app.disabilityLabel.config(text="", fg="black")

        #This section is used to display the bounding boxes on the detected objects and the valuable information 
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for (xmin, ymin, xmax, ymax), confidence, class_id in zip(person.xyxy, person.confidence, person.class_id)
        ]

        frame = boxAnnotator.annotate(
            scene=frame,
            detections=person,
            labels=labels
        )

        #The personCount variable is used to count the number of people within the zone defined by the polygone
        personCount = int(pz.polygonCount)

        zone.trigger(detections=person)
        frame = zoneAnnotator.annotate(scene=frame)

        #Get the timestamp that the frame was captured at
        timestamp = round(i*1/fps, 1)
        i += 1

        cv2.imshow("YOLOv8", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
