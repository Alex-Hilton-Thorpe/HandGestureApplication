import cv2
import csv
import mediapipe as mp # Main package
from mediapipe import Image, ImageFormat # Image and format types used by the tasks API

# Tasks API modules 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def startCamera():
    return cv2.VideoCapture(0)

gesture_name = "nothing" # Change this to the name of the gesture you want to capture for the csv file for training


# Tells mediapipe where to find model file (hand_landmarker.task)
base_opts = python.BaseOptions(model_asset_path="hand_landmarker.task")

# Configures how detector behaves
# num_hands: max number of hands to detect
# *_confidence: thresholds for detection and tracking quality.
#               Higher confidence means more strict detection/tracking, 
#               which may result in fewer detections but higher accuracy.
opts = vision.HandLandmarkerOptions(
    base_options = base_opts,
    num_hands = 2,
    min_hand_detection_confidence = 0.5,
    min_hand_presence_confidence = 0.5,
    min_tracking_confidence = 0.5
)

# Creates the hand tracking object
hand_landmarker = vision.HandLandmarker.create_from_options(opts)


if __name__ == "__main__":
    
    capture = startCamera();
    with open('hand_gesture_training_data.csv', mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        while(1):
            # Capture frame-by-frame
            # ret is a boolean indicating successful frame capture
            # frame is the captured image
            ret, frame = capture.read();
        
            # Check if frame was captured successfully
            if not ret:
                print("Failed to capture frames")
                break
        
            # Convert the cv2 native BGR to RGB for mediapipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);
        
            # Process the RGB frame with mediapipe to detect hands
            mediapipe_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame);
        
            # Run hand landmark detection on this frame
            # Returns HandLandmarkerResult object
            results = hand_landmarker.detect(mediapipe_image);
        
            # If there is a hand detected, results.hand_landmarks will be a list of hand landmarks for each detected hand
            if results.hand_landmarks:
            
                h, w, _ = frame.shape; # Gets height and width of frame
            
                # Loop over each detected hand
                # results.hand_landmarks is a list of hands detected in image
                # hand_landmarks is a list of landmarks for single hand
                # landmark is a single point on the hand
                for hand_landmarks in results.hand_landmarks:
                    # Loop over each landmark in the hand
                    row = []
                    for landmark in hand_landmarks:
                        # Convert normalized landmark coordinates to pixel coordinates
                        x = int(landmark.x * w);
                        y = int(landmark.y * h);
                        row += [landmark.x, landmark.y]
                        # Draw a circle at each landmark position on the original frame
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1); # Red circles for landmarks
                        
                    row.append(gesture_name) # Add the gesture name to the end of the row for training data
                    writer.writerow(row) # Write the landmark coordinates and gesture name to the csv file for training data
                    
                 
            
                
        
            # If captured display the frames
            cv2.imshow('Hand Gesture Frame', frame);
            # Need way to shut it off - "q" key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    # When everything done, release the capture and destroy the window
    capture.release();
    cv2.destroyAllWindows();


