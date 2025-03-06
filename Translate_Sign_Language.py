import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# disable onednn optimizations to avoid potential numerical issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# load the trained model
model_path = "Data/model_best.keras"
model = load_model(model_path)

# initialize mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence = 0.7, min_tracking_confidence = 0.7)
mp_drawing = mp.solutions.drawing_utils

MAX_FRAMES = 40
captured_frames = []
word_predictions = []

# function to preprocess the frame (like how the gif is preprocessed)
def preprocess_frame(frame):
    landmarks = []  # to store the hand landmarks for each frame

    # convert the frame from bgr to rgb (for mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # initialize landmarks for this frame (42 values for 21 landmarks * 2 coords)
    frame_landmarks = np.zeros(42)  # initialize as zeros to pad in case of missing landmarks

    # process the detected hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # extract (x, y) coordinates for each landmark
            landmarks_for_hand = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks.landmark])
            frame_landmarks[:len(landmarks_for_hand.flatten())] = landmarks_for_hand.flatten()

    landmarks.append(frame_landmarks)


# set up the webcam capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Video stream successfully opened.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # process the frame to extract hand landmarks
    processed_frame_landmarks = preprocess_frame(frame, MAX_FRAMES)

    # add the landmarks for this frame to the captured frames
    captured_frames.append(processed_frame_landmarks)

    # if we have captured at least 40 frames and its not empty, make a prediction
    if len(captured_frames) == MAX_FRAMES and captured_frames:
        
        # convert captured frames into the same format as training data
        input_data = np.array(captured_frames, dtype=np.float32)

        # flatten the data
        input_data = input_data.flatten()

    # display the frame with hand landmarks and predictions (if any)
    cv2.imshow("Sign Language Prediction", frame)

    # press 'q' to quit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
