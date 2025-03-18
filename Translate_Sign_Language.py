import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import h5py
from collections import deque

# disable onednn optimizations to avoid potential numerical issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# load the trained model
model_path = "Data/model_best.keras"
model = load_model(model_path)

# load class labels from dataset
h5_file = "Data/test_data.h5"
with h5py.File(h5_file, 'r') as hf:
    all_labels = [hf[k].attrs['label'].decode() if isinstance(hf[k].attrs['label'], bytes) else hf[k].attrs['label'] for k in hf.keys()]
    class_names = sorted(set(all_labels))  # get unique class names
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    index_to_class = {idx: name for name, idx in class_to_index.items()}  # reverse mapping

# initialize mediapipe hands and pose module
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands_model = mp_hands.Hands(min_detection_confidence = 0.3, min_tracking_confidence = 0.3)
pose_model = mp_pose.Pose(min_detection_confidence = 0.3, min_tracking_confidence = 0.3)

mp_drawing = mp.solutions.drawing_utils


MAX_FRAMES = 50
captured_landmarks = []
word_predictions = []
predicted_word = ""

# function to preprocess the frame (like how the gif is preprocessed)
def preprocess_frame(frame):
    # convert the frame from bgr to rgb (for mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands_model.process(rgb_frame)
    pose_results = pose_model.process(rgb_frame)

    # initialize arrays for pose and hand landmarks
    pose = np.zeros(33 * 4)  # 33 pose landmarks * 4 values (x, y, z, visibility)
    lh = np.zeros(21 * 3)    # 21 left hand landmarks * 3 values (x, y, z)
    rh = np.zeros(21 * 3)    # 21 right hand landmarks * 3 values (x, y, z)
    hand_rotation = np.zeros(2)  # left and right hand rotation
    hand_position = np.zeros(6)  # left and right hand relative to body

    # extract body landmarks if available
    if pose_results.pose_landmarks:
        pose_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_results.pose_landmarks.landmark])
        pose = pose_data.flatten()

        # extract key body landmarks
        left_wrist = pose_data[15, :3]  # (x, y, z)
        right_wrist = pose_data[16, :3]  # (x, y, z)
        left_shoulder = pose_data[11, :3]  # (x, y, z)
        right_shoulder = pose_data[12, :3]  # (x, y, z)
        nose = pose_data[0, :3]  # (x, y, z)
    else:
        left_wrist = np.zeros(3)
        right_wrist = np.zeros(3)
        left_shoulder = np.zeros(3)
        right_shoulder = np.zeros(3)
        nose = np.zeros(3)

    # process the detected hand landmarks
    if hand_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            label = hand_results.multi_handedness[i].classification[0].label

            # extract 3d landmarks (x, y, z)
            hand_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            if label == "Left":
                lh = hand_data.reshape(21, 3) - left_wrist  # normalize relative to left wrist
                lh = lh.flatten()
            elif label == "Right":
                rh = hand_data.reshape(21, 3) - right_wrist  # normalize relative to right wrist
                rh = rh.flatten()
    else: # return None if no hands are detected
        return None

    # calculate hand rotation based on wrist and index finger tip (palm facing direction)
    def calculate_hand_rotation(wrist, index_finger_tip):
        if np.any(wrist) and np.any(index_finger_tip):
            hand_vector = index_finger_tip - wrist
            z_axis = np.array([0, 0, 1])  # reference for forward direction
            angle = np.arccos(np.dot(hand_vector, z_axis) / (np.linalg.norm(hand_vector) * np.linalg.norm(z_axis)))
            return angle
        return 0

    if lh.any():
        hand_rotation[0] = calculate_hand_rotation(left_wrist, lh[8:11])  # index finger tip

    if rh.any():
        hand_rotation[1] = calculate_hand_rotation(right_wrist, rh[8:11])  # index finger tip

    # calculate hand position relative to body (chest & head) (in this case, nose and shoulder)
    def calculate_hand_position(wrist, nose, shoulder):
        if np.any(wrist) and np.any(nose) and np.any(shoulder):
            return np.array([
                np.linalg.norm(wrist - nose),  # distance to nose
                np.linalg.norm(wrist - shoulder)  # distance to shoulder
            ])
        return np.zeros(2)

    hand_position[:2] = calculate_hand_position(left_wrist, nose, left_shoulder)
    hand_position[2:4] = calculate_hand_position(right_wrist, nose, right_shoulder)

    # two-hand sign alignment (relative wrist distance)
    if np.any(left_wrist) and np.any(right_wrist):
        hand_position[4:] = np.linalg.norm(left_wrist - right_wrist)  # distance between wrists

    # combine pose, hand landmarks, rotation, and positioning
    frame_landmarks = np.concatenate((pose, lh, rh, hand_rotation, hand_position))

    return frame_landmarks


def predict_sign():

    global predicted_word

    # ensure thread safety
    with threading.Lock():
        if len(captured_landmarks) < MAX_FRAMES:
            return  # ensure enough frames are collected before prediction

        # convert landmarks to numpy arrays
        landmarks = np.array(captured_landmarks)

        # reshape input data to match model's expected shape
        input_data = landmarks.reshape(1, MAX_FRAMES, -1)

        # make prediction
        predictions = model.predict(input_data)
        predicted_class_idx = np.argmax(predictions)
        predicted_word = index_to_class[predicted_class_idx]
        confidence = np.max(predictions)

        # ignore predictions with low confidence
        if confidence >= 0.5:  # set a confidence threshold
            predicted_word = index_to_class[predicted_class_idx]

        # print top predictions for debugging
        top_predictions = np.argsort(predictions[0])[::-1][:3]  # top 3 predictions
        for i, idx in enumerate(top_predictions):
            print(f"{index_to_class[idx]}: {predictions[0][idx]:.2f}")


# set up the webcam capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 30)

# verify if the camera can be connected and opened
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

    # process the frame to extract landmarks
    frame_landmarks = preprocess_frame(frame)

    if frame_landmarks is not None: # if hands were detected

        # convert the frame from bgr to rgb (for mediapipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_model.process(rgb_frame)

        # draw the landmarks on the frame
        if results.multi_hand_landmarks: 
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # add the landmarks for this frame to the captured landmarks
        captured_landmarks.append(frame_landmarks)

    # if hands were detected previously but not anymore and less than max frames, pad with zeros
    elif len(captured_landmarks) > 0 and len(captured_landmarks) < MAX_FRAMES:
        captured_landmarks.append(np.zeros(33 * 4 + 21 * 3 * 2 + 2 + 6))  # pose + hands + rotation + positioning # pad with zeros

    # ensure we maintain a fixed number of frames (50) - model gets a consistent input of 50 frames for prediction
    if len(captured_landmarks) > MAX_FRAMES:
        captured_landmarks.popleft()  # remove oldest frame to maintain sliding window

    # if we have captured at least 50 frames, make a prediction
    if len(captured_landmarks) == MAX_FRAMES:
        
        threading.Thread(target = predict_sign, daemon = True).start()

        # clear predicted word and capture landmarks for next sequence
        predicted_word = ""
        captured_landmarks = []

    # display prediction on frame
    cv2.putText(frame, f"Prediction: {predicted_word}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # display the frame with hand landmarks and predictions (if any)
    cv2.imshow("Sign Language Prediction", frame)

    # press 'q' to quit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
