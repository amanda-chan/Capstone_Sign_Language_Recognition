import cv2
import mediapipe as mp
import torch

# initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

# function to extract hand landmarks
def extract_hand_landmarks(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            return torch.tensor([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype = torch.float32).flatten()
    return torch.zeros(21 * 3) # if no hand detected, return a zero vector

# open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process frame
    results = hands.process(frame_rgb)

    # draw landmarks if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # show frame
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()