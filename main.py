import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import mediapipe as mp


def overlay_image(frame, img, position, size):
    x, y = position
    img_resized = cv2.resize(img, size)

    h, w, _ = frame.shape
    img_h, img_w, _ = img_resized.shape

    if x + img_w > w:
        img_w = w - x
        img_resized = img_resized[:, :img_w]
    if y + img_h > h:
        img_h = h - y
        img_resized = img_resized[:img_h, :]

    alpha_s = img_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        frame[y : y + img_h, x : x + img_w, c] = (
            alpha_s * img_resized[:, :, c]
            + alpha_l * frame[y : y + img_h, x : x + img_w, c]
        )
    return frame


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

animal_img = cv2.imread("animal.png", cv2.IMREAD_UNCHANGED)


while True:
    success, frame = cap.read()
    if success:
        RGB_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )  # cv2 default is BGR and mediapipe is RGB, therefore required conversion
        result = hand.process(RGB_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # palm landmarks mediapipe
                palm_landmarks = [
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                ]

                # bouding box
                x_min = min(lm.x for lm in palm_landmarks)
                x_max = max(lm.x for lm in palm_landmarks)
                y_min = min(lm.y for lm in palm_landmarks)
                y_max = max(lm.y for lm in palm_landmarks)

                h, w, _ = frame.shape
                x_min = int(x_min * w)
                x_max = int(x_max * w)
                y_min = int(y_min * h)
                y_max = int(y_max * h)

                position = (x_min, y_min)
                size = (x_max - x_min, y_max - y_min)

                # overlay the animal image onto the frame
                frame = overlay_image(frame, animal_img, position, size)

        cv2.imshow("capture image", frame)
        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()
