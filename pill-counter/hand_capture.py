import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

        lmk = results.multi_hand_landmarks[0].landmark[-1]
        if 0.5 <= lmk.x <= 0.75 and 0.65 <= lmk.y <= 0.8:
            cv2.imwrite('captured_image.jpg', image)
            print("Image captured and saved as 'captured_image.jpg'")
            break

    cv2.imshow("Output", image)
    cv2.waitKey(1)