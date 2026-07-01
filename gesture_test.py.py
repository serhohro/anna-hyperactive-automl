import cv2
import mediapipe as mp
import pyautogui
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: камера не открывается")
    exit()

print("Камера работает. Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            index = hand.landmark[8]
            thumb = hand.landmark[4]
            h, w, _ = frame.shape
            ix, iy = int(index.x * w), int(index.y * h)
            tx, ty = int(thumb.x * w), int(thumb.y * h)
            cv2.circle(frame, (ix, iy), 10, (0,255,0), -1)
            cv2.circle(frame, (tx, ty), 10, (255,0,0), -1)

            dist = math.hypot(index.x - thumb.x, index.y - thumb.y)
            if dist < 0.05:
                cv2.putText(frame, "CLICK", (ix+10, iy+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                # раскомментируйте следующую строку для реального клика
                # pyautogui.click()

            screen_w, screen_h = pyautogui.size()
            pyautogui.moveTo(index.x * screen_w, index.y * screen_h)

    cv2.imshow("Gesture Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()