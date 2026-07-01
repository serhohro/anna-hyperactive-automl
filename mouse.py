import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
import json
import os
from collections import deque

# ------------------- Настройки -------------------
SETTINGS_FILE = "mouse_settings.json"

def load_settings():
    default = {"sensitivity": 1.3, "smoothing": 6, "click_distance": 0.12, "margin": 45}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                default.update(saved)
        except:
            pass
    return default

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

settings = load_settings()

# ------------------- Камера -------------------
CAMERA_INDEX = 0  # попробуйте 0, если 1 не работает
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"❌ Камера {CAMERA_INDEX} не открылась. Попробуйте 0, 1 или 2.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0
coord_queue = deque(maxlen=4)

# ------------------- Окна -------------------
settings_win = "Mouse Settings"
video_win = "Hand Mouse - Video"

cv2.namedWindow(settings_win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(settings_win, 400, 200)
cv2.moveWindow(settings_win, 50, 50)

cv2.namedWindow(video_win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(video_win, 640, 480)
cv2.moveWindow(video_win, 500, 100)

# Создаём трекбары
cv2.createTrackbar("Sensitivity", settings_win, int(settings["sensitivity"] * 10), 30, lambda x: None)
cv2.createTrackbar("Smoothing", settings_win, settings["smoothing"], 15, lambda x: None)
cv2.createTrackbar("Click dist", settings_win, int(settings["click_distance"] * 100), 25, lambda x: None)
cv2.createTrackbar("Margin", settings_win, settings["margin"], 100, lambda x: None)

print("✅ Управление жестами запущено.")
print("▶ Движение: указательный палец")
print("▶ Клик: сведение большого и указательного")
print("▶ Выход: нажмите 'q' в окне видео")

# ------------------- Основной цикл -------------------
try:
    while True:
        # Проверяем, открыты ли окна
        if cv2.getWindowProperty(video_win, cv2.WND_PROP_VISIBLE) < 1:
            print("Окно видео закрыто, выходим...")
            break
        
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр")
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Получаем значения с ползунков
        sensitivity = cv2.getTrackbarPos("Sensitivity", settings_win) / 10.0
        smoothing = cv2.getTrackbarPos("Smoothing", settings_win)
        smoothing = max(1, smoothing)
        click_dist = cv2.getTrackbarPos("Click dist", settings_win) / 100.0
        margin = cv2.getTrackbarPos("Margin", settings_win)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            index = hand.landmark[8]
            thumb = hand.landmark[4]
            ix = int(index.x * w)
            iy = int(index.y * h)
            tx = int(thumb.x * w)
            ty = int(thumb.y * h)

            # Визуализация
            cv2.circle(frame, (ix, iy), 12, (0, 255, 0), -1)
            cv2.circle(frame, (ix, iy), 12, (0, 255, 0), 2)
            cv2.circle(frame, (tx, ty), 8, (255, 0, 0), -1)

            dist = math.hypot(index.x - thumb.x, index.y - thumb.y)
            cv2.putText(frame, f"dist = {dist:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            coord_queue.append((ix, iy))
            avg_ix = int(np.mean([p[0] for p in coord_queue]))
            avg_iy = int(np.mean([p[1] for p in coord_queue]))

            mouse_x = np.interp(avg_ix, (margin, w - margin), (0, screen_w))
            mouse_y = np.interp(avg_iy, (margin, h - margin), (0, screen_h))
            cur_x = prev_x + (mouse_x - prev_x) / smoothing
            cur_y = prev_y + (mouse_y - prev_y) / smoothing
            cur_x = prev_x + (cur_x - prev_x) * sensitivity
            cur_y = prev_y + (cur_y - prev_y) * sensitivity
            cur_x = max(0, min(cur_x, screen_w))
            cur_y = max(0, min(cur_y, screen_h))
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

            # Клик
            if dist < click_dist:
                cv2.circle(frame, (ix, iy), 20, (0, 0, 255), -1)
                cv2.putText(frame, "CLICK!", (ix+15, iy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                pyautogui.click()
                time.sleep(0.15)

        # Отображаем параметры
        cv2.putText(frame, f"Sensitivity:{sensitivity:.1f}  Smoothing:{smoothing}  Click:{click_dist:.2f}  Margin:{margin}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'q' to quit", (w-150, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(video_win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    # Сохраняем настройки
    settings = {"sensitivity": sensitivity, "smoothing": smoothing, "click_distance": click_dist, "margin": margin}
    save_settings(settings)
    print("Настройки сохранены.")
    cap.release()
    cv2.destroyAllWindows()