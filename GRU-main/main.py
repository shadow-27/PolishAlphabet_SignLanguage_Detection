import tkinter as tk
import cv2
from PIL import Image, ImageTk
import keyboard
import enchant

import csv
import copy
from collections import Counter
from collections import deque

import cv2 as cv
import mediapipe as mp

from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier
from app import *

# Configure
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 498
LEFT_PANEL_WIDTH = 648
RIGHT_PANEL_WIDTH = 452
WEBCAM_DIM = 800
SCALE_FACTOR = 1.5
CONTRAST_COLOR = "#104cf4"
BACKGROUND_COLOR = "#b1c5fc"
FONT_COLOR = "black"
HEADING_FONT = ('Courier New', 36)
BODY_FONT = ('Courier New', 14)
BUTTON_FONT = ('Courier New', 10)
PLACEHOLDER_WIDTH = 27
HEIGHT_GAP = 2

cancel = False
use_brect = True
mode = 0
d = enchant.Dict("en_US")

# Model load
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

keypoint_classifier = KeyPointClassifier()

point_history_classifier = PointHistoryClassifier()

# Read labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]
with open(
        'model/point_history_classifier/point_history_classifier_label.csv',
        encoding='utf-8-sig') as f:
    point_history_classifier_labels = csv.reader(f)
    point_history_classifier_labels = [
        row[0] for row in point_history_classifier_labels
    ]

# FPS Measurement
cvFpsCalc = CvFpsCalc(buffer_len=10)

# Coordinate history
history_length = 16
point_history = deque(maxlen=history_length)

# Finger gesture history
finger_gesture_history = deque(maxlen=history_length)

root = tk.Tk()
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
root.resizable(width=False, height=False)
root.bind('<Escape>', lambda e: root.quit())
root.title("Polish Sign Language Detection App")
root.iconbitmap("images/icon.ico")

# Create panedWindow
mainPanel = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg=BACKGROUND_COLOR, borderwidth=5)
mainPanel.pack(fill=tk.BOTH, expand=True)

# Left Panel
leftPanel = tk.Label(mainPanel, anchor=tk.N)

mainPanel.paneconfigure(leftPanel, minsize=LEFT_PANEL_WIDTH)
mainPanel.add(leftPanel)

# Webcam video display
label = tk.Label(leftPanel, anchor=tk.CENTER, bg=CONTRAST_COLOR)
label.grid(row=0, column=0)
cap = cv2.VideoCapture(0)

# Right Panel
rightPanel = tk.PanedWindow(mainPanel, orient=tk.VERTICAL)
mainPanel.paneconfigure(rightPanel, minsize=RIGHT_PANEL_WIDTH)
mainPanel.add(rightPanel)

rightPanelTopFrame = tk.Frame(rightPanel, bg=BACKGROUND_COLOR)
tk.Label(rightPanelTopFrame, text="OUTPUT", anchor=tk.N, bg=BACKGROUND_COLOR, fg=FONT_COLOR, font=HEADING_FONT).pack()
tk.Label(rightPanelTopFrame, text="", anchor=tk.N, bg=BACKGROUND_COLOR, height=HEIGHT_GAP).pack()
# tk.Label(rightPanelTopFrame, text="RECOGNIZED LETTER", anchor=tk.N, bg=BACKGROUND_COLOR, fg=FONT_COLOR,
#          font=BODY_FONT).pack()
# letterText = tk.Text(rightPanelTopFrame, state=tk.DISABLED, relief=tk.RAISED, fg=FONT_COLOR,
#                      font=BODY_FONT, width=PLACEHOLDER_WIDTH, height=1)
# letterText.pack()
rightPanel.paneconfigure(rightPanelTopFrame, minsize=50)
rightPanel.add(rightPanelTopFrame)

rightPanelBottomFrame = tk.Frame(rightPanel, bg=BACKGROUND_COLOR)
tk.Label(rightPanelBottomFrame, text="", bg=BACKGROUND_COLOR, height=HEIGHT_GAP).pack()
tk.Label(rightPanelBottomFrame, text="WORDS FORMED", bg=BACKGROUND_COLOR, fg=FONT_COLOR,
         font=BODY_FONT).pack()
wordText = tk.Text(rightPanelBottomFrame, state=tk.DISABLED, relief=tk.RAISED, fg=FONT_COLOR,
                   font=BODY_FONT, width=PLACEHOLDER_WIDTH, height=6)
wordText.pack()
tk.Label(rightPanelBottomFrame, text="", bg=BACKGROUND_COLOR, height=HEIGHT_GAP).pack()
tk.Label(rightPanelBottomFrame, text="SUGGESTED WORDS", bg=BACKGROUND_COLOR, fg=FONT_COLOR,
         font=BODY_FONT).pack()
suggestedWordText = tk.Text(rightPanelBottomFrame, state=tk.DISABLED, relief=tk.RAISED, fg=FONT_COLOR,
                            font=BODY_FONT, width=PLACEHOLDER_WIDTH, height=5)
suggestedWordText.pack()
rightPanel.paneconfigure(rightPanelBottomFrame, minsize=448)
rightPanel.add(rightPanelBottomFrame)


# Define function to show frame
def show_frames():
    global mode

    recognized_letter = ""
    word_formed = ""
    letter_approved = False

    fps = cvFpsCalc.get()

    # Process Key (ESC: end)
    key = cv2.waitKey(10)
    if key == 32:
        letter_approved = True
    # else:
    #     print(key)
    number, mode = select_mode(key, mode)

    # Get the latest frame and convert into Image
    ret, cv2image = cap.read()[0], cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    if not ret:
        return

    image = cv2.flip(cv2image, 1)
    debug_image = copy.deepcopy(image)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    # Hand landmarks
    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)
            pre_processed_point_history_list = pre_process_point_history(
                debug_image, point_history)

            # Write to the dataset file
            logging_csv(number, mode, pre_processed_landmark_list,
                        pre_processed_point_history_list)

            # Hand sign classification
            pre_processed_landmark_list = np.expand_dims(pre_processed_landmark_list, axis=0)
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            if hand_sign_id == "Not applicable":  # Point gesture
                point_history.append(landmark_list[8])
            else:
                point_history.append([0, 0])

            # Finger gesture classification
            finger_gesture_id = 0
            point_history_len = len(pre_processed_point_history_list)
            if point_history_len == (history_length * 2):
                finger_gesture_id = point_history_classifier(
                    pre_processed_point_history_list)

            # Calculates the gesture IDs in the latest detection
            finger_gesture_history.append(finger_gesture_id)
            most_common_fg_id = Counter(
                finger_gesture_history).most_common()

            # Drawing part
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            debug_image = draw_landmarks(debug_image, landmark_list)
            debug_image, recognized_letter = draw_info_text(
                debug_image,
                brect,
                handedness,
                keypoint_classifier_labels[hand_sign_id],
                point_history_classifier_labels[most_common_fg_id[0][0]],
            )
    else:
        point_history.append([0, 0])

    debug_image = draw_point_history(debug_image, point_history)
    debug_image = draw_info(debug_image, fps, mode, number)
    debug_image = Image.fromarray(debug_image)

    # Convert image to PhotoImage
    photoImage = ImageTk.PhotoImage(image=debug_image)
    label.photoImage = photoImage
    label.configure(image=photoImage)

    word_formed += recognized_letter

    if word_formed:
        d.check(recognized_letter[-1])
        print(d.suggest(recognized_letter[-1]))

    # if recognized_letter:
    #     # Update letter box
    #     print(recognized_letter)
    #     letterText.configure(state='normal')
    #     letterText.tag_configure("center", justify='center')
    #     letterText.delete('1.0', tk.END)
    #     letterText.insert('1.0', recognized_letter)
    #     letterText.tag_add("center", 1.0, "end")
    #     letterText.configure(state='disabled')
    #
    #     # Update word box
    #     if letter_approved:
    #         word_formed = recognized_letter[- 1] + word_formed
    #         wordText.configure(state='normal')
    #         letterText.delete('1.0', tk.END)
    #         wordText.insert('1.0', word_formed)
    #         wordText.tag_add(1.0, "end")
    #         wordText.configure(state='disabled')
    #         letter_approved = False

    # Repeat after an interval to capture continuously
    label.after(100, show_frames)


show_frames()
root.mainloop()
