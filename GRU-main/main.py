import csv
import copy
from collections import Counter
from collections import deque
import keyboard
import time
from app import *
from codeUtils import *
import mediapipe as mp

import tkinter as tk
from PIL import Image, ImageTk
import os
import time
import cv2
import math
from app import *
from codeUtils import *
from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifierOneFrame
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier

from hunspell import Hunspell
h = Hunspell('pl_PL', hunspell_data_dir=os.getcwd() + '/dictionaries')


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
SUGGESTED_WORD_NUMBER = 5

cancel = False
use_brect = True
mode = 0
button_dict = {}

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

keypoint_classifier_one_frame = KeyPointClassifierOneFrame()

all_letters = ['a',
               'ą',
               'b',
               'c',
               'ć',
               'ch',
               'cz',
               'd',
               'e',
               'ę',
               'f',
               'g',
               'h',
               'i',
               'j',
               'k',
               'l',
               'ł',
               'm',
               'n',
               'ń',
               'o',
               'ó',
               'p',
               'r',
               'rz',
               's',
               'ś',
               'sz',
               't',
               'u',
               'v',
               'w',
               'x',
               'y',
               'z',
               'ź',
               'ż',
               'non']

boundry = 0.6

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


# word callback
def wordCallback(buttonText, wordText, sentence):
    clearCallback(wordText, sentence)
    sentence.configure(state='normal')
    sentence.insert(tk.END, buttonText)
    sentence.configure(state='disabled')


def backspaceCallback(wordText, sentence):

    readSentence = sentence.get("1.0", 'end-1c')
    sentenceStr = readSentence[:len(readSentence)-1]
    sentence.configure(state='normal')
    sentence.delete('1.0', tk.END)
    sentence.insert('1.0', sentenceStr)
    sentence.configure(state='disabled')

    wordText.configure(state='normal')
    wordText.delete('end-2c', tk.END)
    wordText.configure(state='disabled')


# clear callback
def clearCallback(wordText, sentence):
    wordText.configure(state='normal')
    wordText.delete('1.0', tk.END)
    wordText.configure(state='disabled')

    readSentence = sentence.get("1.0", 'end-1c')
    index = readSentence.rfind(" ")
    if index != -1:
        sentenceStr = readSentence[:index + 1]
        sentence.configure(state='normal')
        sentence.delete('1.0', tk.END)
        sentence.insert('1.0', sentenceStr)
        sentence.configure(state='disabled')
    else:
        sentence.configure(state='normal')
        sentence.delete('1.0', tk.END)
        sentence.configure(state='disabled')


# space callback
def spaceCallback(wordText, sentence):
    sentence.configure(state='normal')
    sentence.insert(tk.END, " ")
    sentence.configure(state='disabled')

    wordText.configure(state='normal')
    wordText.delete('1.0', tk.END)
    wordText.configure(state='disabled')


def predict_values_right(obj):
    count = 0
    if len(obj.coordinates_with_movement_right) != 0:
        indexes = define_indexing(len(obj.coordinates_with_movement_right[0]))
    else:
        indexes = None
    relative_to_write = []
    emptyArray = np.empty(shape=(38,))
    emptyList = []
    result = 38
    if indexes is not None:
        for id in indexes:
            new_list = [obj.coordinates_with_movement_right[0][k] for k in id]
            relative_to_write = fill_coordinates(new_list)

            to_predict = copy.deepcopy(obj.historyCoordinates_right[0])
            to_predict.append(relative_to_write[0][0])
            to_predict.append(relative_to_write[1][0])
            to_predict.append(relative_to_write[0][1])
            to_predict.append(relative_to_write[1][1])
            to_predict.append(relative_to_write[0][2])
            to_predict.append(relative_to_write[1][2])
            to_predict.append(relative_to_write[0][3])
            to_predict.append(relative_to_write[1][3])
            to_predict.append(relative_to_write[0][4])
            to_predict.append(relative_to_write[1][4])
            to_predict.append(relative_to_write[0][5])
            to_predict.append(relative_to_write[1][5])
            to_predict.append(relative_to_write[0][6])
            to_predict.append(relative_to_write[1][6])

            logging_csv(0, mode, to_predict,
                        to_predict)
            pre_processed_landmark_list = np.expand_dims(to_predict, axis=0)
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            emptyList.append(hand_sign_id.tolist())
            to_predict = to_predict[:len(to_predict) - 14]
    else:
        return result

    tmp_List = []
    for i in emptyList:
        maxArg = np.argmax(np.array(i))
        tmp_List.append([i[maxArg], maxArg])

    helper = [item[0] for item in tmp_List]
    maxArg = np.argmax(np.array(helper))

    result = tmp_List[maxArg][1]

    result_value = tmp_List[maxArg][0]
    if result_value < 0.8:
        obj.number_of_frames_right = obj.number_of_frames_right - 1
        del obj.historyCoordinates_right[0]
        del obj.beginning_coordinates_right[0]
        del obj.distance_right[0]
        del obj.coordinates_with_movement_right[0]
        result = 38
    else:
        obj.number_of_frames_right = 1
        obj.historyCoordinates_right = []
        obj.beginning_coordinates_right = []
        obj.distance_right = []
        obj.coordinates_with_movement_right = []
        obj.timer_right_hand = time.time()
        obj.ready_to_start_right = False
    return result


def predict_values_left(obj):
    count = 0
    if len(obj.coordinates_with_movement_left) != 0:
        indexes = define_indexing(len(obj.coordinates_with_movement_left[0]))
    else:
        indexes = None
    relative_to_write = []
    emptyArray = np.empty(shape=(38,))
    emptyList = []
    result = 38
    if indexes is not None:
        for id in indexes:
            new_list = [obj.coordinates_with_movement_left[0][k] for k in id]
            relative_to_write = fill_coordinates(new_list)

            to_predict = copy.deepcopy(obj.historyCoordinates_left[0])
            to_predict.append(relative_to_write[0][0])
            to_predict.append(relative_to_write[1][0])
            to_predict.append(relative_to_write[0][1])
            to_predict.append(relative_to_write[1][1])
            to_predict.append(relative_to_write[0][2])
            to_predict.append(relative_to_write[1][2])
            to_predict.append(relative_to_write[0][3])
            to_predict.append(relative_to_write[1][3])
            to_predict.append(relative_to_write[0][4])
            to_predict.append(relative_to_write[1][4])
            to_predict.append(relative_to_write[0][5])
            to_predict.append(relative_to_write[1][5])
            to_predict.append(relative_to_write[0][6])
            to_predict.append(relative_to_write[1][6])

            logging_csv(0, mode, to_predict,
                        to_predict)
            pre_processed_landmark_list = np.expand_dims(to_predict, axis=0)
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            emptyList.append(hand_sign_id.tolist())
            to_predict = to_predict[:len(to_predict) - 14]
    else:
        return result

    tmp_List = []
    for i in emptyList:
        maxArg = np.argmax(np.array(i))
        tmp_List.append([i[maxArg], maxArg])

    helper = [item[0] for item in tmp_List]
    maxArg = np.argmax(np.array(helper))

    result = tmp_List[maxArg][1]

    result_value = tmp_List[maxArg][0]
    if result_value < 0.8:
        obj.number_of_frames_left = obj.number_of_frames_left - 1
        del obj.historyCoordinates_left[0]
        del obj.beginning_coordinates_left[0]
        del obj.distance_left[0]
        del obj.coordinates_with_movement_left[0]
        result = 38
    else:
        obj.number_of_frames_left = 1
        obj.historyCoordinates_left = []
        obj.beginning_coordinates_left = []
        obj.distance_left = []
        obj.coordinates_with_movement_left = []
        obj.timer_left_hand = time.time()
        obj.ready_to_start_left = False
    return result


def predict_values_one_frame(coords_list):
    pre_processed_landmark_list = np.expand_dims(coords_list, axis=0)
    hand_sign_id = keypoint_classifier_one_frame(pre_processed_landmark_list)
    maxArg = np.argmax(hand_sign_id)
    # print(maxArg)
    return hand_sign_id[maxArg]


def check_height(height):
    if height > (450 * boundry):
        return False
    else:
        return True


# telling which hands are on the picture
def which_hands(results):
    how_many_hands = []
    if len(results) == 2:
        how_many_hands.append("Left")
        how_many_hands.append("Right")
    if len(results) == 1:
        how_many_hands.append(results[0].classification[0].label)
    return how_many_hands


# create window
def create_window():
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
    tk.Label(rightPanelTopFrame, text="OUTPUT", anchor=tk.N, bg=BACKGROUND_COLOR, fg=FONT_COLOR,
             font=HEADING_FONT).pack()
    tk.Label(rightPanelTopFrame, text="", anchor=tk.N, bg=BACKGROUND_COLOR, height=HEIGHT_GAP).pack()
    rightPanel.paneconfigure(rightPanelTopFrame, minsize=50)
    rightPanel.add(rightPanelTopFrame)

    rightPanelBottomFrame = tk.Frame(rightPanel, bg=BACKGROUND_COLOR)
    tk.Label(rightPanelBottomFrame, text="", bg=BACKGROUND_COLOR, height=HEIGHT_GAP).pack()
    tk.Label(rightPanelBottomFrame, text="WORDS FORMED", bg=BACKGROUND_COLOR, fg=FONT_COLOR,
             font=BODY_FONT).pack()
    wordText = tk.Text(rightPanelBottomFrame, state=tk.DISABLED, relief=tk.RAISED, fg=FONT_COLOR,
                       font=BODY_FONT, width=PLACEHOLDER_WIDTH, height=2)
    wordText.pack()
    clearBtn = tk.Button(rightPanelBottomFrame, text="CLEAR WORD", command=lambda: clearCallback(wordText, sentence),
                         width=PLACEHOLDER_WIDTH)
    clearBtn.pack(pady=2)
    delBtn = tk.Button(rightPanelBottomFrame, text="DELETE LETTER", command=lambda: backspaceCallback(wordText, sentence),
                       width=PLACEHOLDER_WIDTH)
    delBtn.pack(pady=2)
    tk.Label(rightPanelBottomFrame, text="", bg=BACKGROUND_COLOR, height=HEIGHT_GAP).pack()
    tk.Label(rightPanelBottomFrame, text="SUGGESTED WORDS", bg=BACKGROUND_COLOR, fg=FONT_COLOR,
             font=BODY_FONT).pack()
    suggestedWordText = tk.Text(rightPanelBottomFrame, state=tk.DISABLED, relief=tk.RAISED, fg=FONT_COLOR,
                                font=BODY_FONT, width=PLACEHOLDER_WIDTH, height=5, wrap=tk.CHAR)
    suggestedWordText.pack()
    scrollBar = tk.Scrollbar(suggestedWordText)
    scrollBar.pack(side=tk.RIGHT, fill=tk.BOTH)
    suggestedWordText.config(yscrollcommand=scrollBar.set)
    scrollBar.config(command=suggestedWordText.yview)

    tk.Label(rightPanelBottomFrame, text="", bg=BACKGROUND_COLOR, height=HEIGHT_GAP).pack()
    tk.Label(rightPanelBottomFrame, text="SENTENCE FORMED", bg=BACKGROUND_COLOR, fg=FONT_COLOR,
             font=BODY_FONT).pack()
    sentence = tk.Text(rightPanelBottomFrame, state=tk.DISABLED, relief=tk.RAISED, fg=FONT_COLOR,
                       font=BODY_FONT, width=PLACEHOLDER_WIDTH, height=2)
    sentence.pack()
    spaceBtn = tk.Button(rightPanelBottomFrame, text="SPACE", command=lambda: spaceCallback(wordText, sentence),
                         width=PLACEHOLDER_WIDTH)
    spaceBtn.pack(pady=2)

    rightPanel.paneconfigure(rightPanelBottomFrame, minsize=448)
    rightPanel.add(rightPanelBottomFrame)
    return root, cap, label, wordText, suggestedWordText, sentence


class CoordsInfo:
    presentCoordinates_right = []
    beginning_coordinates_right = []
    coordinates_with_movement_right = []
    relative_landmark_list_right = []
    distance_right = []
    historyCoordinates_right = []
    history_movement_to_right = []
    number_of_frames_right = 1
    timer_right_hand = time.time()
    timer_after_proper_height_right = time.time()
    ready_to_start_right = False

    presentCoordinates_left = []
    beginning_coordinates_left = []
    coordinates_with_movement_left = []
    relative_landmark_list_left = []
    distance_left = 1
    historyCoordinates_left = []
    history_movement_to_left = []
    number_of_frames_left = 1
    timer_left_hand = time.time()
    timer_after_proper_height_left = time.time()
    ready_to_start_left = False

    miss_frame = False


# Define function to show frame
def show_frames(cap, label, wordText, suggestedWordText, sentence, obj):
    global mode, button_dict

    recognized_letter_left = ""
    recognized_letter_right = ""
    suggestedWords = []
    fps = cvFpsCalc.get()

    # Process Key (ESC: end)
    key = cv2.waitKey(10)
    if key == 32:
        letter_approved = True
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

    hands_list = []
    # helper when hand is out of frame with some frames in it
    if results.multi_handedness is not None:
        hands_list = which_hands(results.multi_handedness)

    pred_left = 38
    pred_right = 38

    # Hand landmarks
    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            # Bounding box calculation
            hand = handedness.classification[0].label
            if hand == "Right":
                if obj.number_of_frames_right == 1:

                    # check if it is not too soon for next sign
                    if time.time() - obj.timer_right_hand < 1.5:
                        continue

                    # first time right hand is visible
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # check if the sign is ready
                    predicted = predict_values_one_frame(pre_process_landmark(landmark_list))

                    if predicted < 0.75:
                        continue

                    # check if hand is high enough
                    if not check_height(landmark_list[9][1]):
                        continue

                    if not obj.ready_to_start_right:
                        obj.ready_to_start_right = True
                        obj.timer_after_proper_height_right = time.time()

                    if time.time() - obj.timer_after_proper_height_right < 0.7:
                        continue

                    obj.number_of_frames_right = obj.number_of_frames_right + 1
                    obj.presentCoordinates_right = pre_process_landmark(
                        landmark_list)

                    obj.historyCoordinates_right.append(pre_process_landmark(landmark_list))

                    obj.beginning_coordinates_right.append([landmark_list[12][0], landmark_list[12][1]])
                    obj.distance_right = []
                    obj.distance_right.append(int(math.sqrt(
                        math.pow(landmark_list[0][0] - landmark_list[5][0], 2) + math.pow(
                            landmark_list[0][1] - landmark_list[5][1], 2))))
                    # this can be wring for static ones, instead of 0.01 were 1
                    if obj.distance_right[0] == 0:
                        obj.distance_right[0] = 0.01

                else:
                    obj.number_of_frames_right = obj.number_of_frames_right + 1
                    if obj.number_of_frames_right == 8:
                        pred_right = predict_values_right(obj)
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    obj.historyCoordinates_right.append(pre_process_landmark(landmark_list))
                    obj.beginning_coordinates_right.append([landmark_list[12][0], landmark_list[12][1]])
                    obj.distance_right.append(int(math.sqrt(
                        math.pow(landmark_list[0][0] - landmark_list[5][0], 2) + math.pow(
                            landmark_list[0][1] - landmark_list[5][1], 2))))
                    if obj.distance_right[-1] == 0:
                        obj.distance_right[-1] = 0.01
                    tmp_coords = []
                    obj.coordinates_with_movement_right.append(tmp_coords)

                    # calculating relative change between every possible frames
                    count = 0
                    for i in obj.coordinates_with_movement_right:
                        obj.coordinates_with_movement_right[count].append((relative_change(
                            obj.beginning_coordinates_right[count][0], obj.beginning_coordinates_right[count][1],
                            landmark_list[12][0], landmark_list[12][1], obj.distance_right[count])))
                        count = count + 1
            if hand == "Left":
                if obj.number_of_frames_left == 1:

                    if time.time() - obj.timer_left_hand < 1.5:
                        continue
                        # first time left hand is visible
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # check if the sign is ready
                    predicted = predict_values_one_frame(pre_process_landmark(landmark_list))

                    if predicted < 0.75:
                        continue

                    # check if hand is high enough
                    if not check_height(landmark_list[9][1]):
                        continue
                    # check if it is not too soon for next sign
                    # wait until user stabilises hand
                    if not obj.ready_to_start_left:
                        obj.ready_to_start_left = True
                        obj.timer_after_proper_height_left = time.time()

                    if time.time() - obj.timer_after_proper_height_left < 0.7:
                        continue

                    obj.number_of_frames_left = obj.number_of_frames_left + 1
                    obj.presentCoordinates_left = pre_process_landmark(
                        landmark_list)

                    obj.historyCoordinates_left.append(pre_process_landmark(landmark_list))

                    obj.beginning_coordinates_left.append([landmark_list[12][0], landmark_list[12][1]])
                    obj.distance_left = []
                    obj.distance_left.append(int(math.sqrt(
                        math.pow(landmark_list[0][0] - landmark_list[5][0], 2) + math.pow(
                            landmark_list[0][1] - landmark_list[5][1], 2))))
                    # this can be wring for static ones, instead of 0.01 were 1
                    if obj.distance_left[0] == 0:
                        obj.distance_left[0] = 0.01

                else:
                    obj.number_of_frames_left = obj.number_of_frames_left + 1
                    if obj.number_of_frames_left == 8:
                        pred_left = predict_values_left(obj)
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    obj.historyCoordinates_left.append(pre_process_landmark(landmark_list))
                    obj.beginning_coordinates_left.append([landmark_list[12][0], landmark_list[12][1]])
                    obj.distance_left.append(int(math.sqrt(
                        math.pow(landmark_list[0][0] - landmark_list[5][0], 2) + math.pow(
                            landmark_list[0][1] - landmark_list[5][1], 2))))
                    if obj.distance_left[-1] == 0:
                        obj.distance_left[-1] = 0.01
                    tmp_coords = []
                    obj.coordinates_with_movement_left.append(tmp_coords)

                    # calculating relative change between every possible frames
                    count = 0
                    for i in obj.coordinates_with_movement_left:
                        obj.coordinates_with_movement_left[count].append((relative_change(
                            obj.beginning_coordinates_left[count][0], obj.beginning_coordinates_left[count][1],
                            landmark_list[12][0], landmark_list[12][1], obj.distance_left[count])))
                        count = count + 1
            # Conversion to relative coordinates / normalized coordinates
            # Drawing part
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            debug_image = draw_landmarks(debug_image, landmark_list)
            debug_image, recognized_letter = draw_info_text(
                debug_image,
                brect,
                handedness,
                all_letters[pred_left],
                None,
            )
    else:
        point_history.append([0, 0])

    if (not hands_list.__contains__("Right")) and obj.number_of_frames_right > 1:
        pred_right = predict_values_right(obj)
        if pred_right != 38:
            obj.timer_right_hand = time.time()
            obj.number_of_frames_right = 1
            obj.historyCoordinates_right = []
            obj.beginning_coordinates_right = []
            obj.distance_right = []
            obj.coordinates_with_movement_right = []
            obj.ready_to_start_right = False
        else:
            obj.number_of_frames_right = 1
            obj.historyCoordinates_right = []
            obj.beginning_coordinates_right = []
            obj.distance_right = []
            obj.coordinates_with_movement_right = []
            obj.ready_to_start_right = False

    if (not hands_list.__contains__("Left")) and obj.number_of_frames_left > 1:
        pred_left = predict_values_left(obj)
        if pred_left != 38:
            obj.timer_left_hand = time.time()
            obj.number_of_frames_left = 1
            obj.historyCoordinates_left = []
            obj.beginning_coordinates_left = []
            obj.distance_left = []
            obj.coordinates_with_movement_left = []
            obj.ready_to_start_left = False
        else:
            obj.number_of_frames_left = 1
            obj.historyCoordinates_left = []
            obj.beginning_coordinates_left = []
            obj.distance_left = []
            obj.coordinates_with_movement_left = []
            obj.ready_to_start_left = False

    debug_image = draw_point_history(debug_image, point_history)
    debug_image = draw_info(debug_image, fps, mode, number)
    debug_image = Image.fromarray(debug_image)

    photoImage = ImageTk.PhotoImage(image=debug_image)
    label.photoImage = photoImage
    label.configure(image=photoImage)

    recognized_letter_left = all_letters[pred_left]
    recognized_letter_right = all_letters[pred_right]

    # q case
    if recognized_letter_left == 'o' and recognized_letter_right == 'x':
        recognized_letter_left = 'q'
        recognized_letter_right = None
    elif recognized_letter_left == 'o' and obj.number_of_frames_right > 1 and recognized_letter_left != 'q':
        if all_letters[predict_values_right(obj)] == 'x':
            recognized_letter_left = 'q'
            recognized_letter_right = None
    elif recognized_letter_left == 'x' and obj.number_of_frames_right > 1 and recognized_letter_left != 'q':
        if all_letters[predict_values_right(obj)] == 'o':
            recognized_letter_left = 'q'
            recognized_letter_right = None

    if recognized_letter_right == 'o' and recognized_letter_left == 'x' and recognized_letter_left != 'q':
        recognized_letter_left = 'q'
        recognized_letter_right = None
    elif recognized_letter_right == 'o' and obj.number_of_frames_left > 1 and recognized_letter_left != 'q':
        if all_letters[predict_values_left(obj)] == 'x':
            recognized_letter_left = 'q'
            recognized_letter_right = None
    elif recognized_letter_right == 'x' and obj.number_of_frames_left > 1 and recognized_letter_left != 'q':
        if all_letters[predict_values_left(obj)] == 'o':
            recognized_letter_left = 'q'
            recognized_letter_right = None

    # print result from right hand
    if recognized_letter_right and pred_right != 38:

        readText = wordText.get("1.0", 'end-1c')
        readSentence = sentence.get("1.0", 'end-1c')

        sign_length = len(recognized_letter_right)
        if not readText or recognized_letter_right.lower() != readText[-sign_length:]:
            readText += recognized_letter_right.lower()
            readSentence += recognized_letter_right.lower()

            wordText.configure(state='normal')
            wordText.delete('1.0', tk.END)
            wordText.insert('1.0', readText)
            wordText.tag_add(1.0, "end")
            wordText.configure(state='disabled')

            sentence.configure(state='normal')
            sentence.delete('1.0', tk.END)
            sentence.insert('1.0', readSentence)
            sentence.tag_add(1.0, "end")
            sentence.configure(state='disabled')

            if not h.spell(readText):
                suggestedWords.append(h.suggest(readText))
            else:
                suggestedWords.append(h.suffix_suggest(readText))
                suggestedWords.append(h.stem(readText))

            for i in range(len(button_dict)):
                button_dict[i].destroy()

            suggestedWords = [word for words in suggestedWords for word in words]
            suggestedWordsTrunc = suggestedWords[:len(suggestedWords) if len(
                suggestedWords) < SUGGESTED_WORD_NUMBER else SUGGESTED_WORD_NUMBER]

            for i in range(len(suggestedWordsTrunc)):
                button_dict[i] = tk.Button(suggestedWordText, text=suggestedWordsTrunc[i],
                                           command=lambda idx=i:
                                           wordCallback(button_dict[idx]['text'] + " ", wordText, sentence))
                button_dict[i].pack(side=tk.RIGHT, padx=2, pady=1)

    # print result from left hand
    if (recognized_letter_left and pred_left != 38) or recognized_letter_left == 'q':
        readText = wordText.get("1.0", 'end-1c')
        readSentence = sentence.get("1.0", 'end-1c')

        sign_length = len(recognized_letter_left)
        if not readText or recognized_letter_left.lower() != readText[-sign_length:]:
            readText += recognized_letter_left.lower()
            readSentence += recognized_letter_left.lower()

            wordText.configure(state='normal')
            wordText.delete('1.0', tk.END)
            wordText.insert('1.0', readText)
            wordText.tag_add(1.0, "end")
            wordText.configure(state='disabled')

            sentence.configure(state='normal')
            sentence.delete('1.0', tk.END)
            sentence.insert('1.0', readSentence)
            sentence.tag_add(1.0, "end")
            sentence.configure(state='disabled')

            if not h.spell(readText):
                suggestedWords.append(h.suggest(readText))
            else:
                suggestedWords.append(h.suffix_suggest(readText))
                suggestedWords.append(h.stem(readText))

            for i in range(len(button_dict)):
                button_dict[i].destroy()

            suggestedWords = [word for words in suggestedWords for word in words]
            suggestedWordsTrunc = suggestedWords[:len(suggestedWords) if len(
                suggestedWords) < SUGGESTED_WORD_NUMBER else SUGGESTED_WORD_NUMBER]

            for i in range(len(suggestedWordsTrunc)):
                button_dict[i] = tk.Button(suggestedWordText, text=suggestedWordsTrunc[i],
                                           command=lambda idx=i:
                                           wordCallback(button_dict[idx]['text'] + " ", wordText, sentence))
                button_dict[i].pack(side=tk.RIGHT, padx=2, pady=1)

    # Repeat after an interval to capture continuously
    label.after(150, show_frames, cap, label, wordText, suggestedWordText, sentence, obj)


def start_app(cap, label, root, wordText, suggestedWordText, sentence):
    infoObject = CoordsInfo()
    show_frames(cap, label, wordText, suggestedWordText, sentence, infoObject)
    root.mainloop()


r, c, l, wt, swt, s = create_window()
start_app(c, l, r, wt, swt, s)
