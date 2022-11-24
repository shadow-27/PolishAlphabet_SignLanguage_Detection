# This is used to covert all videos to frames(frames for each video is stored in Sujith's Hard-drive)

import os
import numpy as np
import cv2

root_dir = 'C:\\Users\\HP\\Desktop\\video_to_frames\\PolishAlphabet_SignLanguage_Detection\\Polish_Letters\\train'
dest_dir = 'C:\\Users\\HP\\Desktop\\video_to_frames\\PolishAlphabet_SignLanguage_Detection\\output2'

if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

# To list what are the directories - train, test
data_dir_list = os.listdir(root_dir)

def vid_to_frames():
    for data_dir in data_dir_list: # read the train  directory one 
        data_path = os.path.join(root_dir,data_dir) # 'Alphabet/train'
        dest_data_path = os.path.join(dest_dir,data_dir) # 'frames/train'
        if not os.path.exists(dest_data_path):
            os.mkdir(dest_data_path)
        
        activity_list = os.listdir(data_path) # there are  individual  letters directories ['A', 'Ą', 'B'.....] etc
        
        for activity in activity_list: # loop over every letter folder
            activity_path = os.path.join(data_path,activity) # 'Alphabet/train/A'
            dest_activity_path = os.path.join(dest_data_path,activity) # 'frames/train/A'
            if not os.path.exists(dest_activity_path):
                os.mkdir(dest_activity_path)
            write_frames(activity_path,dest_activity_path)
    
def write_frames(activity_path,dest_activity_path):
    # read the list of video from 'Alphabet/train/A' - [A1.mp4,A2.mp4, ......]
    print(activity_path) 
    vid_list = os.listdir(activity_path)
    
    for vid in vid_list: # A1.mp4
        dest_folder_name = vid[:-4] # v_Archery_g01_c01
        dest_folder_path = os.path.join(dest_activity_path,dest_folder_name) # 'frames/train/A/A1'
        if not os.path.exists(dest_folder_path):
            os.mkdir(dest_folder_path)
            
        vid_path = os.path.join(activity_path,vid)  # 'Alphabet/train/A/A1.mp4'
        #print ('video path: ', vid_path)
        cap = cv2.VideoCapture(vid_path) # initialize a cap object for reading the video
        
        ret=True
        frame_num=0
        while ret:
            ret, img = cap.read()
            
            if ret:
                
                output_file_name = 'img_{:06d}'.format(frame_num) + '.png' # img_000001.png
            # output frame to write 'frames/train/A/A1/img_000001.png'
                output_file_path = os.path.join(dest_folder_path, output_file_name)
                output_file_path = output_file_path.replace('\\', '/')

                print('Output path:',output_file_path)
                cv2.imwrite(output_file_path, img)

                frame_num += 1
                #print("Frame no. ", frame_num)
                try:
                    #cv2.imshow('img',img)
                    cv2.waitKey(5)
                    cv2.imwrite(output_file_path, img) # writing frames to defined location
                except Exception as e:
                    print(e)
                if ret==False:
                    cv2.destroyAllWindows()
                    cap.release()
if __name__ == '__main__':
    vid_to_frames()