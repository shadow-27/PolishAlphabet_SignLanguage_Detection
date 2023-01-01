import cv2
import time
import os

def video_to_frames(input_loc, output_loc,num):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
        num:Integer >2 to extract number of frames
    Returns:
        None
    """
    if num<2:
        print('Enter valid frame number to extract')
        return
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        
        if(count==0 or count==(video_length-1)):
            cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        frame_list = []    
        for f in range(1,num-1):
            n=math.ceil((f/(num-1))*video_length)-1
            frame_list.append(n)

        if count in frame_list:
            cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)

        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % num)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

if __name__=="__main__":

    input_loc = 'C:/Users/HP/Desktop/video_to_frames/PolishAlphabet_SignLanguage_Detection/input2/5Ąs.mp4'
    output_loc = 'C:/Users/HP/Desktop/video_to_frames/PolishAlphabet_SignLanguage_Detection/output2'
    video_to_frames(input_loc, output_loc,4)