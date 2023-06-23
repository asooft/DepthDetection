import streamlit as st
import pandas as pd
import plost
import tempfile
import os
import subprocess
import cv2
import pickle
import VideoToFrame
import requests


st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='Video Object Detection and Depth Estimation')
st.title('Video Object Detection and Depth Estimation')

yolo_model_path = 'modelSmall.pkl'
with open(yolo_model_path, 'rb') as f:
    yolo = pickle.load(f)

@st.cache_data()
def download_models():
    # Clone git repo
    subprocess.run(['git', 'clone', 'https://github.com/compphoto/BoostingMonocularDepth.git'])

    # Download latest_net_G.pth
    
    
    #url = 'https://sfu.ca/~yagiz/CVPR21/latest_net_G.pth'
    #response = requests.get(url)

    #with open('latest_net_G.pth', 'wb') as f:
    #    f.write(response.content)
    #subprocess.run(['wget', 'https://sfu.ca/~yagiz/CVPR21/latest_net_G.pth'])
    subprocess.run(['curl', '-o', 'latest_net_G.pth', 'https://sfu.ca/~yagiz/CVPR21/latest_net_G.pth'])


    # Downloading merge model weights
    subprocess.run(['mkdir', '-p', 'BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/'])
    subprocess.run(['mv', 'latest_net_G.pth', 'BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/'])

    # Downloading Midas weights
    url = 'https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt'
    subprocess.run(['curl', '-o', 'midas_v21-f6b98070.pt', url])
    #subprocess.run(['wget', 'https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt'])
    subprocess.run(['mv', 'midas_v21-f6b98070.pt', 'BoostingMonocularDepth/midas/model.pt'])


def process_images(depth_path, original_path,depth_images_folder):
    depth_image = cv2.imread(depth_path)
    depth_shape = depth_image.shape

    original = cv2.imread(original_path)
    original = cv2.resize(original, (depth_shape[1], depth_shape[0]))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    predictions = yolo.predict(original_rgb)
    #draw bounding boxes on depth map
    for box in list(predictions._images_prediction_lst)[0].prediction.bboxes_xyxy:
        x1, y1, x2, y2 = box
        cv2.rectangle(depth_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    filename = depth_path.split('/')[-1]
    filepath = os.path.join(depth_images_folder, filename)
    cv2.imwrite(filepath, depth_image)

# Allow the user to upload a video file
video_file = st.file_uploader('Upload a video file', type=['mp4'])

# Download needed models
download_models()

# If the user has uploaded a video file, perform object detection on it
if video_file:
    # Specify the path where you want to save the file
    
    input_video_path = "INPUT_VIDEO"

    if not os.path.exists(input_video_path):
        os.makedirs(input_video_path)
    #input_video_path = os.environ.get('INPUT_VIDEO_PATH')
    save_path = os.path.join(input_video_path, video_file.name)

    # Save the file to the specified path
    with open(save_path, "wb") as f:
        f.write(video_file.getbuffer())

    # Video to frames
    frames_dir = "INPUT_VIDEO_FRAMES"

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        
    #frames_dir = os.environ.get('INPUT_VIDEO_FRAMES_PATH')
    frame_interval = int(os.environ.get('FPS') ) # Capture every 5th frame
    VideoToFrame.convert_video_to_frames(save_path, frames_dir, frame_interval)

    # Depth maps
    depth_frames_dir = "DEPTH_FRAMES"

    if not os.path.exists(depth_frames_dir):
        os.makedirs(depth_frames_dir)
    #depth_frames_dir = os.environ.get('DEPTH_FRAMES_PATH')
    depth_model_path = 'BoostingMonocularDepth/run.py'

    # change directory
    os.chdir("BoostingMonocularDepth/")

    # Construct the command using string formatting
    command = ["python", depth_model_path, "--Final", "--data_dir", frames_dir,
               "--output_dir", depth_frames_dir, "--depthNet", str(0), "--colorize_results", "--max_res", str(2000)]

    # Run the command
    with st.spinner('Running depth estimation'):
        subprocess.run(command, check=True)

    # yolo and merge
    with st.spinner('Running object detection'):
        output_frames = "OUTPUT_FRAMES"

        if not os.path.exists(output_frames):
            os.makedirs(output_frames)
        #output_frames = os.environ.get('OUTPUT_FRAMES_PATH')
        for img in os.listdir(frames_dir):
            depth_path = os.path.join(depth_frames_dir, img.split('.jpg')[0] + '.png')
            process_images(depth_path=depth_path, original_path=os.path.join(frames_dir, img),
                           depth_images_folder=output_frames)

    # Frames to output video
    output_path = os.path.join(output_frames, "converted.mp4")
    fps = int(os.environ.get('FPS')) # Frames per second
    VideoToFrame.convert_frames_to_video(output_frames, output_path, fps)

    st.video(output_path)

