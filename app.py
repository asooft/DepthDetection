import streamlit as st
import pandas as pd
import plost
import tempfile
import os
import subprocess
import cv2
import numpy as np
import pickle
import VideoToFrame
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
import torch


st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.title('Video Object Detection and Depth Estimation')

# Allow the user to upload a video file
video_file = st.file_uploader('Upload a video file', type=['mp4'])

yolo_model_path = 'modelSmall.pkl'

with st.spinner('Loading Models'):
    feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-kitti")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")
    with open(yolo_model_path, 'rb') as f:
        yolo = pickle.load(f)


def GLPN(frames_dir, output_folder):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the predicted_depth model to the GPU (if needed)
    model.to(device)

    frame_count = 0

    for img in os.listdir(frames_dir):
        image = cv2.imread(os.path.join(frames_dir, img))

        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values

        # print(frame_count)
        with torch.no_grad():
            # Transfer pixel_values to the GPU (if needed)
            pixel_values = pixel_values.to(device)

            outputs = model(pixel_values)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=pixel_values.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        prediction = prediction.squeeze().cpu().numpy()

        # Invert the depth values
        inverted_prediction = np.max(prediction) - prediction

        # Normalize the inverted depth values
        # normalized_prediction = inverted_prediction / np.max(inverted_prediction)

        # Scale the normalized depth values to the range [0, 255]
        # scaled_prediction = (normalized_prediction * 255).astype("uint8")

        # Convert to grayscale
        # pred_d_gray = cv2.cvtColor(scaled_prediction, cv2.COLOR_BGR2GRAY)

        pred_d_numpy = (prediction / prediction.max()) * 255
        pred_d_numpy = pred_d_numpy.astype(np.uint8)
        pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)

        frame_number_padded = str(frame_count).zfill(6)  # Pad frame number with leading zeros

        output_path = os.path.join(output_folder, f"frame_{frame_number_padded}.jpg")
        cv2.imwrite(output_path, pred_d_color)
        frame_count += 1
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


# If the user has uploaded a video file, perform object detection on it
if video_file:
    # Specify the path where you want to save the file
    input_video_path = "INPUT_VIDEO"

    if not os.path.exists(input_video_path):
        os.makedirs(input_video_path)
        
    save_path = os.path.join(input_video_path, video_file.name)

    # Save the file to the specified path
    with open(save_path, "wb") as f:
        f.write(video_file.getbuffer())

    # Video to frames
    frames_dir = "INPUT_VIDEO_FRAMES"

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    frame_interval = int(os.environ.get('FPS', '5'))
    #frame_interval = int(os.environ.get('FRAME_INTERVAL') ) # Capture every 5th frame
    VideoToFrame.convert_video_to_frames(save_path, frames_dir, frame_interval)

    # Depth maps    
    depth_frames_dir = "DEPTH_FRAMES"

    if not os.path.exists(depth_frames_dir):
        os.makedirs(depth_frames_dir)

    # Run the command
    with st.spinner('Running depth estimation'):
        GLPN(frames_dir, depth_frames_dir)

    # yolo and merge
    with st.spinner('Running object detection'):
        output_frames = "OUTPUT_FRAMES"

        if not os.path.exists(output_frames):
            os.makedirs(output_frames)
            
        output_video = "OUTPUT_Video"

        if not os.path.exists(output_video):
            os.makedirs(output_video)
            
        for img in os.listdir(frames_dir):
            process_images(depth_path=os.path.join(depth_frames_dir, img), original_path=os.path.join(frames_dir, img),
                           depth_images_folder=output_frames)

    # Frames to output video
    output_path = os.path.join(output_video, "converted.webm")
    #output_path = os.path.join(os.environ.get('OUTPUT_VIDEO_PATH'), "converted.webm")
    fps = int(os.environ.get('FPS', '5'))
    VideoToFrame.convert_frames_to_video(output_frames, output_path, fps)
    print('Okkkkkkkkkkkkkkkkkkkk')
    st.write('okkkkkkkkkkk')

    st.video(output_path)

