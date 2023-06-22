import cv2
import os

def convert_video_to_frames(video_path, output_dir, frame_interval):
    # Read the video
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize frame count and frame index
    frame_count = 0
    frame_index = 0

    # Read until the end of the video
    while True:
        # Read a single frame from the video
        ret, frame = video.read()

        # Break if no more frames are available
        if not ret:
            break

        # Check if the frame should be captured
        if frame_index % frame_interval == 0:
            # Generate the output frame filename
            frame_number_padded = str(frame_count).zfill(6)  # Pad frame number with leading zeros

            output_frame_path = os.path.join(output_dir, f"frame_{frame_number_padded}.jpg")

            # Save the frame as an image file
            cv2.imwrite(output_frame_path, frame)

            # Increment frame count
            frame_count += 1

        # Increment frame index
        frame_index += 1

    # Release the video capture object
    video.release()


def convert_frames_to_video(frames_dir, output_path, fps):
    # Get the list of frame filenames in the directory
    frame_filenames = sorted(os.listdir(frames_dir))

    # Determine the frame size by reading the first frame
    first_frame_path = os.path.join(frames_dir, frame_filenames[0])
    first_frame = cv2.imread(first_frame_path)
    frame_size = (first_frame.shape[1], first_frame.shape[0])

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Specify the video codec
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Write each frame to the video
    for frame_filename in frame_filenames:
        frame_path = os.path.join(frames_dir, frame_filename)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

