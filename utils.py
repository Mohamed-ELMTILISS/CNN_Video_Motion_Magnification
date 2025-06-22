import cv2
import os
from tqdm import tqdm
import glob

def extract_frames(video_path, output_dir):
    """
    Extracts all frames from a video file and saves them as PNGs in an output directory.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
    """
    print(f"Extracting frames from {video_path} to {output_dir}...")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_num = 0
    with tqdm(total=frame_count, desc="Extracting frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame as a PNG file, with zero-padded filename
            output_filename = os.path.join(output_dir, f"{frame_num:06d}.png")
            cv2.imwrite(output_filename, frame)
            frame_num += 1
            pbar.update(1)
            
    cap.release()
    print("Frame extraction complete.")
    return frame_count


def compile_frames(input_dir, output_video_path, fps=30):
    """
    Compiles a sequence of frames from a directory into a video file.

    Args:
        input_dir (str): Directory containing the input frames (e.g., '000001.png').
        output_video_path (str): Path to save the output video file.
        fps (int): Frames per second for the output video.
    """
    print(f"Compiling frames from {input_dir} into {output_video_path}...")
    frame_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if not frame_files:
        print(f"No frames found in {input_dir}. Cannot compile video.")
        return

    # Get frame dimensions from the first image
    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'XVID', 'MJPG', etc.
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in tqdm(frame_files, desc="Compiling video"):
        frame = cv2.imread(frame_file)
        out.write(frame)

    out.release()
    print("Video compilation complete.")

