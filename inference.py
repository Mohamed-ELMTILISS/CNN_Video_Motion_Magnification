import torch
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm
import shutil

from model import MagNet
from utils import extract_frames, compile_frames # Import the new functions

def magnify_video(config):
    """
    Magnifies a video file using a trained MagNet model.
    Handles frame extraction and final video compilation.

    Args:
        config (dict): A dictionary containing inference configuration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup Temporary Directories ---
    temp_input_dir = os.path.join(config['checkpoint_dir'], 'temp_input_frames')
    temp_output_dir = os.path.join(config['checkpoint_dir'], 'temp_output_frames')

    if os.path.exists(temp_input_dir): shutil.rmtree(temp_input_dir)
    if os.path.exists(temp_output_dir): shutil.rmtree(temp_output_dir)

    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)

    # --- 1. Extract frames from the input video ---
    extract_frames(config['input_video'], temp_input_dir)

    # Initialize the model and load the trained weights
    model = MagNet(n_res_blocks=config['n_res_blocks']).to(device)
    model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device))
    model.eval()  # Set the model to evaluation mode

    # Get the list of input frames
    frame_files = sorted(glob.glob(os.path.join(temp_input_dir, '*.png')))
    if not frame_files:
        print(f"Frame extraction failed. No frames found in {temp_input_dir}.")
        return

    # Define image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Get the reference frame (Frame A)
    if config['mode'] == 'static':
        ref_frame_path = frame_files[0]
        ref_frame_img = Image.open(ref_frame_path).convert('RGB')
        ref_frame_tensor = transform(ref_frame_img).unsqueeze(0).to(device)

    print(f"Starting frame-by-frame magnification in '{config['mode']}' mode...")

    # --- 2. Process frames with the model ---
    with torch.no_grad(): # Disable gradient calculations
        for i in tqdm(range(len(frame_files)), desc="Magnifying frames"):
            if config['mode'] == 'dynamic':
                ref_idx = max(0, i - 1) # Use previous frame, or first frame for the very first
                ref_frame_path = frame_files[ref_idx]
                ref_frame_img = Image.open(ref_frame_path).convert('RGB')
                ref_frame_tensor = transform(ref_frame_img).unsqueeze(0).to(device)

            current_frame_path = frame_files[i]
            current_frame_img = Image.open(current_frame_path).convert('RGB')
            current_frame_tensor = transform(current_frame_img).unsqueeze(0).to(device)
            
            amp_factor = torch.tensor([config['amplification_factor']], dtype=torch.float32).to(device)
            
            output_tensor = model(ref_frame_tensor, current_frame_tensor, amp_factor)

            output_image = output_tensor.squeeze(0).cpu().detach()
            output_image = (output_image * 0.5) + 0.5
            output_image = transforms.ToPILImage()(output_image)
            
            output_filename = os.path.basename(current_frame_path)
            output_path = os.path.join(temp_output_dir, output_filename)
            output_image.save(output_path)

    # --- 3. Compile magnified frames back into a video ---
    compile_frames(temp_output_dir, config['output_video'])
    
    # --- 4. Clean up temporary directories ---
    if not config['keep_frames']:
        print("Cleaning up temporary frame directories...")
        shutil.rmtree(temp_input_dir)
        shutil.rmtree(temp_output_dir)

    print(f"Inference complete. Magnified video saved to {config['output_video']}")

