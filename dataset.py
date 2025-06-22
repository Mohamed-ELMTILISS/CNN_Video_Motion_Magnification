import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class VideoMagnificationDataset(Dataset):
    """
    Custom PyTorch Dataset for loading video frames for the magnification task.
    This version is adapted for a dataset structure of (t-1, t, t+1).
    - FrameA: Corresponds to time t-1
    - FrameB: Corresponds to time t
    - FrameC: Corresponds to time t+1 (used as the ground truth)
    """
    def __init__(self, data_root, mode='train', image_size=(256, 256)):
        """
        Args:
            data_root (str): The root directory of the dataset.
            mode (str): 'train' or 'val'.
            image_size (tuple): The size to resize images to.
        """
        self.data_root = os.path.join(data_root, mode)
        self.image_size = image_size
        
        # Define paths to the frame directories
        self.frame_a_dir = os.path.join(self.data_root, 'frameA')
        self.frame_b_dir = os.path.join(self.data_root, 'frameB')
        self.frame_c_dir = os.path.join(self.data_root, 'frameC')
        
        # Get the list of image filenames (assuming they are consistently named)
        self.image_files = sorted(os.listdir(self.frame_a_dir))
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            # Normalize to the range [-1, 1] as expected by the Tanh activation in the model
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieves a data sample: frame_a (t-1), frame_b (t), ground_truth (t+1),
        and a fixed amplification factor of 2.0.
        """
        image_name = self.image_files[idx]
        
        # Load the images
        frame_a_path = os.path.join(self.frame_a_dir, image_name)
        frame_b_path = os.path.join(self.frame_b_dir, image_name)
        frame_c_path = os.path.join(self.frame_c_dir, image_name)
        
        frame_a = Image.open(frame_a_path).convert('RGB')
        frame_b = Image.open(frame_b_path).convert('RGB')
        ground_truth = Image.open(frame_c_path).convert('RGB')
        
        # Apply transformations
        frame_a = self.transform(frame_a)
        frame_b = self.transform(frame_b)
        ground_truth = self.transform(ground_truth)
        
        # To predict frame t+1 (ground_truth) from t-1 (frame_a) and t (frame_b),
        # we are performing a linear extrapolation of motion. This corresponds
        # to an amplification factor of 2.0 in the model's formulation.
        amplification_factor = 2.0
        
        sample = {
            'frame_a': frame_a,
            'frame_b': frame_b,
            'ground_truth': ground_truth,
            'amplification_factor': amplification_factor
        }
        
        return sample

def get_dataloader(data_root, batch_size=4, shuffle=True, num_workers=4):
    """
    Creates a PyTorch DataLoader for the video magnification dataset.
    """
    dataset = VideoMagnificationDataset(data_root=data_root, mode='train')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
