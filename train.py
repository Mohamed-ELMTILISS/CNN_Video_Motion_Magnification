import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os

from model import MagNet
from dataset import get_dataloader

def train_model(config):
    """
    Main function to train the MagNet model, with support for resuming from a checkpoint.
    
    Args:
        config (dict): A dictionary containing training configuration.
    """
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = MagNet(n_res_blocks=config['n_res_blocks']).to(device)
    
    # Define optimizer
    # Use a smaller learning rate for fine-tuning if specified
    lr = config['finetune_lr'] if config['finetune_lr'] is not None else config['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Loss function
    criterion = nn.L1Loss()

    # --- Load Checkpoint if Resuming ---
    start_epoch = 0
    if config['resume_checkpoint']:
        if os.path.exists(config['resume_checkpoint']):
            checkpoint = torch.load(config['resume_checkpoint'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming training from epoch {start_epoch + 1}...")
        else:
            print(f"Warning: Checkpoint not found at {config['resume_checkpoint']}. Starting from scratch.")

    # Create the data loader
    dataloader = get_dataloader(
        data_root=config['data_root'],
        batch_size=config['batch_size']
    )

    print("Starting training...")
    
    # --- Training Loop ---
    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        total_loss = 0.0
        
        for i, batch in enumerate(progress_bar):
            frame_a = batch['frame_a'].to(device)
            frame_b = batch['frame_b'].to(device)
            ground_truth = batch['ground_truth'].to(device)
            amp_factor = batch['amplification_factor'].float().to(device)

            output_frame = model(frame_a, frame_b, amp_factor)
            
            loss = criterion(output_frame, ground_truth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': optimizer.param_groups[0]['lr']})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        
        # --- Save Checkpoint ---
        if (epoch + 1) % config['save_epoch_freq'] == 0:
            checkpoint_dir = config['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"magnet_epoch_{epoch+1}.pth")
            
            # Save model, optimizer state, and epoch number
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training finished.")
