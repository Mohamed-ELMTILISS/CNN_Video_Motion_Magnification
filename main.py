import argparse
from train import train_model
from inference import magnify_video

def main():
    parser = argparse.ArgumentParser(description="Learning-based Video Motion Magnification")
    parser.add_argument('--phase', type=str, required=True, choices=['train', 'inference'],
                        help="The phase to run: 'train' or 'inference'")
    
    # --- Training Arguments ---
    parser.add_argument('--data_root', type=str, default='./data',
                        help="Root directory of the dataset for training")
    parser.add_argument('--num_epochs', type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="Learning rate for the Adam optimizer")
    parser.add_argument('--save_epoch_freq', type=int, default=5,
                        help="Frequency (in epochs) to save model checkpoints")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Path to a checkpoint to resume training from (for fine-tuning).")
    parser.add_argument('--finetune_lr', type=float, default=None,
                        help="Use a different (usually smaller) learning rate for fine-tuning. Overrides --learning_rate.")


    # --- Inference Arguments ---
    parser.add_argument('--input_video', type=str, default='./input.mp4',
                        help="Path to the input video file to magnify")
    parser.add_argument('--output_video', type=str, default='./output.mp4',
                        help="Path to save the magnified output video")
    parser.add_argument('--amplification_factor', type=float, default=10.0,
                        help="The factor by which to magnify motion")
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'dynamic'],
                        help="Magnification mode: 'static' (vs. first frame) or 'dynamic' (vs. previous frame)")
    parser.add_argument('--keep_frames', action='store_true',
                        help="If set, keeps the temporary directories with extracted and magnified frames.")

    # --- Common Arguments ---
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help="Directory to save/load model checkpoints")
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/magnet_epoch_100.pth',
                        help="Path to a specific checkpoint file for inference")
    parser.add_argument('--n_res_blocks', type=int, default=9,
                        help="Number of residual blocks in the model")
                        
    args = parser.parse_args()
    config = vars(args)

    if args.phase == 'train':
        train_model(config)
    elif args.phase == 'inference':
        magnify_video(config)

if __name__ == '__main__':
    main()
