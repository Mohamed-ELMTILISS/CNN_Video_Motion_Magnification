
# A Learning-Based Approach to Video Motion Magnification in PyTorch

This repository provides a PyTorch implementation of the research paper:
**"Learning-Based Video Motion Magnification"** by Tae-Hyun Oh *et al.*

It offers a deep convolutional neural network (CNN) to **amplify subtle motions** in video sequences, enabling training on custom datasets, fine-tuning of pretrained models, and inference to produce motion-magnified video outputs.

---

## 🔍 Core Features

* **Encoder-Manipulator-Decoder Architecture**
  Implements the model architecture as proposed in the original paper.

* **Flexible Training**
  Supports training from scratch on your own dataset.

* **Checkpoint Fine-Tuning**
  Resume training or fine-tune pretrained models for domain adaptation.

* **End-to-End Inference Pipeline**
  Includes video frame extraction, motion magnification, and reassembly into video.

* **Two Magnification Modes**

  * **Static Mode**: Amplifies motion with respect to the first frame.
  * **Dynamic Mode**: Amplifies motion between consecutive frames (i.e., velocity magnification).

---

## 📁 Project Structure

```
├── data/                    # Training data directory
│   └── train/
│       ├── FrameA/          # Frames at time t-1
│       ├── FrameB/          # Frames at time t
│       └── FrameC/          # Ground-truth frames at time t+1
├── checkpoints/             # Saved model weights
├── main.py                  # Main entry point for training and inference
├── model.py                 # CNN architecture (MagNet)
├── dataset.py               # Dataset and DataLoader definitions
├── train.py                 # Training loop
├── inference.py             # Inference logic
├── utils.py                 # Video/frame utility functions
└── requirements.txt         # Python dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/video-motion-magnification.git
cd video-motion-magnification
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

Make sure to install a compatible PyTorch version for your system and GPU.

```bash
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Training

### 📂 Dataset Format

Prepare your training dataset in the following structure under `./data/train/`:

```
FrameA/    -> frame at time t-1  
FrameB/    -> frame at time t  
FrameC/    -> ground truth frame at time t+1  
```

Each folder should contain aligned image sequences with identical file names.

### 🚀 Train from Scratch

```bash
python main.py --phase train --num_epochs 50 --learning_rate 0.0001 --batch_size 4
```

### 🔄 Fine-Tune from Checkpoint

```bash
python main.py \
    --phase train \
    --num_epochs 50 \
    --resume_checkpoint ./checkpoints/magnet_epoch_50.pth \
    --finetune_lr 1e-5
```

---

## 🎥 Inference

Apply the trained model to magnify motion in a video:

```bash
python main.py \
    --phase inference \
    --input_video path/to/input.mp4 \
    --output_video path/to/output.mp4 \
    --checkpoint_path ./checkpoints/magnet_epoch_50.pth \
    --amplification_factor 15.0 \
    --mode static
```

### 📌 Inference Arguments

* `--input_video`: Path to input video file
* `--output_video`: Path for saving magnified output video
* `--checkpoint_path`: Path to a trained `.pth` model checkpoint
* `--amplification_factor`: Controls the intensity of motion magnification
* `--mode`: `"static"` or `"dynamic"`

---

## 📖 Citation

If you use this code or find it helpful, please cite the original paper:

```bibtex
@article{oh2018learning,
  title={Learning-based Video Motion Magnification},
  author={Oh, Tae-Hyun and Jaroensri, Ronnachai and Kim, Changil and Elgharib, Mohamed and Durand, Fr{\'e}do and Freeman, William T and Matusik, Wojciech},
  journal={arXiv preprint arXiv:1804.02684},
  year={2018}
}
```

---

## 💡 Acknowledgements

This implementation is inspired by the official work of the authors. Please consider visiting the [original paper](https://arxiv.org/abs/1804.02684) for more details.


