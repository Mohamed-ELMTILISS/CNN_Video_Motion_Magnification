import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers.
    The structure is: ReflectionPad -> Conv -> ReLU -> ReflectionPad -> Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        # Padding is calculated to maintain the spatial dimensions
        padding = kernel_size // 2
        
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        )

    def forward(self, x):
        """
        The forward pass adds the input to the output of the convolutional block (skip connection).
        """
        return x + self.conv_block(x)


class MagNet(nn.Module):
    """
    The main MagNet model for video motion magnification.
    This implementation is based on the architecture described in "Learning-based Video Motion Magnification".
    It consists of three main parts: Encoder, Manipulator, and Decoder.
    """
    def __init__(self, n_channels=3, n_res_blocks=9):
        super(MagNet, self).__init__()
        
        # --- Encoder ---
        # The encoder takes an image and decomposes it into texture and shape representations.
        
        # Initial convolution layers
        self.enc_conv1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(n_channels, 64, 7), nn.ReLU(inplace=True))
        self.enc_conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.enc_conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        
        # Residual blocks for common encoding
        enc_res_blocks = [ResidualBlock(256, 256) for _ in range(n_res_blocks)]
        self.enc_res_blocks = nn.Sequential(*enc_res_blocks)
        
        # Separate branches for shape and texture
        self.shape_branch = nn.Sequential(ResidualBlock(256, 256), ResidualBlock(256, 256))
        self.texture_branch = nn.Sequential(ResidualBlock(256, 256), ResidualBlock(256, 256))

        # --- Decoder ---
        # The decoder takes the manipulated shape and texture representations to reconstruct the magnified frame.
        
        # Residual blocks for decoding
        dec_res_blocks = [ResidualBlock(512, 512) for _ in range(n_res_blocks)] # 512 because we concat shape and texture
        self.dec_res_blocks = nn.Sequential(*dec_res_blocks)
        
        # Upsampling layers
        self.dec_upsample1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True))
        self.dec_upsample2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True))
        
        # Final output layer
        self.dec_output = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(128, n_channels, 7), nn.Tanh())

    def encode(self, x):
        """
        Encodes the input image into shape and texture representations.
        """
        # Pass through initial convolutions and downsampling
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)
        
        # Pass through main residual blocks
        x = self.enc_res_blocks(x)
        
        # Split into two representations
        shape = self.shape_branch(x)
        texture = self.texture_branch(x)
        
        return shape, texture

    def manipulate(self, shape_a, shape_b, amplification_factor):
        """
        Magnifies the motion by manipulating the shape representations.
        Formula: M_a + alpha * (M_b - M_a)
        """
        # Ensure amplification_factor is broadcastable
        amp_factor_reshaped = amplification_factor.view(-1, 1, 1, 1)
        
        # Calculate the magnified shape representation
        motion = shape_b - shape_a
        magnified_motion = motion * amp_factor_reshaped
        magnified_shape = shape_a + magnified_motion
        
        return magnified_shape

    def decode(self, shape, texture):
        """
        Decodes the shape and texture representations back into an image.
        """
        # Concatenate shape and texture representations along the channel dimension
        x = torch.cat([shape, texture], dim=1)
        
        # Pass through decoder residual blocks
        x = self.dec_res_blocks(x)
        
        # Upsample to restore original dimensions
        x = self.dec_upsample1(x)
        x = self.dec_upsample2(x)
        
        # Generate the final output image
        output = self.dec_output(x)
        return output

    def forward(self, frame_a, frame_b, amplification_factor):
        """
        The complete forward pass for motion magnification.
        """
        # Encode both frames to get their shape and texture representations
        shape_a, texture_a = self.encode(frame_a)
        shape_b, _ = self.encode(frame_b) # We only need texture_a for reconstruction
        
        # Manipulate the shape representations to magnify motion
        magnified_shape = self.manipulate(shape_a, shape_b, amplification_factor)
        
        # Decode the magnified shape and original texture to get the final frame
        output_frame = self.decode(magnified_shape, texture_a)
        
        return output_frame

