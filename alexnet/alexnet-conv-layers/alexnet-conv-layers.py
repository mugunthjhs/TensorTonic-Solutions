import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """
    AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation).
    """
    B,H,W,C = image.shape
    conv_size = ((H + 2*(2) - 11)//4)+1
    out = [B,conv_size,conv_size,96]
    
    return np.zeros(out)