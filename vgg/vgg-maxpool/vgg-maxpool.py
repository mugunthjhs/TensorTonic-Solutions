import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    B, H, W, C = x.shape

    h_out = H // 2
    w_out = W // 2 

    out = np.zeros([B, h_out,w_out,C])

    stride = 2
    filter_size = 2

    for i in range(h_out):
        for j in range(w_out):

            start_row = i * stride
            start_col = j * stride

            window = x[:, start_row: start_row + filter_size, start_col: start_col + filter_size,:]
            max = window.max(axis=(1,2))
            out[:,i,j,:] = max

    return out