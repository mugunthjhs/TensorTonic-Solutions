import numpy as np

def vgg_conv_block(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, H, W, C_out) after sequential linear transforms with ReLU
    """
    current = x.copy()
    
    for w, b in zip(weights, biases):
        w = np.array(w)
        output = np.zeros((current.shape[0],current.shape[1],current.shape[2],w.shape[1]))
        for i in range(current.shape[0]):
            for j in range(current.shape[1]):
                for k in range(current.shape[2]):
                    feature = current[i,j,k]
                    out = feature @ np.array(w) + np.array(b)
                    out = np.maximum(0, out)
                    output[i,j,k] = out
        current = output


    return current
    