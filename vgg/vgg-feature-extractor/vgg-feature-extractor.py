import numpy as np

def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H//2, 2, W//2, 2, C).max(axis=(2, 4))

def vgg_features(x: np.ndarray, config: list, conv_weights: list, conv_biases: list) -> np.ndarray:
    """
    Returns: np.ndarray feature tensor after applying conv layers and max pooling
    """
    current = x.copy()
    weight_index = 0
    
    for i in config:
        if isinstance(i,int):
            weight = np.array(conv_weights[weight_index])
            bias = np.array(conv_biases[weight_index])

            output = np.zeros([current.shape[0], current.shape[1], current.shape[2], weight.shape[1]])

            for i in range(current.shape[0]):
                for j in range(current.shape[1]):
                    for k in range(current.shape[2]):
                        feature = current[i, j, k]
                        out = feature @ weight + bias
                        out = np.maximum(0, out)
                        output[i,j,k] = out

            current = output
            weight_index += 1

        else:
            current = maxpool_2x2(current)

    return current