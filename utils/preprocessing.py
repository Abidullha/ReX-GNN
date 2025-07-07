import numpy as np

def normalize_data(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std

def create_sliding_windows(data, input_len=12, output_len=3):
    T, N = data.shape
    X, Y = [], []
    for t in range(T - input_len - output_len + 1):
        x = data[t:t+input_len]
        y = data[t+input_len:t+input_len+output_len]
        X.append(x)
        Y.append(y)
    return np.stack(X), np.stack(Y)
