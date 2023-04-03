def Get_Filtered_Image(path):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    image_original = cv2.imread(path, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
    filtered_image_y = cv2.filter2D(image_gray, -1, sobel_y)
    filtered_image_x = cv2.filter2D(image_gray, -1, sobel_x)
    sobel = cv2.addWeighted(filtered_image_x, 0.5, filtered_image_y, 0.5, 0)
    return sobel


def create_dataset(paths):
    import numpy as np
    from PIL import Image
    import os
    import torch
    from torch import tensor, randn

    torch.set_default_dtype(torch.float64)

    xdata = []
    xmeta = []
    dataset = []
    for i in range(len(paths)):
        listdir = os.listdir(paths[i])
        for j in range(len(listdir)):
            listdir[j] = [i, Get_Filtered_Image(paths[i]+'/'+listdir[j])]
        dataset += listdir

    return tensor(dataset).to(torch.device("cuda"))
