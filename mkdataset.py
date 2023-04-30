import numpy as np
from PIL import Image
import os
import torch
from torch import tensor, randn
import cv2

torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

def init_dataset():
    x_train, y_train = create_dataset(["seg_train/sea", "seg_train/buildings",
               "seg_train/street", "seg_train/forest", "seg_train/mountain"])
    x_test, y_test = create_dataset(["seg_test/sea", "seg_test/buildings",
               "seg_test/street", "seg_test/forest", "seg_test/mountain"])
    
    def F(data):
        return tensor(data ,device=device)
    
    return map(F, (x_train, y_train, x_test, y_test))

def Get_Filtered_Image(path):
    # cv2 needed
    image_original = cv2.imread(path, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
    filtered_image_y = cv2.filter2D(image_gray, -1, sobel_y)
    filtered_image_x = cv2.filter2D(image_gray, -1, sobel_x)
    sobel = cv2.addWeighted(filtered_image_x, 0.5, filtered_image_y, 0.5, 0)
    return sobel

def get_image(path):
    image_original = cv2.imread(path, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    return image_gray / 255
    

def create_dataset(paths):
    def to_categorical(y, num_classes):
        return np.eye(num_classes)[y]
    
    dataset = []
    for i in range(len(paths)):        # Бежим по массиву каталогов 
        listdir = os.listdir(paths[i]) 
        for j in range(len(listdir)):
            listdir[j] = [get_image(paths[i]+'/'+listdir[j]), i]    # Здесь конвертируются файлы и соединяются с номером их типа
        dataset += listdir
    
    xdata = [0] * len(dataset) 
    ydata = [0] * len(dataset)
    
    np.random.shuffle(dataset)         # Перемешиваем, разделяем x и y
    for i in range(len(dataset)):
        xdata[i] = dataset[i][0]
        ydata[i] = dataset[i][1]
    
    xdata = np.array(xdata)
    ydata = to_categorical(ydata, 5)   # np.eye()
    
    
    return xdata, ydata                # возвращаем 2 тензора