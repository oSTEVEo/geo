import numpy as np
from PIL import Image
import os
import torch
from torch import tensor, randn
import cv2

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


def create_dataset(paths, raito=0.70):
    # raito - процентное отношение, по умолчанию 70% на тренировку
    
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda")
    
    def to_categorical(y, num_classes):
        return tensor(np.eye(num_classes)[y]).to(device)
    
    dataset = []
    for i in range(len(paths)):         # Бежим по массиву каталогов 
        listdir = os.listdir(paths[i]) 
        for j in range(len(listdir)):
            # Здесь конвертируются файлы и соединяются с номером их типа
            listdir[j] = [Get_Filtered_Image(paths[i]+'/'+listdir[j]), i]
        dataset += listdir
    
    np.random.shuffle(dataset)          # Перемешиваем
    border = int(len(dataset) * raito)
    train_dataset = dataset[:border]    # Разделяем
    test_dataset = dataset[border:]
    
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    # Разбиваем x и y по отдельным массивам
    for item in train_dataset:          
        x_train += item[0]
        y_train += item[1] 
    
    for item in test_dataset:
        x_test += item[0] 
        y_test += item[1]
         
    # np.eye()
    y_train = to_categorical(y_train, 5)
    y_test  = to_categorical(y_test,  5)
    
    # Переводим массивы с x в тензоры
    x_train = tensor(x_train).to(device)
    x_test  = tensor(x_test ).to(device)
    
    # возвращаем 4 тензора: 2 тензора с кртинками и 2 тензора с ответами
    return tensor(xdata).to(device),