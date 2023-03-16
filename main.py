def create_dataset(paths):
    import numpy as np
    from PIL import Image
    import os
    import torch
    from torch import tensor, randn
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda")
    
    def to_categorical(y, num_classes):
        return tensor(np.eye(num_classes)[y])
    
    
    xdata = [] 
    ydata = []
    dataset = []
    for i in range(len(paths)):     # Бежим по массиву каталогов 
        listdir = os.listdir(paths[i]) 
        for j in range(len(listdir)):
            listdir[j] = [np.asarray(Image.open(paths[i] + "/" + listdir[j]).convert('RGB')), i]    # Здесь конвертируются файлы и соединяются с номером их типа
        dataset += listdir
    
    np.random.shuffle(dataset)      # Перемешиваем, разделяем
    for item in dataset:
        xdata = item[0]
        ydata = item[1]
    ydata = to_categorical(ydata, 10)   # np.eye()
    
    return tensor(xdata).to(device), tensor(ydata).to(device)     # возвращаем 2 тензора




print(1)