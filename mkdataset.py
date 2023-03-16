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
            listdir[j] = [np.asarray(Image.open(paths[i] + "/" + listdir[j]).convert('RGB')), i]
        dataset += listdir
    
    np.random.shuffle(dataset)
    
    return dataset
    
    
create_dataset(["seg_train/sea", "seg_train/buildings", "seg_train/glacier", "seg_train/street", "seg_train/forest", "seg_train/mountain"])
print(1)
    