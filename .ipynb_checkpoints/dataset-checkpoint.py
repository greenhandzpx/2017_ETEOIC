import imghdr
import torch
from torch.utils import data
from torchvision import transforms
from glob import glob
from PIL import Image
import os

class DataSet(data.Dataset):
    def __init__(self, data_dir, image_size=256) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_path = sorted(glob(os.path.join(self.data_dir, '*.*')))[:1500]

    def __getitem__(self, index):
        # 定义每次读到的图像
        image_name = self.image_path[index]
        image = Image.open(image_name).convert('RGB')
        #print(image)
        # 做一下图像增强
        tranform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        )
        #print(tranform(image))
        #return tranform(image).to(torch.float)
        return tranform(image) #* 255 

    def __len__(self):
        # 定义需要迭代的次数
        return len(self.image_path)

def dataLoader(data_path, batch_size, image_size=256):
    dataset = DataSet(data_path, image_size=image_size)
    #test_dataset = DataSet(test_data_path)

    images =  data.DataLoader(dataset, batch_size=batch_size, num_workers=13)
    #test_data = data.DataLoader(test_dataset, batch_size=batch_size)
    
    return images#, test_data