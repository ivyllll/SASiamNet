import glob,os,torch
from PIL import Image
from utils import *
import numpy as np
from torchvision import transforms
import cv2

class Change_Detect(torch.utils.data.Dataset):
    """
    对img1,img2,label进行处理
    img1和img2->tensor,
    label->one-hot->（n_classes,H,W）
    """
    #__XX__: magic function
    def __init__(self,img1_path,img2_path,label_path,csv_path,mode='train'): #__init__魔法函数，每次调用这个类的时候都会先调用这个函数（相当于构造函数）
        super().__init__()
        self.mode = mode

        self.img1_list  = glob.glob(os.path.join(img1_path,'*.jpg'))#glob.glob()返回一个某一种文件夹下面的某一类型文件路径列表
        self.img2_list  = glob.glob(os.path.join(img2_path,'*.jpg'))
        self.label_list = glob.glob(os.path.join(label_path,'*.jpg'))

        # self.img1_list.sort()
        # self.img2_list.sort()
        # self.label_list.sort()

        self.label_info = get_label_info(csv_path)
        self.to_tensor  = transforms.ToTensor() #将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor,其将每一个数值归一化到[0,1](直接除以255)

    def __getitem__(self, index):
        img1  = Image.open(self.img1_list[index])
        img1  = self.to_tensor(img1).float() #torch.Size([3, 512, 512])
        img2  = Image.open(self.img2_list[index])
        img2  = self.to_tensor(img2).float()
        label = Image.open(self.label_list[index])

        label = one_hot_it(label,self.label_info).astype(np.uint8) #torch.Size([512, 512, 2]
        label = np.transpose(label,[2,0,1]).astype(np.float32)
        label = torch.from_numpy(label) #torch.from_numpy()用来将数组array转换为张量Tensor(2,512,512)
        return img1,img2,label

    def __len__(self):
        # print(len(self.img1_list))
        return len(self.img1_list)

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # data = Change_Detect(img1_path="D:/.ipynb_checkpoints/CD/DATA/LEVIR/train/img1",
    #                      img2_path="D:/.ipynb_checkpoints/CD/DATA/LEVIR/train/img2",
    #                      label_path="D:/.ipynb_checkpoints/CD/DATA/LEVIR/train/label",
    #                      csv_path="D:/.ipynb_checkpoints/CD/DATA/LEVIR/class_dict.csv",
    #                      )

    # data = Change_Detect(img1_path ="D:/.ipynb_checkpoints/CD/DATA/train/img1",
    #                      img2_path ="D:/.ipynb_checkpoints/CD/DATA/train/img2",
    #                      label_path="D:/.ipynb_checkpoints/CD/DATA/train/label",
    #                      csv_path  ="D:/.ipynb_checkpoints/CD/DATA/class_dict.csv",
    #                     )

    # data = Change_Detect(img1_path="D:/.ipynb_checkpoints/CD/DATA/LEVIR_512/train/img1",
    #                      img2_path="D:/.ipynb_checkpoints/CD/DATA/LEVIR_512/train/img2",
    #                      label_path="D:/.ipynb_checkpoints/CD/DATA/LEVIR_512/train/label",
    #                      csv_path="D:/.ipynb_checkpoints/CD/DATA/LEVIR_512/class_dict.csv",
    #                      )

    data = Change_Detect(img1_path="D:/.ipynb_checkpoints/CD/DATA/CDD/train/img1",
                         img2_path="D:/.ipynb_checkpoints/CD/DATA/CDD/train/img2",
                         label_path="D:/.ipynb_checkpoints/CD/DATA/CDD/train/label",
                         csv_path="D:/.ipynb_checkpoints/CD/DATA/CDD/class_dict.csv",
                         )

    # data = Change_Detect(img1_path="C:/YHY/CDYHY/data_6/train/img1",
    #                      img2_path="C:/YHY/CDYHY/data_6/train/img2",
    #                      label_path="C:/YHY/CDYHY/data_6/train/label",
    #                      csv_path="C:/YHY/CDYHY/data_6/class_dict.csv",
    #                      )

    # data = Change_Detect(img1_path="C:/YHY/CDYHY/CDD/train/img1",
    #                      img2_path="C:/YHY/CDYHY/CDD/train/img2",
    #                      label_path="C:/YHY/CDYHY/CDD/train/label",
    #                      csv_path="C:/YHY/CDYHY/CDD/class_dict.csv",
    #                      )

    dataloader_test = DataLoader(data,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=6,
                                )

    for _,(img1,img2,label) in enumerate(dataloader_test):#enumerate会将列表中的每一个元素单独拆分出来，加上索引，以元组的形式返回
        print('img1',img1)
        print(img1.shape)
        print('img2',img2)
        print(img2.shape)
        print('label',label)
        print(label.shape)
        if _ == 0:
            break





























