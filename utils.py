import pandas as pd
import numpy as np
import torch


def get_label_info(csv_path="D:\.ipynb_checkpoints\CD\DATA\class_dict.csv"):
    """
    变化区域的rgb值置为255（白色），
    不变区域的rgb值置为 0（黑色)
    """
    data = pd.read_csv(csv_path)
    label= {}
    for _,row in data.iterrows():# 这里的iterrows()返回值为元组(_,row)
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r),int(g),int(b)]
    return label


def one_hot_it(label,label_info=get_label_info()):
    """返回分割图像--->[H,W,num_classes]"""
    senantic_map = []
    for info in label_info:
        color = label_info[info]
        equality = np.equal(label,color)
        # print('eq',equality)
        class_map = np.all(equality,axis=-1)
        # print('cl',class_map)
        senantic_map.append(class_map)
    senantic_map=np.stack(senantic_map,axis=-1)
    return senantic_map


def reverse_one_hot(image):

    image = image.permute(1,2,0)  # [2, 512, 512] ==> [512, 512, 2] 将H,W,C变为W,C,H
    # print('1',torch.__version__, image.max())
    x = torch.argmax(image, dim=-1)  # [512, 512, 2] ==> [512, 512]
    # print('2', x.max())
    return x


if __name__ == '__main__':
    a=get_label_info()
    print(a)
    b=[[[ 0, 0 , 1],
         [ 0,0,0],
         [ 6 , 7 , 8]],

        [[ 255 , 255 , 255],
         [34, 13, 14],
         [15 ,16, 17]],

         [[0 ,0, 20],
          [21 ,22 ,23],
          [24 ,25 ,26]]]
    # b=np.array(b)
    # b=torch.from_numpy(b)
    # c=torch.argmax(b,dim=0)
    # print(c)
    # d=reverse_one_hot(b)
    # print(d)


    c=np.array(b)

    print(c)
    d=one_hot_it(c,get_label_info())
    print(d)
    e=np.transpose(d,[2,0,1])
    print(e)







