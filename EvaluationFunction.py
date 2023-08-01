import torch


def Pixel_Accuracy(predict, label):
    """像素精确度"""
    predict     = predict.flatten()
    label       = label.flatten()
    corr        = torch.sum(predict==label)
    tensor_size = predict.size(0)
    PA          = float(corr) / float(tensor_size)
    return PA

def Overall_Accuracy(predict, label):
    """总体确度"""
    predict = predict.flatten()
    label = label.flatten()
    TP = torch.sum(torch.mul(predict, label))
    FP = torch.sum((label == 0) & (predict == 1))
    TN = torch.sum((label == 0) & (predict == 0))
    FN = torch.sum((label == 1) & (predict == 0))
    recall = float(torch.sum(TP+TN)) / (float(torch.sum(TP + TN + FN + FP)) + 1e-6)
    return recall

def Kappa(predict, label):
    predict = predict.flatten()
    label = label.flatten()
    TP = torch.sum(torch.mul(predict, label))
    FP = torch.sum((label == 0) & (predict == 1))
    TN = torch.sum((label == 0) & (predict == 0))
    FN = torch.sum((label == 1) & (predict == 0))
    Po = Overall_Accuracy(predict,label)
    TNFN = TN + FN
    TNFP = TN + FP
    FPTP = FP + TP
    FNTP = FN + TP
    Pe =float(((TNFN*TNFP)+(FPTP*FNTP)))/float((TP+FP+TN+FN)*(TP+FP+TN+FN))
    recall = float(Po-Pe) / (float(1-Pe) + 1e-6)
    return recall

def Recall(predict, label):
    """查全率亦称召回率"""
    predict     = predict.flatten()
    label       = label.flatten()
    TP          = torch.sum((label == 1) & (predict == 1))
    FN          = torch.sum((label==1)&(predict==0))
    recall      = float(torch.sum(TP))/(float(torch.sum(TP+FN))+1e-6)
    return recall

def Precision(predict, label):
    """查准率亦称准确率"""
    predict     = predict.flatten()
    label       = label.flatten()
    TP          = torch.sum((label == 1) & (predict == 1))
    FP          = torch.sum((label==0)&(predict==1))
    Precision   = float(torch.sum(TP)) / (float(torch.sum(TP+FP)) + 1e-6)
    return Precision

def F1(predict, label):
    recall      = Recall(predict,label)
    precision   = Precision(predict,label)
    F1          = 2 * recall * precision / (recall + precision + 1e-6)
    return F1

####################################
# computer Miou
# SR: Segmentation Result-->predict
# GT: Ground Truth-->label
def extract_classes(GT):
    cl = torch.unique(GT) # 调出GT中独立不重复的元素 [0,1]
    n_cl = len(cl)  #  2
    return cl, n_cl

def union_classes(SR, GT):
    eval_cl, _ = extract_classes(SR)
    gt_cl, _   = extract_classes(GT)
    # print('1',eval_cl)
    # print('2',gt_cl)
    # print(torch.cat([eval_cl, gt_cl]))#-->tensor([0, 1, 0, 1], dtype=torch.int32)
    cl = torch.unique(torch.cat([eval_cl, gt_cl]).view(-1))
    # print('cl',cl)
    n_cl = len(cl)
    return cl, n_cl


def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise
    return height, width


def extract_masks(segm, cl, n_cl):#  cl-->tensor([0, 1], dtype=torch.int32) n_cl--> 2 , segm -->[512,512]
    h, w  = segm_size(segm)
    masks = torch.zeros((n_cl, h, w)) #-->tensor([2,512,512])
    for i, c in enumerate(cl):
        # print('i',i)#i 0
        # print('c',c)#c tensor(0, device='cuda:0')
        masks[i, :, :] = segm == c
        # print(masks)
    return masks

def extract_both_masks(SR, GT, cl, n_cl):
    eval_mask = extract_masks(SR, cl, n_cl)
    gt_mask   = extract_masks(GT, cl, n_cl)
    return eval_mask, gt_mask

def mean_IU(SR, GT):
    cl, n_cl = union_classes(SR, GT)
    _, n_cl_gt = extract_classes(GT)
    eval_mask, gt_mask = extract_both_masks(SR, GT, cl, n_cl)

    IU = torch.FloatTensor(list([0]) * n_cl)

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        if (torch.sum(curr_eval_mask) == 0) or (torch.sum(curr_gt_mask) == 0):
            continue

        n_ii = torch.sum((curr_eval_mask == 1) & (curr_gt_mask == 1))
        t_i = torch.sum(curr_gt_mask)
        n_ij = torch.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    miou = torch.sum(IU) / n_cl_gt
    return miou


if __name__ == "__main__":
    import numpy as np
    # SR = torch.IntTensor([1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]).view(5,5)
    # GT = torch.IntTensor([1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]).view(5,5)
    # PA = Pixel_Accuracy(SR, GT)
    # recall = Recall(SR, GT)
    # precision = Precision(SR, GT)
    # f1 = F1(SR, GT)
    # print('PA           code: {:.3f} '.format(PA))
    # print('Recall       code: {:.3f} '.format(recall))
    # print('Precision    code: {:.3f} '.format(precision))
    # print('F1           code: {:.3f} '.format(f1))
    # # from PIL import Image
    # img_1 = torch.from_numpy(np.array(Image.open('ceship.png')) // 255)
    # img_2 = torch.from_numpy(np.array(Image.open('ceshil.png')) // 255)
    # print(img_1)
    # print(img_2.shape)
    # print(mean_IU(img_1, img_2))
    # a,b=extract_classes(SR)
    # print(a,b)
    # c,d=union_classes(SR,GT)
    # print(c,d)
    from LoadData import *
    csv_path         = os.path.join('C:/YHY/CDYHY/dataset', 'class_dict.csv')
    # print(csv_path)
    label_info = get_label_info(csv_path)
    # print(label_info)
    img1 = Image.open('C:/YHY/CDYHY/UNet/ceshi1l.png')
    Image._show(img1)
    img1 = one_hot_it(img1,label_info).astype(np.uint8)
    # print(img1.shape)
    img1 = np.transpose(img1,[2,0,1]).astype(np.uint8)
    # img1 = np.array(img1).astype(np.float32)
    img1 = torch.from_numpy(img1)
    a = reverse_one_hot(img1)


    img2 = Image.open('C:/YHY/CDYHY/UNet/cheshi1p.png')
    Image._show(img2)
    img2 = one_hot_it(img2, label_info).astype(np.uint8)
    # img2 = np.array(img2).astype(np.float32)
    img2 = np.transpose(img2, [2,0,1]).astype(np.uint8)
    img2 = torch.from_numpy(img2)
    b = reverse_one_hot(img2)
    x    = mean_IU(a,b)
    print(x)





