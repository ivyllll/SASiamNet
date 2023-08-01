import numpy as np
import torch
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from EvaluationFunction import *


def val(args,lr,model,dataloader_val,epoch,loss_train_mean,writer):
    print('Start Validation')
    start = time.time()
    with torch.no_grad():
        model.eval()

        PA_all = []
        recall_all = []
        precision_all = []
        f1_all = []
        miou_all = []
        oa_all = []
        kappa_all = []

        for i, (img1, img2, label)in enumerate(dataloader_val):
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
            predict = model(img1,img2)# [1, 2, 512, 512] ==> [2, 512, 512]
            predict = predict.squeeze()
            # print('0', predict.max())
            predict = reverse_one_hot(predict)
            # print('3', predict.max())
            label = label.squeeze()
            label = reverse_one_hot(label)

            pa        = Pixel_Accuracy(predict,label)
            recall    = Recall(predict,label)
            precision = Precision(predict,label)
            f1        = F1(predict,label)
            miou      = mean_IU(predict,label)
            oa        = Overall_Accuracy(predict,label)
            kappa     = Kappa(predict,label)

            PA_all.append(pa)
            recall_all.append(recall)
            precision_all.append(precision)
            f1_all.append(f1)
            miou_all.append(miou)
            oa_all.append(oa)
            kappa_all.append(kappa)

        # 计算各指标的平均值
        pa = np.mean(PA_all)
        recall = np.mean(recall_all)
        precision = np.mean(precision_all)
        f1 = np.mean(f1_all)
        miou = np.mean(miou_all)
        oa = np.mean(oa_all)
        kappa = np.mean(kappa_all)

        str_ = ("%15.5g;" * 10) % (epoch + 1, lr ,loss_train_mean, pa, recall, precision, f1, miou, oa, kappa)
        with open(f'{args.save_model_path}/{args.result}.txt', 'a') as f:
            f.write(str_ + '\n')

        # print correction evaluation index
        print('PA:          {:}'.format(pa))
        print('Recall:      {:}'.format(recall))
        print('Precision:   {:}'.format(precision))
        print('F1:          {:}'.format(f1))
        print('Miou:        {:}'.format(miou))
        print('OA:          {:}'.format(oa))
        print('Kappa:       {:}'.format(kappa))

        print('Time:        {:}s'.format(time.time() - start))

        # write to the log
        writer.add_scalar('{}_Pa'.format('val'), pa, epoch + 1)
        writer.add_scalar('{}_Recall'.format('val'), recall, epoch + 1)
        writer.add_scalar('{}_Precision'.format('val'), precision, epoch + 1)
        writer.add_scalar('{}_F1'.format('val'), f1, epoch + 1)
        writer.add_scalar('{}_Miou'.format('val'), miou, epoch + 1)
        writer.add_scalar('{}_OA'.format('val'), oa, epoch + 1)
        writer.add_scalar('{}_Kappa'.format('val'),kappa, epoch + 1)
        # return main evaluation index
        return miou


def train(args,model,optimizer,dataloader_train,dataloder_val,lr_scheduler):
    print("Start Training...")

    s = ("%15s;" * 10) % ("epoch", "lr", "loss", "PA", "Recall", "Precision", "F1", "Miou","OA","Kappa")
    with open(f'{args.save_model_path}/{args.result}.txt', 'a') as file:
        file.write(args.result + '\n')
        file.write(s + '\n')

    miou_max = args.miou_max
    writer = SummaryWriter(logdir=f'{args.log_path}')

    for epoch in range(args.num_epochs):# epoch
        model.train()
        lr_scheduler.step()#学习率调整
        lr = optimizer.param_groups[0]['lr']
        loss_record = []
        #进度条
        tq=tqdm(total=len(dataloader_train)*args.batch_size)
        tq.set_description('epoch %d ,lr %f' % (epoch+1,lr))

        for i ,(img1,img2,label) in enumerate(dataloader_train):
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
            # 开始训练时防止数据震荡
            if args.warmup == 1 and epoch == 0:
                lr = args.lr / (len(dataloader_train) - i)
                tq.set_description('epoch %d, lr %f' % (epoch+1 , lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            output = model(img1, img2) #img1:[2,3,256,256] img2:[2,3,256,256] label:[2,2,256,256]
            # print(type(output)) # <class 'tuple'>
            # print(output.shape) # [4,2,256,256]
            loss = torch.nn.BCEWithLogitsLoss()(output, label) #MFGAN返回的是tuple,除非删掉逗号后的两个返回值就要改成output[0]


            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())
        tq.close()

        loss_train_mean = np.mean(loss_record)
        print('Loss for train :{:.6f}'.format(loss_train_mean))
        writer.add_scalar('{}_loss'.format('train'),loss_train_mean,epoch+1)

        if epoch % args.checkpoint_step == 0 and epoch != 0:
            save_path = f'{args.save_model_path}/{args.model_name}/' + 'epoch_{:}.pth'.format(epoch)
            torch.save(model.state_dict(), save_path)

        if epoch % args.validation_step == 0:
            miou = val(args,lr, model, dataloder_val, epoch, loss_train_mean, writer)

            if miou > miou_max:
                save_path = f'{args.save_model_path}/{args.model_name}/' + 'max_epoch_{:}'.format(epoch+1) + '_miou_{:.6f}.pth'.format(miou)
                torch.save(model.state_dict(), save_path)
                miou_max = miou

    writer.close()
    save_path = f'{args.save_model_path}' + 'last.pth'
    torch.save(model.state_dict(), save_path)



























