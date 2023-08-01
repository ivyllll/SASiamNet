import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d


class FC_Siam_diff(nn.Module):
    def __init__(self,in_ch=3,classes=2,bias=False):
        super(FC_Siam_diff, self).__init__()

        self.in_ch = in_ch

        self.conv11 = nn.Conv2d(in_ch, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.dro11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.dro12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn21 = nn.BatchNorm2d(32)
        self.dro21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn22 = nn.BatchNorm2d(32)
        self.dro22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn31 = nn.BatchNorm2d(64)
        self.dro31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn32 = nn.BatchNorm2d(64)
        self.dro32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn33 = nn.BatchNorm2d(64)
        self.dro33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn41 = nn.BatchNorm2d(128)
        self.dro41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn42 = nn.BatchNorm2d(128)
        self.dro42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn43 = nn.BatchNorm2d(128)
        self.dro43 = nn.Dropout2d(p=0.2)

#######################################################

        self.up4 = nn.ConvTranspose2d(128,128,kernel_size=3,stride=2,padding=1,output_padding=1)

        self.conv43d = nn.ConvTranspose2d(256,128,kernel_size=3,padding=1)
        self.bn43d = nn.BatchNorm2d(128)
        self.dro43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.dro42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.dro41d = nn.Dropout2d(p=0.2)

        self.up3 = nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1,output_padding=1)

        self.conv33d = nn.ConvTranspose2d(128,64,kernel_size=3,padding=1)
        self.bn33d = nn.BatchNorm2d(64)
        self.dro33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.dro32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.dro31d = nn.Dropout2d(p=0.2)

        self.up2 = nn.ConvTranspose2d(32,32,kernel_size=3,stride=2,padding=1,output_padding=1)

        self.conv22d = nn.ConvTranspose2d(64,32,kernel_size=3,padding=1)
        self.bn22d = nn.BatchNorm2d(32)
        self.dro22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.dro21d = nn.Dropout2d(p=0.2)

        self.up1 = nn.ConvTranspose2d(16,16,kernel_size=3,stride=2,padding=1,output_padding=1)

        self.conv12d = nn.ConvTranspose2d(32,16,kernel_size=3,padding=1)
        self.bn12d = nn.BatchNorm2d(16)
        self.dro12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(16,classes,kernel_size=3,padding=1)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self,x1,x2):

        # Stage 1
        x11 = self.dro11(F.relu(self.bn11(self.conv11(x1))))
        x12_1 = self.dro12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1,kernel_size=2,stride=2)

        x21 = self.dro21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.dro22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.dro31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.dro32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.dro33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.dro41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.dro42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.dro43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)

############################################################

        # Stage 1
        x11 = self.dro11(F.relu(self.bn11(self.conv11(x2))))
        x12_2 = self.dro12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.dro21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.dro22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.dro31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.dro32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.dro33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.dro41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.dro42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.dro43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)

###########################################################

        x4d = self.up4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), 1)
        # print(x4d.shape)
        x43d = self.dro43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.dro42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.dro41d(F.relu(self.bn41d(self.conv41d(x42d))))

        x3d = self.up3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), 1)
        x33d = self.dro33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.dro32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.dro31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.up2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), 1)
        x22d = self.dro22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.dro21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.up1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), 1)
        x12d = self.dro12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)


        return self.logsoftmax(x11d)


if __name__ == '__main__':
    x1 = torch.rand(4,3,512,512)
    x2 = torch.rand(4,3,512,512)
    model = FC_Siam_diff()
    a = model(x1,x2)
    print(a.shape)















