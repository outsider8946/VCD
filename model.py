import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,(3,3),padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,(3,3),padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)


class DownConv(nn.Module):
    def __init__(self,in_c, out_c):
        super().__init__()
        self.conv = DoubleConv(in_c,out_c)
        self.pool =  nn.MaxPool2d((2,2),stride=2)

    def forward(self,x):
        skip_conn = self.conv(x)
        return skip_conn, self.pool(skip_conn)


class UpConv(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.up = nn.ConvTranspose2d(in_c,out_c,(2,2),stride=2)
        self.conv = DoubleConv(in_c,out_c)

    def forward(self,x1,x2):
        x2 = self.up(x2)
        x = torch.cat([x1,x2],dim=1)
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DownConv(1,64)
        self.down2 = DownConv(64,128)
        self.down3 = DownConv(128,256)
        self.down4 = DownConv(256,512)

        self.bottle = DoubleConv(512,1024)

        self.up1 = UpConv(1024,512)
        self.up2 = UpConv(512,256)
        self.up3 = UpConv(256,128)
        self.up4 = UpConv(128,64)

        self.final = nn.Conv2d(64,1,(1,1))

    def forward(self,x):

        s1,x = self.down1(x)
        s2,x = self.down2(x)
        s3,x = self.down3(x)
        s4,x = self.down4(x)

        x = self.bottle(x)

        x = self.up1(s4,x)
        x = self.up2(s3,x)
        x = self.up3(s2,x)
        x = self.up4(s1,x)

        return self.final(x)
