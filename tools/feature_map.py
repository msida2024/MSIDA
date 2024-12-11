# import torch
# from torchvision import models, transforms
# import torchvision.models as models
# from PIL import Image
# import matplotlib.pyplot as plt
# import ssl
# import numpy as np
#
# def load_image(image_path):
#     im = Image.open(image_path)
#     im = im.resize((224, 224))
#     im = transforms.ToTensor()(im)
#     im = im.unsqueeze(0)
#     return im
#
#
# def preprocess_image(image):
#     preprocess = transforms.Compose([
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     input_tensor = preprocess(image)
#     return input_tensor
#
#
# if __name__ == '__main__':
#     ssl._create_default_https_context = ssl._create_unverified_context
#
#     # pthfile = r'/ssd/lh/DeepSolo-self/weights/self/totaltext-EMA-semantic/k300-mlt.pth'
#     model = models.resnet50(pretrained=True)
#     model.eval()
#     print(model)
#
#     # 加载图像并进行预处理
#     image_path = "/ssd/lh/DeepSolo-self/datasets/totaltext/test_images/0000000.jpg"
#     img = load_image(image_path)
#     input_tensor = preprocess_image(img)
#     print(img.shape)
#     input_tensor = input_tensor.permute(0, 3, 2, 1)
#     plt.imshow(input_tensor[0])
#     plt.show()
#
#     input_tensor = input_tensor.permute(0, 3, 2, 1)
#     with torch.no_grad():
#         features = model(input_tensor)
#
#     # 可视化特征图
#     features = features.squeeze(1)
#     features = features.detach().numpy()
#     print(features.shape)
#     plt.imshow(features, cmap='gray')
#     plt.show()
import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

imgname = '/ssd/lh/DeepSolo-self/datasets/totaltext/test_images/0000093.jpg'
savepath='/ssd/lh/DeepSolo-self/output/original'
if not os.path.isdir(savepath):
    os.makedirs(savepath)


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor  # 分组因子
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)  # softmax操作
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 1×1平均池化层
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # X平均池化层 h=1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Y平均池化层 w=1
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 分组操作
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1,
                                 padding=0)  # 1×1卷积分支
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                 padding=1)  # 3×3卷积分支

        self.conv1x1_up = nn.Conv2d(channels // self.groups, 2 * (channels // self.groups), kernel_size=1, stride=1,
                                    padding=0)
        self.conv3x3_mid = nn.Conv2d(2 * (channels // self.groups), 2 * (channels // self.groups), kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.conv1x1_down = nn.Conv2d(2 * (channels // self.groups), channels // self.groups, kernel_size=1, stride=1,
                                      padding=0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w

        x_h = self.pool_h(group_x)  # 得到平均池化之后的h
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 得到平均池化之后的w
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 先拼接，然后送入1×1卷积
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)  # 3×3卷积分支
        # x2 = self.conv1x1_down(self.conv3x3_mid(self.conv1x1_up(group_x)))  # 3×3卷积分支

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


def draw_features(width,height,x,savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
        img=img.astype(np.uint8)  #转成unit8
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
        img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))


class ft_net(nn.Module):

    def __init__(self):
        super(ft_net, self).__init__()
        model_ft = models.resnet50()
        print(model_ft)
        state_dict = torch.load('/ssd/lh/DeepSolo-self/weights/self/totaltext-EMA-semantic/k300-mlt.pth').pop("model")
        # print(state_dict)
        model_ft.load_state_dict(state_dict, strict=False)
        self.model = model_ft
        self.EMA1 = EMA(256)
        self.EMA2 = EMA(512)
        self.EMA3 = EMA(1024)
        self.EMA4 = EMA(2048)

    def forward(self, x):
        if True: # draw features or not
            x = self.model.conv1(x)
            # draw_features(8, 8, x.cpu().numpy(),"{}/f1_conv1.png".format(savepath))

            x = self.model.bn1(x)
            # draw_features(8, 8, x.cpu().numpy(),"{}/f2_bn1.png".format(savepath))

            x = self.model.relu(x)
            # draw_features(8, 8, x.cpu().numpy(), "{}/f3_relu.png".format(savepath))

            x = self.model.maxpool(x)
            # draw_features(8, 8, x.cpu().numpy(), "{}/f4_maxpool.png".format(savepath))

            x = self.model.layer1(x)
            x = self.EMA1(x)
            draw_features(16, 16, x.cpu().numpy(), "{}/f5_layer1.png".format(savepath))

            x = self.model.layer2(x)
            # x = self.EMA2(x)
            # draw_features(16, 32, x.cpu().numpy(), "{}/f6_layer2.png".format(savepath))

            x = self.model.layer3(x)
            # x = self.EMA3(x)
            # draw_features(32, 32, x.cpu().numpy(), "{}/f7_layer3.png".format(savepath))

            x = self.model.layer4(x)
            # x = self.EMA4(x)
            # draw_features(32, 32, x.cpu().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(savepath))
            # draw_features(32, 32, x.cpu().numpy()[:, 1024:2048, :, :], "{}/f8_layer4_2.png".format(savepath))

            x = self.model.avgpool(x)
            plt.plot(np.linspace(1, 2048, 2048), x.cpu().numpy()[0, :, 0, 0])
            plt.savefig("{}/f9_avgpool.png".format(savepath))
            plt.clf()
            plt.close()

            x = x.view(x.size(0), -1)
            x = self.model.fc(x)
            plt.plot(np.linspace(1, 1000, 1000), x.cpu().numpy()[0, :])
            plt.savefig("{}/f10_fc.png".format(savepath))
            plt.clf()
            plt.close()
        else :
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.model.fc(x)

        return x


model = ft_net().cuda()

# pretrained_dict = resnet50.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)
model.eval()
img = cv2.imread(imgname)
img = cv2.resize(img, (288, 288))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img = transform(img).cuda()
img = img.unsqueeze(0)

with torch.no_grad():
    start = time.time()
    out = model(img)
    # print("total time:{}".format(time.time()-start))
    result = out.cpu().numpy()
    # ind=np.argmax(out.cpu().numpy())
    ind = np.argsort(result, axis=1)
    # for i in range(5):
    #     print("predict:top {} = cls {} : score {}".format(i+1,ind[0,1000-i-1],result[0,1000-i-1]))
    # print("done")
