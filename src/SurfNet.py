import torch
import torch.nn as nn
import torchvision
import time
import logging
import os
import argparse
from sklearn import utils
from sklearn import metrics
import numpy as np
from PIL import Image
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

torch.manual_seed(1)  # 设置随机种子；可复现性


# torchvision.datasets.ImageFolder:也可以实现dataset数据集的构建
class Mydata(Dataset):
    def __init__(self, root_dir, train=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        file_list = os.listdir(self.root_dir)
        self.target = []
        self.data = []
        for index, file in enumerate(file_list):
            file_list_img = os.listdir(self.root_dir + '/' + file)
            for img_name in file_list_img:
                image_path = self.root_dir + '/' + file + img_name
                # image = Image.open(self.root_dir + '/' + file + img_name).convert('RGB')
                self.target.append(index)
                self.data.append(image_path)

    def __getitem__(self, index):  # pytorch中的图像必须是tensor

        img_path, target = self.data[index], self.target[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if isinstance(index, slice):

            start = index.start
            stop = index.stop
            if start is None:
                start = 0
            imgs, targets = [], []
            for x in range(stop):
                if x >= start:
                    img_path, target = self.data[x], self.target[x]
                    img = Image.open(img_path).convert('RGB')

                    if self.transform is not None:
                        img = self.transform(img)

                    if self.target_transform is not None:
                        target = self.target_transform(target)
                    imgs.append(img)
                    targets.append(target)
            return imgs, targets
        else:
            return img, target

    def __len__(self):  # 定义这个函数，可谓该类的实例调用len()方法
        return len(self.data)


class SurfNet(nn.Module):
    def __init__(self):
        super(SurfNet, self).__init__()  # 首先继承父类的属性和方法

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=6
            ),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1
            ),
            nn.PReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=5,
                stride=2,
                padding=6
            ),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                stride=1
            ),
            nn.PReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=1024,
                kernel_size=5,
                stride=2,
                padding=6
            ),
            nn.PReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=1,
                stride=1
            ),
            nn.PReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=1,
                stride=1
            ),
            nn.PReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=1,
                stride=1
            ),
            nn.PReLU()
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=1,
                stride=1
            ),
            nn.PReLU(),
        )
        self.out1 = nn.Linear(1024 * 11 * 11, 1024)
        self.out2 = nn.Linear(1024, 7)

    def forward(self, x):
        x = self.conv1(x)
        identity = x

        x = self.conv2(x)
        x += identity

        x = self.conv3(x)
        identity = x

        x = self.conv4(x)
        x += identity

        x = self.conv5(x)
        identity = x

        x = self.conv6(x)
        identity6 = x
        x += identity

        x = self.conv7(x)
        identity7 = x
        x += identity6

        x = self.conv8(x)
        identity8 = x
        x += identity7

        x = self.conv9(x)
        x += identity8

        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.out1(x)
        x = self.out2(x)

        return x


def train(args):
    EPOCH = args.epoch
    LR = args.lr
    BS = args.batch_size

    TRANSFORM_IMG = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.ImageFolder(root=args.train_root, transform=TRANSFORM_IMG)
    test_data = torchvision.datasets.ImageFolder(root=args.test_root, transform=TRANSFORM_IMG)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BS, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=200, shuffle=True)

    for step, (x_test1, y_test1) in enumerate(test_loader):
        if step <= 1:
            x_test, y_test = x_test1.cuda(), y_test1.cuda()


    # set log info
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=os.path.join(args.save_model_path, now + '.log'),
                        level=logging.DEBUG, format=LOG_FORMAT)

    surfnet = SurfNet().cuda()
    if args.load_from:
        surfnet.load_state_dict(torch.load(args.load_from))
    optimizer = torch.optim.RMSprop(surfnet.parameters(), lr=LR)  # 优化器为RMSprop
    loss_func = nn.CrossEntropyLoss()                          # 损失函数为负对数似然损失negative log likelihood loss

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            pred_y = surfnet(batch_x.cuda())
            loss = loss_func(pred_y, batch_y.cuda()).cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.print_every == 0 or step == len(train_loader):
                y_test = y_test.cpu()
                test_output = surfnet(x_test).cpu()
                pred_y = torch.max(test_output, 1)[1].cpu().numpy()    # torch.max(test_output, 1)返回两组数据，一个是每行的最大值，另一组为每行最大值的index
                accuracy = float((pred_y == y_test.cpu().numpy()).astype(int).sum()) / float(y_test.size(0))
                sw = utils.class_weight.compute_sample_weight(class_weight='balanced', y=y_test.cpu())
                precision = metrics.precision_score(y_test, pred_y, labels=np.unique(pred_y), pos_label=1,
                                                    average='weighted',
                                                    sample_weight=sw)
                recall = metrics.recall_score(y_test, pred_y, labels=np.unique(pred_y), pos_label=1, average='weighted',
                                              sample_weight=sw)
                F1_score = metrics.f1_score(y_test, pred_y, labels=np.unique(pred_y), pos_label=1, average='weighted',
                                            sample_weight=sw)
                confusion_matrix = metrics.multilabel_confusion_matrix(y_test, pred_y, labels=np.unique(pred_y),
                                                                       sample_weight=sw)
                specificity = 0
                G_mean = 0
                for row in confusion_matrix:
                    specificity += row[0][0] / (row[0][0] + row[0][1])
                    G_mean += np.sqrt((row[0][0] * row[1][1]) / ((row[1][1] + row[1][0]) * (row[0][0] + row[0][1])))
                specificity /= 7
                G_mean /= 7
                info = ('Epoch: ', epoch, '|Iteration:', step, '| train loss: %.4f' % loss.data.cpu().numpy(),
                        '| test accuracy: %.4f' % accuracy,
                        '| test precision: %.4f' % precision,
                        '| test recall: %.4f' % recall,
                        '| test F1_score: %.4f' % F1_score,
                        '| test Specificity: %.4f' % specificity,
                        '| test G-mean: %.4f' % G_mean)
                logging.info(info)
                print(info)

    # save model
    torch.save(surfnet.state_dict(), os.path.join(args.save_model_path, now + f'epoch_{epoch}.pkl'))

    # 完全结束训练之后，打印前十个测试结果和真实结果进行对比
    test_output = surfnet(x_test).cpu()
    pred_y = torch.max(test_output, 1)[1].numpy()
    print(pred_y, 'prediction number')
    print(y_test.numpy(), 'real number')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.00011)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--train_root", type=str, default="/data2/TDL/paper_fabric/fabric_datasets/fabric_classic_dataset_280/train/")
    parser.add_argument("--test_root", type=str, default="/data2/TDL/paper_fabric/fabric_datasets/fabric_classic_dataset_280/val/")
    parser.add_argument("--save_model_path", type=str, default="/data2/TDL/paper_fabric/workdir/4_30_short_eassy/surfnet/CE_RMS")
    parser.add_argument("--load_from", type=str, default=None)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    train(args)
