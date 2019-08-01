import torch
import torch.nn as nn


class SpecNet(nn.Module):

    def __init__(self, ):
        super(SpecNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv0 = torch.nn.Conv2d(1, 64, 5, stride=1, padding=2, bias=True)
        self.bn0 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=(16, 1), padding=8, stride=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_3 = nn.Conv2d(64, 64, kernel_size=(16, 1), padding=0, stride=1)
        self.bn1_3 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1, stride=(1, 2))
        self.bn1_2 = nn.BatchNorm2d(128)

        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=(16, 1), padding=0, stride=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_3 = nn.Conv2d(128, 128, kernel_size=(16, 1), padding=0, stride=1)
        self.bn2_3 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1, stride=(1, 2))
        self.bn2_2 = nn.BatchNorm2d(256)

        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=(16, 1), padding=0, stride=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(5, 1), padding=0, stride=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1, stride=(1, 2))
        self.bn3_2 = nn.BatchNorm2d(512)

        self.downsample1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 13, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # self.new_fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)

        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        # x0 = x #64*87
        # x1 = self.conv1_1(x) #64*87
        # x1 = self.bn1_1(x1)
        # x1 = self.relu(x1)
        # x1 = self.conv1_2(x1)
        # x1 = self.bn1_2(x1)
        # x0 = self.downsample1(x0)
        # x = x1 + x0
        # x = self.relu(x)

        return x
