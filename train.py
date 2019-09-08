import torch
import argparse
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from data.dataloader import PascalVocDataLoader
from models.vgg_backbone import SSD_VGG
from models.mobilenetv2_backbone import SSD_MobileNetV2
from models.mobilenetv3_backbone import SSD_MobileNetV3
from models.efficientnet_backbone import SSD_EfficientNet
from layers.multiboxloss import MultiBoxLoss


def train(args):
    if args.dataset == 'pascalvoc':
        dataset = PascalVocDataLoader(args.rootpath, transform=True)
        data_loader = data.DataLoader(dataset, args.batchsize, shuffle=True, collate_fn=dataset.detection_collate)
        num_classes = 21

    datasize = len(dataset) // args.batchsize
    epochs = 50000

    if args.network == 'vgg':
        net = SSD_VGG(num_classes=num_classes)
        print('VGG')

    elif args.network == 'mobilenetv2':
        net = SSD_MobileNetV2(num_classes=num_classes)
        print('MobileNetV2')

    elif args.network == 'mobilenetv3':
        net = SSD_MobileNetV3(num_classes=num_classes)
        print('MobileNetV3')

    elif args.network == 'efficientnet':
        net = SSD_EfficientNet.from_name('efficientnet-b4', num_classes=2, phase='train')
        print('EfficientNet')

    if args.pretrained == True:
        if args.network == 'efficientnet':
            pretrained_model = torch.load('pretrained_weights/efficientnet-b4-e116e8b3.pth')
            net_dict = net.state_dict()

            pretrained_layers = list(pretrained_model.keys())
            net_layers = list(net_dict.keys())

            for i in range(len(pretrained_layers) - 2):
                net_dict[net_layers[i]] = pretrained_model[pretrained_layers[i]]

            net.load_state_dict(net_dict)

    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    criterion = MultiBoxLoss(num_classes=num_classes)

    net.train()

    for i in range(epochs):
        batch_iterator = iter(data_loader)
        loss_l_list = []
        loss_c_list = []
        total_loss_list = []
        for j in range(datasize):
            images, targets = next(batch_iterator)

            images = Variable(images, volatile=True)
            targets = [Variable(target, volatile=True) for target in targets]

            preds = net(images)

            optimizer.zero_grad()
            loss_l, loss_c = criterion(preds, targets)
            loss = loss_l + loss_c
            loss_l_list.append(loss_l)
            loss_c_list.append(loss_c)
            total_loss_list.append(loss)
            loss.backward()
            optimizer.step()

        print('Epoch : %s Regression Loss : %s Classification Loss : %s Total Loss : %s' % (
        i, sum(loss_l_list) / float(len(loss_l_list)), sum(loss_c_list) / float(len(loss_c_list)),
        sum(total_loss_list) / float(len(total_loss_list))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pascalvoc', type=str)
    parser.add_argument('--rootpath', default='/Users/bogireddyteja/Work/datasets/VOCdevkit/VOC2012/', type=str)
    parser.add_argument('--network', default='vgg', type=str)
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--pretrained', default=False, type=bool)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Train Network
    train(args)