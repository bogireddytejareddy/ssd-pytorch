import torch
import argparse
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from data.pascalvoc import PascalVocDataLoader
from data.fishdata_v2 import FishDataLoader
from ssd_vgg import SSD_VGG
from ssd_mobilenetv2 import SSD_MobileNetV2
from ssd_mobilenetv3 import SSD_MobileNetV3
from layers.multiboxloss import MultiBoxLoss
from utils.augmentation import SSDAugumentation

def detection_collate(batch):
    targets = []
    imgs = []
    
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    
    return torch.stack(imgs, 0), targets

def train(args):
    if args.dataset == 'pascalvoc':
        dataset = PascalVocDataLoader(args.rootpath, transform=SSDAugumentation())
        data_loader = data.DataLoader(dataset, args.batchsize, shuffle=True, collate_fn=detection_collate)
        num_classes = 21

    elif args.dataset == 'fishdata':
        dataset = FishDataLoader(args.rootpath, transform=SSDAugumentation())
        data_loader = data.DataLoader(dataset, args.batchsize, shuffle=True, collate_fn=detection_collate)
        num_classes = 2

    datasize = len(dataset) // args.batchsize
    epochs = 50000

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    if args.network == 'vgg':
        net = SSD_VGG(num_classes=num_classes)
        print ('SSD')

    elif args.network == 'mobilenetv2':
        net = SSD_MobileNetV2(num_classes=num_classes)
        print ('MobileNetV2')

    elif args.network == 'mobilenetv3':
        net = SSD_MobileNetV3(num_classes=num_classes)
        print ('MobileNetV3')

    if args.pretrained == True:
       if args.network == 'mobilenetv2':
          pretrained_model = torch.load('pretrained_weights/mb2-imagenet-71_8.pth')
          net_dict = net.state_dict()
          pretrained_layers = list(pretrained_model.keys())
          net_layers = list(net_dict.keys())
          j, k = 0, 0
          for i in range(234):
              if 'num_batches_tracked' in net_layers[k]:
                 k = k + 1
                 continue
              else:
                 net_dict[net_layers[k]] = pretrained_model[pretrained_layers[j]]
                 k = k + 1
                 j = j + 1
          net.load_state_dict(net_dict)

    if args.network == 'mobilenetv2':
       optimizer = optim.SGD([
                {'params': net.conv_layer1.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer1.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer2.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer3.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer4.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer5.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer6.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer7.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer8.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer9.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer10.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer11.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer12.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer13.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer14.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer15.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer16.parameters(), 'lr': 1e-3},
                {'params': net.invertedresidual_layer17.parameters(), 'lr': 1e-3},
                {'params': net.conv_layer2.parameters(), 'lr': 1e-3}
            ], lr=1e-4, momentum=0.9, weight_decay=5e-4)
       criterion = MultiBoxLoss(num_classes=num_classes)
    else:
       optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
       criterion = MultiBoxLoss(num_classes=num_classes)

    net = net.cuda()
    net.train()

    for i in range(epochs):
        batch_iterator = iter(data_loader)
        loss_l_list = []
        loss_c_list = []
        total_loss_list = []
        flag = 0
        for j in range(datasize):
            images, targets = next(batch_iterator)

            images = Variable(images.cuda())
            targets = [Variable(target.cuda(), volatile=True) for target in targets]
    
            preds = net(images)

            optimizer.zero_grad()
            loss_l, loss_c = criterion(preds, targets)
            loss = loss_l + loss_c
            loss_l_list.append(loss_l)
            loss_c_list.append(loss_c)
            total_loss_list.append(loss)
            loss.backward()
            optimizer.step()

        if i % 10 == 0:
             torch.save(net.state_dict(), 'weights/SSD300_' + args.network + '_' + args.dataset + '_' + str(i) + '_' + str(float(loss)) + '.pth')

        print ('Epoch : %s Regression Loss : %s Classification Loss : %s Total Loss : %s' %(i, sum(loss_l_list)/float(len(loss_l_list)), sum(loss_c_list)/float(len(loss_c_list)), sum(total_loss_list)/float(len(total_loss_list))))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pascalvoc', type=str)
    parser.add_argument('--rootpath', default='../Data/VOCdevkit/VOC2012/', type=str)
    parser.add_argument('--network', default='vgg', type=str)
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--pretrained', default=True, type=bool)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    train(args)
