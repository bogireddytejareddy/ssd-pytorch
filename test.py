from matplotlib import pyplot as plt
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from ssd_vgg import SSD_VGG
from ssd_mobilenetv2 import SSD_MobileNetV2
from ssd_mobilenetv3 import SSD_MobileNetV3
import argparse

parser = argparse.ArgumentParser(description='Test Params')
parser.add_argument('--weights')
parser.add_argument('--imagepath')
parser.add_argument('--network')
args = parser.parse_args()

means = (127, 127, 127)

if args.network == 'vgg':
   net = SSD_VGG(phase='test', num_classes=2)
   net.load_state_dict(torch.load(args.weights, map_location=lambda storage, loc: storage))
   net.eval()
elif args.network == 'mobilenetv2':
   net = SSD_MobileNetV2(phase='test', num_classes=2)
   net.load_state_dict(torch.load(args.weights, map_location=lambda storage, loc: storage))
   net.eval()
elif args.network == 'mobilenetv3':
   net = SSD_MobileNetV3(phase='test', num_classes=2)
   net.load_state_dict(torch.load(args.weights, map_location=lambda storage, loc: storage))
   net.eval()

test_images = os.listdir(args.imagepath)

font = cv2.FONT_HERSHEY_SIMPLEX

for image_name in test_images:
    org_img = cv2.imread(args.imagepath + image_name)
    image = cv2.imread(args.imagepath + image_name)
    height, width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))
    image = np.array(image, dtype=np.float32)
    means = np.array(means, dtype=np.float32)
    image -= means
    image = image[:, :, (2, 1, 0)]
    image = torch.from_numpy(image).float().permute(2, 0, 1)
    image = Variable(image.unsqueeze(0))
    detections = net(image).data
    for i in range(1, detections.size(1)):
           dets = detections[0, i, :]
           mask = dets[:, 0].gt(0.65).expand(5, dets.size(0)).t()
           dets = torch.masked_select(dets, mask).view(-1, 5)

           boundingboxes = dets[:, 1:]
           boundingboxes[:, 0] *= width
           boundingboxes[:, 1] *= height
           boundingboxes[:, 2] *= width
           boundingboxes[:, 3] *= height

           scores = dets[:, 0].cpu().numpy()
           for i in range(scores.shape[0]):
               cv2.rectangle(org_img, (int(boundingboxes[i][0]),int(boundingboxes[i][1])),(int(boundingboxes[i][2]),int(boundingboxes[i][3])),(0,0,255),3)
               cv2.rectangle(org_img, (int(boundingboxes[i][0]), int(boundingboxes[i][3]) - 35), (int(boundingboxes[i][2]), int(boundingboxes[i][3])), (0, 0, 255), cv2.FILLED)
               cv2.putText(org_img, 'fish : ' + str("{0:.2f}".format(scores[i])), (int(boundingboxes[i][0]) + 6, int(boundingboxes[i][3]) - 6), font, 1.4, (255, 255, 255), 1)

    cv2.imwrite('./results/' + image_name, org_img)
