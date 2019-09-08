import os
import torch
import numpy as np
import cv2
import argparse
from models.mobilenetv2_backbone import SSD_MobileNetV2


VOC_LABELS = {0 : "background", 1 : "aeroplane", 2 : "bicycle", 3 : "bird", 4 : "boat", 5 : "bottle",
              6 : "bus", 7 : "car", 8 : "cat", 9 : "chair", 10 : "cow", 11 : "diningtable", 12 : "dog",
              13 : "horse", 14 : "motorbike", 15 : "person", 16 : "pottedplant", 17 : "sheep", 18 : "sofa",
              19 : "train", 20 : "tvmonitor"}


parser = argparse.ArgumentParser(description='Test Params')
parser.add_argument('--weights')
parser.add_argument('--imagepath')
parser.add_argument('--network')
args = parser.parse_args()

means = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

if args.network == 'mobilenetv2':
   net = SSD_MobileNetV2(phase='test', num_classes=21)
   net.load_state_dict(torch.load(args.weights, map_location=lambda storage, loc: storage).state_dict())
   net.eval()

test_images = os.listdir(args.imagepath)
font = cv2.FONT_HERSHEY_SIMPLEX

for image_name in test_images:
    org_img = cv2.imread(args.imagepath + image_name)
    image = cv2.imread(args.imagepath + image_name)
    height, width, _ = image.shape
    image = cv2.resize(image, (300, 300))
    image = np.array(image, dtype=np.float32)/255.
    means = np.array(means, dtype=np.float32)
    image -= means
    image /= std
    image = image[:, :, (2, 1, 0)]
    image = torch.from_numpy(image).float().permute(2, 0, 1)
    image = torch.autograd.Variable(image.unsqueeze(0))
    detections = net(image).data
    print(detections.shape)
    for i in range(1, detections.size(1)):
        dets = detections[0, i, :]
        mask = dets[:, 0].gt(0.5).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)

        boundingboxes = dets[:, 1:]
        boundingboxes[:, 0] *= width
        boundingboxes[:, 1] *= height
        boundingboxes[:, 2] *= width
        boundingboxes[:, 3] *= height

        scores = dets[:, 0].cpu().numpy()
        for j in range(scores.shape[0]):
            cv2.rectangle(org_img, (int(boundingboxes[j][0]),int(boundingboxes[j][1])),(int(boundingboxes[j][2]),int(boundingboxes[j][3])),(0,0,255),3)
            cv2.rectangle(org_img, (int(boundingboxes[j][0]), int(boundingboxes[j][3]) - 15),
                          (int(boundingboxes[j][2]), int(boundingboxes[j][3])), (0, 0, 255), cv2.FILLED)
            cv2.putText(org_img, 'class : ' + VOC_LABELS[i],
                        (int(boundingboxes[j][0]) + 6, int(boundingboxes[j][3]) - 6), font, 0.4, (255, 255, 255), 1)

    cv2.imwrite('./results/' + image_name, org_img)
