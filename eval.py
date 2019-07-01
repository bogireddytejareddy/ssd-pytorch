import os
import cv2
import json
import time
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from ssd_vgg import SSD_VGG
from ssd_mobilenetv2 import SSD_MobileNetV2
from ssd_mobilenetv3 import SSD_MobileNetV3
from torch.autograd import Variable
from data.fishdata import FishDataLoader

def evaluate(network, dataset, num_classes):
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_boxes = [[[] for _ in range(len(dataset))] for _ in range(num_classes)]

    for img_num in tqdm(range(len(dataset))):
       img, target, width, height = dataset.pull_item(img_num)
       img = Variable(img.unsqueeze(0))

       detections = network(img).data
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

           cls_dets = np.hstack((boundingboxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)
           all_boxes[i][img_num] = cls_dets

    for label_idx, label in enumerate(labels):
        filename = os.path.join('results', '%s.txt' %(label))
        with open(filename, 'wt') as f:
            for idx, image in enumerate(dataset.images):
                dets = all_boxes[label_idx+1][idx]
                if dets == []:
                    continue
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(image[:-4], dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))

    for threshold in thresholds:
        for label_idx, label in enumerate(labels):
            filename = os.path.join('results', '%s.txt' %(label))
            average_precision = ap_calculate(filename, dataset.images, threshold, label_idx)
            print ('Average Precision:%s @ Threshold:%s' %(average_precision, threshold))

def ap_calculate(detection_file, image_names, threshold, label_idx):
    ground_truth_meta = {}
    for idx, image in enumerate(image_names):
        annotation_file = os.path.join('../Data/FishData/annotations/', image[:-4] + '.json')
        ground_truth_meta[image[:-4]] = parse_json(annotation_file)

    ground_truth = {}
    num_true_detections = 0

    for image in image_names:
        R = [meta for meta in ground_truth_meta[image[:-4]] if int(meta['label']) == label_idx+1]
        bbox = np.asarray([meta['bbox'] for meta in R])
        det = [False] * len(R)
        num_true_detections = num_true_detections + len(R)
        ground_truth[image[:-4]] = {'bbox' : bbox, 'det' : det}

    with open(detection_file, 'r') as f:
        lines = f.readlines()


    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BoundingBoxes = np.array([[float(z) for z in x[2:]] for x in splitlines])
        
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
        
    BoundingBoxes = BoundingBoxes[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
        
    num_detections = len(image_ids)
    tp = np.zeros(num_detections)
    fp = np.zeros(num_detections)

    for detection in range(num_detections):
        imagepath = '../Data/FishData/images/' + image_ids[detection] + '.jpg'
        img = cv2.imread(imagepath)
        R = ground_truth[image_ids[detection]]
        boundingbox = BoundingBoxes[detection, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            ixmin = np.maximum(BBGT[:, 0], boundingbox[0])
            iymin = np.maximum(BBGT[:, 1], boundingbox[1])
            ixmax = np.minimum(BBGT[:, 2], boundingbox[2])
            iymax = np.minimum(BBGT[:, 3], boundingbox[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            intersection = iw * ih
            union = ((boundingbox[2] - boundingbox[0]) * (boundingbox[3] - boundingbox[1]) +
                    (BBGT[:, 2] - BBGT[:, 0]) *
                    (BBGT[:, 3] - BBGT[:, 1]) - intersection)
            overlaps = intersection / union
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > threshold:
            if not R['det'][jmax]:
                tp[detection] = 1.
                R['det'][jmax] = True
            else:
                fp[detection] = 1.
        else:
            fp[detection] = 1.
    
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    rec = tp / float(num_true_detections)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap

def parse_json(annotation):
    target = []
    for line in open(annotation, 'r'):
        data = json.loads(line)
        
        store = {}
        store['bbox'] = data['bbox']
        store['label'] = data['label']
        
        target.append(store)
    return target

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation Params')
    parser.add_argument('--network')
    parser.add_argument('--weights_path', help="path to pretrained weights")
    parser.add_argument('--dataset')
    parser.add_argument('--data_directory')
    parser.add_argument('--num_classes', type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.network == 'vgg':
       net = SSD_VGG(phase='test', num_classes=args.num_classes)
       net.load_state_dict(torch.load(args.weights_path, map_location=lambda storage, loc: storage))
       net.eval()
    elif args.network == 'mobilenetv2':
       net = SSD_MobileNetV2(phase='test', num_classes=args.num_classes)
       net.load_state_dict(torch.load(args.weights_path, map_location=lambda storage, loc: storage))
       net.eval()
    elif args.network == 'mobilenetv3':
       net = SSD_MobileNetV3(phase='test', num_classes=args.num_classes)
       net.load_state_dict(torch.load(args.weights_path, map_location=lambda storage, loc: storage))
       net.eval()

    if args.dataset == 'fishdata':
        labels = ['fish']
        dataset = FishDataLoader(args.data_directory)

    evaluate(net, dataset, args.num_classes)
