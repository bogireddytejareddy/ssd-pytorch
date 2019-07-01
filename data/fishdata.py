import os
import json
import torch
import numpy as np
import cv2
from torchvision import transforms

class FishDataLoader(object):
    def __init__(self, path = "../Data/FishData/", transform=None, phase='train'):
        self.path = path
        self.transform = transform
        self.phase = phase

        self.images = sorted(os.listdir(path + "images"))
        self.annotations = sorted(os.listdir(path + "annotations"))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, annotation = self.path + 'images/' + self.images[idx], self.path + 'annotations/' + self.annotations[idx]
        image = cv2.imread(image)
        height, width, channel = image.shape

        target = parse_json(annotation, height, width)
        if self.transform is not None:
           image, boxes, labels = self.transform(image, target[:, :4], target[:, 4])
           image = image[:, :, (2, 1, 0)]
           target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(image).float().permute(2, 0, 1), torch.from_numpy(target).float()

    def pull_item(self, idx):
        image, annotation = self.path + 'images/' + self.images[idx], self.path + 'annotations/' + self.annotations[idx]

        image = cv2.imread(image)
        height, width, channel = image.shape

        target = parse_json(annotation, height, width)

        if self.transform:
           image, boxes, labels = self.transform(image, target[:, :4], target[:, 4])
           image = image[:, :, (2, 1, 0)]
           target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(image).float().permute(2, 0, 1), torch.from_numpy(target).float(), width, height

def parse_json(annotation, height, width):
    target = []

    for line in open(annotation, 'r'):
        data = json.loads(line)
        boundingbox = []
        bbox = data['bbox']
        for i, pt in enumerate(bbox):
            point = pt / width if i % 2 == 0 else pt / height
            if point > 1.:
               point = 1.
            elif point < 0:
               point = 0.
            boundingbox.append(point)

        label = data['label']
        boundingbox.append(label)

        target += [boundingbox]
    return np.asarray(target)
