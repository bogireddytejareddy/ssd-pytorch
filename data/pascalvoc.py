import os
import cv2
import torch
import numpy as np
import torch.utils.data as data
import xml.etree.ElementTree as ET

VOC_LABELS = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle",
              "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
              "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
              "train", "tvmonitor")

CLASS_DICT = {class_name : i for i, class_name in enumerate(VOC_LABELS)}

def parse_xml(annotation_path, height, width):
    xml_parser = ET.parse(annotation_path).getroot()

    target = []
    for objc in xml_parser.iter('object'):
        name = objc.find('name').text.lower().strip()
        bbox = objc.find('bndbox')

        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        boundingbox = []
        for i, pt in enumerate(pts):
            point = int(bbox.find(pt).text) - 1
            point = point / width if i % 2 == 0 else point / height
            boundingbox.append(point)
        
        label_idx = CLASS_DICT[name]
        boundingbox.append(label_idx)
        target += [boundingbox]

    return np.asarray(target)

class PascalVocDataLoader(data.Dataset):
    def __init__(self, rootpath, transform=None, phase='train'):
        self.rootpath = rootpath
        self.phase = phase
        self.means = (104, 117, 123)
        self.transform = transform

        with open(os.path.join(self.rootpath, 'ImageSets/Main/train.txt')) as txt:
            filenames = txt.readlines()
        
        filenames = [filenames[x][:-1] for x in range(len(filenames))]

        self.images = [self.rootpath + 'JPEGImages/' + x + '.jpg' for x in filenames]
        self.annotations = [self.rootpath + 'Annotations/' + x + '.xml' for x in filenames]

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        height, width, channels = img.shape

        target = parse_xml(self.annotations[index], height, width)

        if self.transform:
            image, boxes, labels = self.transform(image, target[:, :4], target[:, 4])
            image = image[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).float().permute(2, 0, 1), torch.from_numpy(target).float()

    def __len__(self):
        return len(self.images)
