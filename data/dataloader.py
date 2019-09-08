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


CLASS_DICT = {class_name: i for i, class_name in enumerate(VOC_LABELS)}


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
    def __init__(self, rootpath, transform=True, phase='train'):
        self.rootpath = rootpath
        self.phase = phase
        self.means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
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


        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(target)):
            cv2.rectangle(img, (int(target[i][0] * width), int(target[i][1] * height)),
                          (int(target[i][2] * width), int(target[i][3] * height)), (0, 0, 255), 3)
            cv2.rectangle(img, (int(target[i][0] * width), int(target[i][3] * height) - 15),
                          (int(target[i][2] * width), int(target[i][3] * height)), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, 'class : ' + VOC_LABELS[int(target[i][4])],
                        (int(target[i][0] * width) + 6, int(target[i][3] * height) - 6), font, 0.4, (255, 255, 255), 1)

        cv2.imwrite('./plot/result' + str(index) + '.jpg', img)


        if self.transform:
            img = cv2.resize(img, (300, 300))
            img = np.array(img, dtype=np.float32)
            self.means = np.array(self.means, dtype=np.float32)
            self.std = np.array(self.std, dtype=np.float32)
            img = ((img / 255.) - self.means) / self.std
            img = img[:, :, (2, 1, 0)]

        return torch.from_numpy(img).float().permute(2, 0, 1), torch.from_numpy(target).float()

    def __len__(self):
        return len(self.images)

    def detection_collate(self, batch):
        targets = []
        imgs = []

        for sample in batch:
            imgs.append(sample[0])
            targets.append(torch.FloatTensor(sample[1]))

        return torch.stack(imgs, 0), targets


# Test
if __name__ == "__main__":
    dataset = PascalVocDataLoader('/Users/bogireddyteja/Work/datasets/VOCdevkit/VOC2007/', transform=True)
    dataloader = data.DataLoader(dataset, 8, shuffle=True, collate_fn=dataset.detection_collate)
    num_classes = 21

    for images, targets in dataloader:
        print(images.shape)