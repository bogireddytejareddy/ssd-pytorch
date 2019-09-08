import torch
import math


def priorbox(model):
    if model == 'vgg':
        feature_maps = [38, 19, 10, 5, 3, 1]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    if model == 'mobilenetv2' or model == 'mobilenetv3' or model == 'efficientnet':
        feature_maps = [19, 10, 5, 3, 2, 1]
        aspect_ratios = [[2, 3], [3, 4], [3, 4], [3], [3], [3]]

    maps = len(feature_maps)
    s_min = 0.1
    s_max = 0.9
    anchor_box_list = []

    for k, dimension in enumerate(feature_maps):
        scale = s_min + (s_max - s_min) * k / (maps - 1)
        scale_next = s_min + (s_max - s_min) * (k + 1) / (maps - 1)
        scale_2 = math.sqrt(scale * scale_next)
        for i in range(dimension):
            for j in range(dimension):
                cx = (j + 0.5) / dimension
                cy = (i + 0.5) / dimension

                anchor_box_list += [cx, cy, scale, scale]
                anchor_box_list += [cx, cy, scale_2, scale_2]

                for r in aspect_ratios[k]:
                    anchor_box_list += [cx, cy, scale * math.sqrt(r), scale / math.sqrt(r)]
                    anchor_box_list += [cx, cy, scale / math.sqrt(r), scale * math.sqrt(r)]

    output = torch.Tensor(anchor_box_list).view(-1, 4)
    return output


def point_form(box):
    return torch.cat([box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2], 1)


def encode(loc_t, priors, variance):
    gt_cxcy = (loc_t[:, :2] + loc_t[:, 2:]) / 2 - priors[:, :2]
    gt_cxcy /= (priors[:, 2:] * variance[0])
    gt_wh = (loc_t[:, 2:] - loc_t[:, :2]) / priors[:, 2:]
    gt_wh = torch.log(gt_wh) / variance[1]
    return torch.cat([gt_cxcy, gt_wh], 1)


def decode(loc_p, priors, variance):
    cxcy_p = loc_p[:, :2] * (priors[:, 2:] * variance[0]) + priors[:, :2]
    wh_p = torch.exp(loc_p[:, 2:] * variance[1]) * priors[:, 2:]
    return torch.cat([cxcy_p - wh_p / 2, cxcy_p + wh_p / 2], 1)


def assign(threshold, variance, gt_box, gt_class, priors, loc_t, class_t, idx):
    iou = iou_jaccard(gt_box, point_form(priors))
    best_iou, best_gt_idx = iou.max(0, keepdim=False)
    _, best_prior_idx = iou.max(1, keepdim=False)
    best_iou.index_fill_(0, best_prior_idx, 2.0)
    loc_t[idx] = encode(gt_box[best_gt_idx], priors, variance)
    class_t_idx = gt_class[best_gt_idx]
    class_t_idx[best_iou < threshold] = 0
    class_t[idx] = class_t_idx


def iou_jaccard(box1, box2):
    len1 = box1.size(0)
    len2 = box2.size(0)

    left = torch.max(box1[:, 0].unsqueeze(1).expand(len1, len2),
                     box2[:, 0].unsqueeze(0).expand(len1, len2))
    top = torch.max(box1[:, 1].unsqueeze(1).expand(len1, len2),
                    box2[:, 1].unsqueeze(0).expand(len1, len2))
    right = torch.min(box1[:, 2].unsqueeze(1).expand(len1, len2),
                      box2[:, 2].unsqueeze(0).expand(len1, len2))
    bottom = torch.min(box1[:, 3].unsqueeze(1).expand(len1, len2),
                       box2[:, 3].unsqueeze(0).expand(len1, len2))

    intersect = torch.clamp(right - left, min=0) * torch.clamp(bottom - top, min=0)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area1 = area1.unsqueeze(1).expand_as(intersect)
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    area2 = area2.unsqueeze(0).expand_as(intersect)

    union = area1 + area2 - intersect
    return intersect / union


def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]

        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]

        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])

        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1

        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h

        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union

        idx = idx[IoU.le(overlap)]
    return keep, count
