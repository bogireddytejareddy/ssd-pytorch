import torch
from torch.autograd import Function
from utils.box_utils import nms
from utils.box_utils import decode

class InferenceLayer(Function):
    def __init__(self, top_k, conf_thresh, nms_thresh):
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.num_classes = 2
        self.variance = [0.1, 0.2]
        
    def forward(self, loc_p, class_p, priors):
        batch_size = loc_p.size(0)
        num_priors = priors.size(0)
        output = torch.zeros(batch_size, self.num_classes, self.top_k, 5)
        class_p = class_p.transpose(1,2)
        
        for idx in range(batch_size):
            decoded_boxes = decode(loc_p[idx], priors, self.variance)
            for c in range(1, self.num_classes):
                c_mask = (class_p[idx][c]>self.conf_thresh)
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                scores = class_p[idx][c][c_mask]
                boxes = decoded_boxes[l_mask].view(-1, 4)
                if len(scores)==0:
                    continue
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[idx, c, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                    boxes[ids[:count]]), 1)
        return output
