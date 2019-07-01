from utils.box_utils import *
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes):
        super(MultiBoxLoss, self).__init__()
        self.variance = [0.1, 0.2]
        self.num_classes = num_classes

    def forward(self, preds, targets):
        loc_preds, conf_preds, priors = preds
        
        batch_size = loc_preds.size(0)
        num_priors = priors.size(0)

        loc_tars = torch.FloatTensor(batch_size, num_priors, 4)
        class_tars = torch.LongTensor(batch_size, num_priors)

        for idx in range(batch_size):
            target_box = targets[idx][:, :-1].data
            target_class = targets[idx][:, -1].data
            assign(0.45, self.variance, target_box, target_class, priors.data, loc_tars, class_tars, idx)

        loc_tars = loc_tars.cuda()
        class_tars = class_tars.cuda()        

        loc_tars = Variable(loc_tars, requires_grad=False)
        class_tars = Variable(class_tars, requires_grad=False)

        non_background_mask = (class_tars > 0)
        num_pos_each_sample = non_background_mask.long().sum(1, keepdim=True)

        loss_l = F.smooth_l1_loss(loc_preds[non_background_mask, :], loc_tars[non_background_mask, :], size_average=False)

        loss_gt = -torch.log(F.softmax(conf_preds.view(-1, self.num_classes)))
        loss_gt = loss_gt.gather(dim=1, index=class_tars.view(-1, 1)) #find softmax of class from bt_sz * priors * num_classes
        loss_gt[non_background_mask.view(-1, 1)] = 0 # zero non background
        loss_gt = loss_gt.view(batch_size, -1)
        
        _, loss_sort_idx = loss_gt.sort(1, descending=True) # find backgrounds with highest softmax value 300, 500, 200, 100, 600
        _, idx_rank = loss_sort_idx.sort(1) # sort 100, 200, 300, 500, 600 -> 4, 3, 1, 2, 5

        num_neg_each_sample = torch.clamp(3*num_pos_each_sample, max=non_background_mask.size(1)-1) # fix no. of neg samples
        background_mask = idx_rank < num_neg_each_sample.expand_as(idx_rank) # if neg_samples > actual rank from sort of softmax then consider
        
        pos_mask_class = non_background_mask.unsqueeze(2).expand_as(conf_preds) 
        neg_mask_class = background_mask.unsqueeze(2).expand_as(conf_preds)

        conf_preds = conf_preds[(pos_mask_class + neg_mask_class) > 0].view(-1, self.num_classes)
        class_tars = class_tars[(non_background_mask + background_mask) > 0].view(-1)
        
        loss_c = F.cross_entropy(conf_preds, class_tars, size_average=False)

        N = num_pos_each_sample.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
