from __future__ import print_function

import json
import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Variable

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

def predict_label_prot(model, data_loader, num_class, batch_size):
    model.eval()
    prototypes = []
    pred_list = []

    for i, (inputs, _) in enumerate(data_loader):
        inputs = inputs.cuda()
        outputs, feature = model(inputs, out_feature=True)
        feature = feature.data.cpu().numpy()
        pred = outputs.data.max(1)[1]
        pred_list.append(pred)
        if i == 0:
            features = np.zeros((len(data_loader.dataset), feature.shape[1]), dtype='float32')
        if i < len(data_loader) - 1:
            features[i * batch_size: (i + 1) * batch_size] = feature
        else:
            # special treatment for final batch
            features[i * batch_size:] = feature

    num_list = []
    temp_pred = torch.cat(pred_list, dim=0)
    # temp_pred = temp_pred.tolist()
    # for i in range(10):
    #     num = temp_pred.count(i)
    #     num_list.append(num)
    # print(num_list)
    temp_pred = temp_pred.cpu()
    for c in range(num_class):
        prototype = features[np.where(temp_pred == c)].mean(0)  # compute prototypes with pseudo-label
        prototypes.append(torch.Tensor(prototype))
    prototypes = torch.stack(prototypes).cuda()
    prototypes = F.normalize(prototypes, p=2, dim=1)


    return torch.cat(pred_list, dim=0), prototypes

def predict_label(model, data_loader):
    model.eval()
    pred_list = []
    index = 0
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.cuda()
        outputs = model(inputs)
        pred = outputs.data.max(1)[1]
        index += inputs.shape[0]
        pred_list.append(pred)
    temp_pred = torch.cat(pred_list, dim=0)
    temp_pred = temp_pred.tolist()
    # print(temp_pred)
    num_list = []
    for i in range(10):
        num = temp_pred.count(i)
        num_list.append(num)
    print(num_list)
    return torch.cat(pred_list, dim=0)

def cal_prot(model, tranloader, num_class, batch_size, targets):
    prototypes = []
    model.eval()
    for i, (images, _) in enumerate(tranloader):
        images = Variable(images).cuda()
        with torch.no_grad():
            _, feature = model(images, out_feature=True)
            feature = feature.data.cpu().numpy()
            targets = targets.cpu()
            if i == 0:
                features = np.zeros((len(tranloader.dataset), feature.shape[1]), dtype='float32')
            if i < len(tranloader) - 1:
                features[i * batch_size: (i + 1) * batch_size] = feature
            else:
                # special treatment for final batch
                features[i * batch_size:] = feature
    for c in range(num_class):
        prototype = features[np.where(targets == c)].mean(0)  # compute prototypes with pseudo-label
        prototypes.append(torch.Tensor(prototype))
    prototypes = torch.stack(prototypes).cuda()
    prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes

def cal_prot(model, tranloader, num_class, opt, embed_t):
    prototypes = []
    model.eval()
    for i, (images, labels) in enumerate(tranloader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        with torch.no_grad():
            feats, _ = model(images, is_feat=True)
            feature = embed_t(feats[-1])
            feature = feature.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            if i == 0:
                features = np.zeros((len(tranloader.dataset), feature.shape[1]), dtype='float32')
                targets = np.zeros((len(tranloader.dataset)), dtype='int')
            if i < len(tranloader) - 1:
                features[i * opt.batch_size: (i + 1) * opt.batch_size] = feature
                targets[i * opt.batch_size: (i + 1) * opt.batch_size] = labels
            else:
                # special treatment for final batch
                features[i * opt.batch_size:] = feature
                targets[i * opt.batch_size:] = labels
    for c in range(num_class):
        prototype = features[np.where(targets == c)].mean(0)  # compute prototypes with pseudo-label
        prototypes.append(torch.Tensor(prototype))
    prototypes = torch.stack(prototypes).cuda()
    prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes

def pro_constrastive_loss(target, feat_t, feat_s, prototypes, opt):

    temperature = 0.3
    if opt.dataset == 'cifar10':
        labels = torch.zeros(len(target), 10).cuda().scatter_(1, target.view(-1, 1), 1)
    else:
        labels = torch.zeros(len(target), 100).cuda().scatter_(1, target.view(-1, 1), 1)
    logits_proto_t = torch.mm(feat_t, prototypes.t()) / temperature
    logits_proto_s = torch.mm(feat_s, prototypes.t()) / temperature
    loss_proto_t = -torch.mean(torch.sum(F.log_softmax(logits_proto_t, dim=1) * labels, dim=1))
    loss_proto_s = -torch.mean(torch.sum(F.log_softmax(logits_proto_s, dim=1) * labels, dim=1))

    return loss_proto_t + loss_proto_s


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

# def adjust_learning_rate(optimizer, epoch, step, len_epoch, old_lr):
#     """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
#     if epoch < 5:
#         lr = old_lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)
#     elif 5 <= epoch < 60: return
#     else:
#         factor = epoch // 30
#         factor -= 1
#         lr = old_lr*(0.1**factor)

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def min_max(v):
    min_v = torch.min(v)
    max_v = torch.max(v)
    n1 = (v - min_v) / (max_v - v)
    return 1.0 - n1

def stand(v):
    mean_a = torch.mean(v)
    std_a = torch.std(v)
    n1 = (v - mean_a) / std_a
    return 1.0 - n1


def weight_mse_loss(input, target, weights):
    weights = torch.tensor(weights)
    t_size = input.size(0)
    pct_var = (input - target) ** 2
    pct_var_temp = pct_var.clone()

    for i in range(t_size):
        pct_var_temp[i] = pct_var[i] * weights[i]

    loss = pct_var_temp.mean()
    return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim = 1, largest = True, sorted = True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def load_json_to_dict(json_path):
    """Loads json file to dict

    Args:
        json_path: (string) path to json file
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def reduce_tensor(tensor, world_size = 1, op='avg'):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size > 1:
        rt = torch.true_divide(rt, world_size)
    return rt
def TimeConverter(s, epoch, epochs):
    T_time = s * epochs
    T_m = round(T_time / 60, 2)
    T_h = round(T_m / 60, 2)

    R_time = s * (epochs - epoch)
    R_m = round(R_time / 60, 2)
    R_h = round(R_m / 60, 2)

    text = 'Epoch: {} Total time: {} hours, Rest time: {} hours'.format(epoch, T_h, R_h)
    return text

def text_record(text, full_path):
    file = open(full_path, 'a')
    file.write(text)
    file.write('\n')
    file.close()

def get_mask(pred_s, pred_t):
    mask = []
    for i in range(len(pred_s)):
        if pred_s[i] == pred_t[i]:
            mask.append(1)
        else:
            mask.append(0)
    return mask

if __name__ == '__main__':

    pass

