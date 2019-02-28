import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def cross_flow(input, target, flow=None, flow_weight=1.0, max_shift=1, threshold=0.2, use_label_and_flow=False, weight=None,
               size_average=True):
    if target is None:
        return flow_consistency(input, flow, flow_weight, max_shift, threshold, size_average)
    else:
        return cross_entropy2d(input, target, size_average=size_average)


def flow_consistency_v2(input, flow, flow_weight=1.0, max_shift=1, threshold=0.5, size_average=True):
    flow_weight = float(flow_weight)
    input = F.softmax(input, dim=1).permute(1, 0, 2, 3)
    flow = flow.permute(1, 0, 2, 3)
    assert np.shape(flow)[0] == 2
    f_dist = torch.zeros_like(input)
    for s_height in range(-max_shift, max_shift+1):
        for s_width in range(-max_shift, max_shift+1):
            if s_height == 0 and s_width == 0:
                continue
            shifted_input, shifted_flow = shift(input.detach(), flow, s_height, s_width)
            norm = (threshold - torch.norm(flow - shifted_flow, dim=0, keepdim=True)) / threshold
            ignore = torch.norm(shifted_flow, dim=0, keepdim=True) == 0
            norm[ignore] = 0
            norm[norm <= 0] = 0
            s_level = max(abs(s_height), abs(s_width))
            n_shifts = 8 * s_level
            f_dist += norm * shifted_input / n_shifts
    f_ce = -torch.sum(f_dist * torch.log(input + 1.0e-10), dim=0, keepdim=True)
    if size_average:
        f_loss = torch.sum(f_ce) / f_ce.numel()
    else:
        f_loss = torch.sum(f_ce)

    return flow_weight * f_loss


def flow_consistency(input, flow, flow_weight=1.0, max_shift=1, threshold=0.2, size_average=True):
    flow_weight = float(flow_weight)
    input = F.softmax(input, dim=1).permute(1, 0, 2, 3)
    flow = flow.permute(1, 0, 2, 3)
    assert np.shape(flow)[0] == 2
    f_dist = torch.zeros_like(input)
    for s_height in range(-max_shift, max_shift+1):
        for s_width in range(-max_shift, max_shift+1):
            if s_height == 0 and s_width == 0:
                continue
            shifted_input, shifted_flow = shift(input.detach(), flow, s_height, s_width)
            norm = torch.norm(flow - shifted_flow, dim=0, keepdim=True)
            ignore = torch.norm(shifted_flow, dim=0, keepdim=True) == 0
            norm[ignore] = 0
            norm[norm <= threshold] = 0
            s_level = max(abs(s_height), abs(s_width))
            n_shifts = 8 * s_level
            f_dist += norm * shifted_input / n_shifts
    f_ce = torch.sum(f_dist * torch.log(input + 1.0e-10), dim=0, keepdim=True)
    if size_average:
        f_loss = torch.sum(f_ce) / f_ce.numel()
    else:
        f_loss = torch.sum(f_ce)

    return flow_weight * f_loss

# this loss function uses cross entropy loss if labels are available and otherwise uses flow consistency loss
def cross_flow_old(input, target, flow=None, flow_weight=1.0, max_shift=1, threshold=0.2, use_label_and_flow=False, weight=None,
               size_average=True):
    flow_weight = float(flow_weight)
    # split the batch into the flow loss and the cross entropy loss part
    f_batch = torch.zeros(target.size()[0], dtype=torch.uint8)
    for i in range(target.size()[0]):
        f_batch[i] = (target[i, 0, 0] == 255)
    # select the data used for flow loss part
    if use_label_and_flow:
        f_input = input[...]
    else:
        f_input = input[f_batch, ...]
    # select the data used for cross entropy part
    ce_input = input[1 - f_batch, ...]
    ce_target = target[1 - f_batch, ...]

    # use flow consistency loss if target is not available
    f_loss = 0.0
    if flow is not None and len(f_input) > 0:
        if use_label_and_flow:
            f_flow = flow[...]
        else:
            f_flow = flow[f_batch, ...]
        f_input = F.softmax(f_input, dim=1).permute(1, 0, 2, 3)
        f_flow = f_flow.permute(1, 0, 2, 3)
        assert np.shape(f_flow)[0] == 2
        f_dist = torch.zeros_like(f_input)
        for s_height in range(-max_shift, max_shift+1):
            for s_width in range(-max_shift, max_shift+1):
                if s_height == 0 and s_width == 0:
                    continue
                shifted_input, shifted_flow = shift(f_input.detach(), f_flow, s_height, s_width)
                norm = torch.norm(f_flow - shifted_flow, dim=0, keepdim=True)
                ignore = torch.norm(shifted_flow, dim=0, keepdim=True) == 0
                norm[ignore] = 0
                norm[norm <= threshold] = 0
                s_level = max(abs(s_height), abs(s_width))
                n_shifts = 8 * s_level
                f_dist += norm * shifted_input / n_shifts
        f_ce = torch.sum(f_dist * torch.log(f_input + 1.0e-10), dim=0, keepdim=True)
        if size_average:
            f_loss = torch.sum(f_ce) / f_ce.numel()
        else:
            f_loss = torch.sum(f_ce)

    # if target is available, use the standard cross entropy loss.
    if len(ce_input) > 0:
        assert not ce_target[0, 0, 0] == 255
        ce_loss = cross_entropy2d(ce_input, ce_target, weight=weight, size_average=size_average)
    else:
        ce_loss = 0.0

    return flow_weight * f_loss + ce_loss


def shift(input, flow, h_add, w_add):
    # calculate the shifted input and flow
    shifted_input_h = torch.zeros_like(input)
    shifted_input = torch.zeros_like(input)
    shifted_flow_h = torch.zeros_like(flow)
    shifted_flow = torch.zeros_like(flow)
    if h_add > 0:
        shifted_input_h[:, :, h_add:, :] = input[:, :, :-h_add, :]
        shifted_flow_h[:, :, h_add:, :] = flow[:, :, :-h_add, :]
    elif h_add == 0:
        shifted_input_h[:, :, :, :] = input[:, :, :, :]
        shifted_flow_h[:, :, :, :] = flow[:, :, :, :]
    else:
        h_add = abs(h_add)
        shifted_input_h[:, :, :-h_add, :] = input[:, :, h_add:, :]
        shifted_flow_h[:, :, :-h_add, :] = flow[:, :, h_add:, :]
    if w_add > 0:
        shifted_input[:, :, :, w_add:] = shifted_input_h[:, :, :, :-w_add]
        shifted_flow[:, :, :, w_add:] = shifted_flow_h[:, :, :, :-w_add]
    elif w_add == 0:
        shifted_input[:, :, :, :] = shifted_input_h[:, :, :, :]
        shifted_flow[:, :, :, :] = shifted_flow_h[:, :, :, :]
    else:
        w_add = abs(w_add)
        shifted_input[:, :, :, :-w_add] = shifted_input_h[:, :, :, w_add:]
        shifted_flow[:, :, :, :-w_add] = shifted_flow_h[:, :, :, w_add:]
    return shifted_input, shifted_flow


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def multi_scale_cross_entropy2d(
    input, target, weight=None, size_average=True, scale_weight=None
):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp))

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input,
                                  target, 
                                  K, 
                                  weight=None, 
                                  size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, 
                                   target, 
                                   K, 
                                   weight=None,
                                   size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input, 
                               target, 
                               weight=weight, 
                               reduce=False,
                               size_average=False, 
                               ignore_index=250)

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)
