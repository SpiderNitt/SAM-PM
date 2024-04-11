import torch

def jaccard_dice(pred, gt, eps=1e-6):
    intersection = (pred & gt)
    union = (pred | gt)
    j = intersection.sum() / (union.sum() + eps)
    d = (2 * intersection.sum()) / (pred.sum() + gt.sum() + eps)
    return j, d