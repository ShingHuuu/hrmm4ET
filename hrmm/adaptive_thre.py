
from functools import partial
from scorer import strict
import numpy as np

def tune(baseline, dist, type_, num_types, init_threshold):
    idx2threshold = {idx: init_threshold for idx in range(num_types)}
    func = partial(search_threshold,
                   init_threshold=init_threshold,
                   num_types=num_types,
                   dist=dist,
                   type_=type_,
                   baseline=baseline)
    for idx, best_t in map(func, range(num_types)):
        idx2threshold[idx] = best_t
    return idx2threshold


def search_threshold(idx, init_threshold, num_types, dist, type_, baseline):
    # Search the best thresholds.
    idx2threshold = {i: init_threshold for i in range(num_types)}
    best_t = idx2threshold[idx]
    for t in list(np.linspace(0, 1.0, num=20)):
        idx2threshold[idx] = t
        pred = predict(dist, type_, idx2threshold)
        _, _, temp = strict(pred)
        if temp > baseline:
            best_t = t
    print ('-', end='')
    return idx, best_t


def predict(pred_dist, Y, idx2threshold=None):
    ret = []
    batch_size = pred_dist.shape[0]
    for i in range(batch_size):
        dist = pred_dist[i]
        type_vec = Y[i]
        pred_type = []
        gold_type = []
        for idx, score in enumerate(list(type_vec)):
            if score > 0:
                gold_type.append(idx)
        midx, score = max(enumerate(list(dist)), key=lambda x: x[1])
        pred_type.append(midx)
        for idx, score in enumerate(list(dist)):
            if idx2threshold is None:
                threshold = 0.5
            else:
                threshold = idx2threshold[idx]
            if score > threshold and idx != midx:
                pred_type.append(idx)
        ret.append([gold_type, pred_type])
    return ret