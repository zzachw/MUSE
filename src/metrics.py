import heapq
import logging
from collections import OrderedDict

import numpy as np
from scipy.special import softmax
from sklearn import metrics


def update_dict(d, k, v):
    if k not in d:
        d[k] = []
    d[k].append(v)
    return d


def select_sample_with_replacement(y_true, y_score, num_samples):
    num_elements = y_true.shape[0]
    idx = np.random.choice(np.arange(num_elements), num_samples)
    y_true_sample = y_true[idx]
    y_score_sample = y_score[idx]
    return y_true_sample, y_score_sample


def bootstrapping(y_true, y_score, metric_func, num_iterations, num_samples, **kwargs):
    if num_samples is None:
        num_samples = y_true.shape[0]
    outputs = {}
    for _ in range(num_iterations):
        while True:
            y_true_sample, y_score_sample = None, None
            try:
                y_true_sample, y_score_sample = select_sample_with_replacement(y_true, y_score, num_samples)
                output = metric_func(y_true_sample, y_score_sample, **kwargs)
                break
            except ValueError:
                # sampled set may not contains all classes
                cls = set(y_true)
                sample_cls = set(y_true_sample)
                miss_cls = {c for c in cls if c not in sample_cls}
                cls_msg = "\n".join(
                    [f"\tMissing class {c} prevalence: {(y_true == c).sum()}/{len(y_true)}" for c in miss_cls])
                logging.debug(f"Sample discarded! Unmatched number of classes. \n {cls_msg}")
        for k, v in output.items():
            outputs = update_dict(outputs, k, v)
    statistics = {}
    for k, v in outputs.items():
        statistics[k] = np.mean(v)
        statistics[k + "_std"] = np.std(v)
    return statistics


def get_metrics(func, y_true, y_score, bootstrap, num_iterations, num_samples, **kwargs):
    if bootstrap:
        ret = bootstrapping(y_true, y_score, func, num_iterations, num_samples,
                            **kwargs)
    else:
        ret = func(y_true, y_score, **kwargs)
    return OrderedDict(sorted(ret.items()))


def get_metrics_binary(y_true, y_score, bootstrap=False, num_iterations=1000, num_samples=None):
    assert len(set(y_true)) == 2, f"Unmatched number of classes. Missing {[c for c in range(2) if c not in y_true]}"
    return get_metrics(metrics_binary, y_true, y_score, bootstrap, num_iterations, num_samples)


def get_metrics_multiclass(y_true, y_score, bootstrap=False, num_iterations=1000, num_samples=None):
    assert len(set(y_true)) == y_score.shape[1], \
        f"Unmatched number of classes. Missing {[c for c in range(y_score.shape[1]) if c not in y_true]}"
    return get_metrics(metrics_multiclass, y_true, y_score, bootstrap, num_iterations, num_samples)


def get_metrics_multilabel_v1(y_true, y_score, bootstrap=False, num_iterations=1000, num_samples=None):
    return get_metrics(metrics_multilabel_v1, y_true, y_score, bootstrap, num_iterations, num_samples)


def get_metrics_multilabel_v2(y_true, y_score, ks, bootstrap=False, num_iterations=1000, num_samples=None):
    return get_metrics(metrics_multilabel_v2, y_true, y_score, bootstrap, num_iterations, num_samples, ks=ks)


def metrics_binary(y_true, y_score):
    y_pred = (y_score > 0.5).astype(int)
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    roc_auc = metrics.roc_auc_score(y_true, y_score)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    pr_auc = metrics.auc(recall, precision)
    cohen_kappa = metrics.cohen_kappa_score(y_true, y_pred)
    return {"acc": acc, "f1": f1, "pr_auc": pr_auc, "roc_auc": roc_auc, "cohen_kappa": cohen_kappa}


def metrics_multiclass(y_true, y_score):
    y_score = softmax(y_score, axis=-1)
    y_pred = np.argmax(y_score, axis=-1)
    acc = metrics.accuracy_score(y_true, y_pred)
    bacc = metrics.balanced_accuracy_score(y_true, y_pred)
    f1_micro = metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = metrics.f1_score(y_true, y_pred, average="macro")
    f1_weighted = metrics.f1_score(y_true, y_pred, average="weighted")
    ave_auc_macro_ovo = metrics.roc_auc_score(y_true, y_score, average="macro", multi_class="ovo")
    ave_auc_macro_ovr = metrics.roc_auc_score(y_true, y_score, average="macro", multi_class="ovr")
    ave_auc_weighted_ovo = metrics.roc_auc_score(y_true, y_score, average="weighted", multi_class="ovo")
    ave_auc_weighted_ovr = metrics.roc_auc_score(y_true, y_score, average="weighted", multi_class="ovr")

    return {"acc": acc,
            "bacc": bacc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "auc_macro_ovo": ave_auc_macro_ovo,
            "auc_macro_ovr": ave_auc_macro_ovr,
            "auc_weighted_ovo": ave_auc_weighted_ovo,
            "auc_weighted_ovr": ave_auc_weighted_ovr}


def metrics_multilabel_v1(y_true, y_score):
    auc_scores = metrics.roc_auc_score(y_true, y_score, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, y_score, average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, y_score, average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, y_score, average="weighted")

    return {"auc_scores": auc_scores,
            "auc_micro": ave_auc_micro,
            "auc_macro": ave_auc_macro,
            "auc_weighted": ave_auc_weighted}


def metrics_multilabel_v2(y_true, y_score, ks):
    result = {}
    for k in ks:
        r, m = metrics_multilabel_v2_at_k(y_true, y_score, k)
        result[f"R@{k}"] = r
        result[f"MAP@{k}"] = m
    return result


def metrics_multilabel_v2_at_k(y_true, y_score, k):
    batch_size, output_size = y_score.shape
    if k > output_size:
        k = output_size
    recalls = []
    maps = []
    for i in range(batch_size):
        labels = y_true[i, :]
        predicts = y_score[i, :]
        yNum = float(np.sum(labels))
        topK = heapq.nlargest(k, range(output_size), predicts.take)
        hit = 0
        _map = 0
        for r in range(k):
            if labels[topK[r]] == 1:
                hit += 1
                _map += float(hit) / float(r + 1)
        _map /= yNum
        recall = float(hit) / yNum
        recalls.append(recall)
        maps.append(_map)
    return np.mean(recalls), np.mean(maps)


if __name__ == "__main__":
    # binary
    y_true = np.random.randint(2, size=100)
    y_score = np.random.random(size=100)
    print(get_metrics_binary(y_true, y_score, bootstrap=False))
    print(get_metrics_binary(y_true, y_score, bootstrap=True))

    # multi class
    y_true = np.random.randint(4, size=100)
    y_score = np.random.randn(100, 4)
    print(get_metrics_multiclass(y_true, y_score, bootstrap=False))
    print(get_metrics_multiclass(y_true, y_score, bootstrap=True))

    # multi label v1
    y_true = np.random.randint(2, size=(100, 4))
    y_score = np.random.random(size=(100, 4))
    print(get_metrics_multilabel_v1(y_true, y_score, bootstrap=False))
    print(get_metrics_multilabel_v1(y_true, y_score, bootstrap=True))

    # multi label v2
    y_true = np.random.randint(2, size=(10, 1000))
    y_score = np.random.random(size=(10, 1000))
    print(get_metrics_multilabel_v2(y_true, y_score, bootstrap=False, ks=[10, 50, 100, 500]))
    print(get_metrics_multilabel_v2(y_true, y_score, bootstrap=True, ks=[10, 50, 100, 500]))
