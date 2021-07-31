import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Scalar(nn.Layer):
    def __init__(self):
        super().__init__()
        self.register_buffer('total', paddle.to_tensor([0.0]))
        self.register_buffer('count', paddle.to_tensor([0.0]))

    def forward(self, scalar):
        if not scalar.equal(paddle.to_tensor([0.0])):
            self.count += 1
            self.total += scalar.astype("float32")
        return self.value.cpu()

    @property
    def value(self):
        return self.total / self.count

    def clear(self):
        self.total = self.total * 0
        self.count = self.count * 0

class Accuracy(nn.Layer):
    def __init__(self,
                 dim=1,
                 ignore_idx=-1,
                 threshold=0.5,
                 encode_background_as_zeros=True):
        super().__init__()
        self.register_buffer('total', paddle.to_tensor([0.0]))
        self.register_buffer('count', paddle.to_tensor([0.0]))
        self._ignore_idx = ignore_idx
        self._dim = dim
        self._threshold = threshold
        self._encode_background_as_zeros = encode_background_as_zeros

    def forward(self, labels, preds, weights=None):
        # labels: [N, ...]
        # preds: [N, C, ...]
        if self._encode_background_as_zeros:
            scores = F.sigmoid(preds)
            labels_pred = paddle.argmax(preds, axis=self._dim) + 1
            pred_labels = paddle.where((scores > self._threshold).any(self._dim),
                                      labels_pred,
                                      paddle.to_tensor(0).astype(labels_pred.dtype))
        else:
            pred_labels = paddle.argmax(preds, axis=self._dim)
        N, *Ds = labels.shape
        labels = labels.reshape((N, int(np.prod(Ds))))
        pred_labels = pred_labels.reshape((N, int(np.prod(Ds))))
        if weights is None:
            weights = (labels != self._ignore_idx).astype("float32")
        else:
            weights = weights.astype("float32")

        num_examples = paddle.sum(weights)
        num_examples = paddle.clip(num_examples, min=1.0).astype("float32")
        total = paddle.sum((pred_labels == labels.long()).astype("float32"))
        self.count += num_examples
        self.total += total
        return self.value.cpu()
        # return (total /  num_examples.data).cpu()
    @property
    def value(self):
        return self.total / self.count

    def clear(self):
        self.total = self.total * 0
        self.count = self.count * 0


class Precision(nn.Layer):
    def __init__(self, dim=1, ignore_idx=-1, threshold=0.5):
        super().__init__()
        self.register_buffer('total', paddle.to_tensor([0.0]))
        self.register_buffer('count', paddle.to_tensor([0.0]))
        self._ignore_idx = ignore_idx
        self._dim = dim
        self._threshold = threshold

    def forward(self, labels, preds, weights=None):
        # labels: [N, ...]
        # preds: [N, C, ...]
        if preds.shape[self._dim] == 1:  # BCE
            pred_labels = (F.sigmoid(preds) >
                           self._threshold).long().squeeze(self._dim)
        else:
            assert preds.shape[self._dim] == 2, "precision only support 2 class"
            pred_labels = paddle.argmax(preds, axis=self._dim)
        N, *Ds = labels.shape
        labels = labels.reshape((N, int(np.prod(Ds))))
        pred_labels = pred_labels.reshape((N, int(np.prod(Ds))))
        if weights is None:
            weights = (labels != self._ignore_idx).astype("float32")
        else:
            weights = weights.astype("float32")

        pred_trues = pred_labels > 0
        pred_falses = pred_labels == 0
        trues = labels > 0
        falses = labels == 0
        true_positives = (weights * (trues & pred_trues).astype("float32")).sum()
        true_negatives = (weights * (falses & pred_falses).astype("float32")).sum()
        false_positives = (weights * (falses & pred_trues).astype("float32")).sum()
        false_negatives = (weights * (trues & pred_falses).astype("float32")).sum()
        count = true_positives + false_positives
        # print(count, true_positives)
        if count > 0:
            self.count += count
            self.total += true_positives
        return self.value.cpu()
        # return (total /  num_examples.data).cpu()
    @property
    def value(self):
        return self.total / self.count
    def clear(self):
        self.total = self.total * 0
        self.count = self.count * 0

class Recall(nn.Layer):
    def __init__(self, dim=1, ignore_idx=-1, threshold=0.5):
        super().__init__()
        self.register_buffer('total', paddle.to_tensor([0.0]))
        self.register_buffer('count', paddle.to_tensor([0.0]))
        self._ignore_idx = ignore_idx
        self._dim = dim
        self._threshold = threshold

    def forward(self, labels, preds, weights=None):
        # labels: [N, ...]
        # preds: [N, C, ...]
        if preds.shape[self._dim] == 1:  # BCE
            pred_labels = (F.sigmoid(preds) >
                           self._threshold).long().squeeze(self._dim)
        else:
            assert preds.shape[self._dim] == 2, "precision only support 2 class"
            pred_labels = paddle.argmax(preds, axis=self._dim)
        N, *Ds = labels.shape
        labels = labels.reshape((N, int(np.prod(Ds))))
        pred_labels = pred_labels.reshape((N, int(np.prod(Ds))))
        if weights is None:
            weights = (labels != self._ignore_idx).astype("float32")
        else:
            weights = weights.astype("float32")
        pred_trues = pred_labels == 1
        pred_falses = pred_labels == 0
        trues = labels == 1
        falses = labels == 0
        true_positives = (weights * (trues & pred_trues).astype("float32")).sum()
        true_negatives = (weights * (falses & pred_falses).astype("float32")).sum()
        false_positives = (weights * (falses & pred_trues).astype("float32")).sum()
        false_negatives = (weights * (trues & pred_falses).astype("float32")).sum()
        count = true_positives + false_negatives
        if count > 0:
            self.count += count
            self.total += true_positives
        return self.value.cpu()
        # return (total /  num_examples.data).cpu()
    @property
    def value(self):
        return self.total / self.count
    def clear(self):
        self.total = self.total * 0
        self.count = self.count * 0


def _calc_binary_metrics(labels,
                         scores,
                         weights=None,
                         ignore_idx=-1,
                         threshold=0.5):

    pred_labels = (scores > threshold).long()
    N, *Ds = labels.shape
    labels = labels.view(N, int(np.prod(Ds)))
    pred_labels = pred_labels.view(N, int(np.prod(Ds)))
    pred_trues = pred_labels > 0
    pred_falses = pred_labels == 0
    trues = labels > 0
    falses = labels == 0
    true_positives = (weights * (trues & pred_trues).astype("float32")).sum()
    true_negatives = (weights * (falses & pred_falses).astype("float32")).sum()
    false_positives = (weights * (falses & pred_trues).astype("float32")).sum()
    false_negatives = (weights * (trues & pred_falses).astype("float32")).sum()
    return true_positives, true_negatives, false_positives, false_negatives


class PrecisionRecall(nn.Layer):
    def __init__(self,
                 dim=1,
                 ignore_idx=-1,
                 thresholds=0.5,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True):
        super().__init__()
        if not isinstance(thresholds, (list, tuple)):
            thresholds = [thresholds]

        self.register_buffer('prec_total', paddle.zeros([len(thresholds)]))
        self.register_buffer('prec_count', paddle.zeros([len(thresholds)]))
        self.register_buffer('rec_total', paddle.zeros([len(thresholds)]))
        self.register_buffer('rec_count', paddle.zeros([len(thresholds)]))

        self._ignore_idx = ignore_idx
        self._dim = dim
        self._thresholds = thresholds
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros

    def forward(self, labels, preds, weights=None):
        # labels: [N, ...]
        # preds: [N, ..., C]
        if self._encode_background_as_zeros:
            # this don't support softmax
            assert self._use_sigmoid_score is True
            total_scores = F.sigmoid(preds)
        else:
            if self._use_sigmoid_score:
                total_scores = F.sigmoid(preds)[..., 1:]
            else:
                total_scores = F.softmax(preds, axis=-1)[..., 1:]
        """
        if preds.shape[self._dim] == 1:  # BCE
            scores = paddle.sigmoid(preds)
        else:
            # assert preds.shape[
            #     self._dim] == 2, "precision only support 2 class"
            # TODO: add support for [N, C, ...] format.
            # TODO: add multiclass support
            if self._use_sigmoid_score:
                scores = paddle.sigmoid(preds)[:, ..., 1:].sum(-1)
            else:
                scores = F.softmax(preds, dim=self._dim)[:, ..., 1:].sum(-1)
        """
        scores = paddle.max(total_scores, axis=-1)
        if weights is None:
            weights = (labels != self._ignore_idx).astype("float32")
        else:
            weights = weights.astype("float32")
        for i, thresh in enumerate(self._thresholds):
            tp, tn, fp, fn = _calc_binary_metrics(labels, scores, weights,
                                                  self._ignore_idx, thresh)
            rec_count = tp + fn
            prec_count = tp + fp
            if rec_count > 0:
                self.rec_count[i] += rec_count
                self.rec_total[i] += tp
            if prec_count > 0:
                self.prec_count[i] += prec_count
                self.prec_total[i] += tp

        return self.value
        # return (total /  num_examples.data).cpu()
    @property
    def value(self):
        prec_count = paddle.clip(self.prec_count, min=1.0)
        rec_count = paddle.clip(self.rec_count, min=1.0)
        return ((self.prec_total / prec_count).cpu(),
                (self.rec_total / rec_count).cpu())

    @property
    def thresholds(self):
        return self._thresholds

    def clear(self):
        self.rec_count = self.rec_count * 0
        self.prec_count = self.prec_count * 0
        self.prec_total = self.prec_total * 0
        self.rec_total = self.rec_total * 0