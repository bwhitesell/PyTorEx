import torch
import numpy as np


def eval_in_batches(ds, net, criteria, metric=None):
    loss = 0

    for i, data in enumerate(ds):
        inputs, labels = data
        outputs = net(inputs)
        loss += criteria(net(inputs), labels).item()
        if metric:
            metric.update(outputs, labels)

    metric_eval = metric.eval() if metric else None
    return loss / (len(ds) + 1), metric_eval


class Accuracy:
    correct = 0
    total = 0

    def update(self, outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

    def eval(self):
        return self.correct / self.total


class RMSPE:
    err = 0
    n_labels = 1

    def update(self, outputs, labels):
        if (labels == 0).sum() == 0:
            labels = torch.exp(labels) - 1
            outputs = torch.exp(outputs.flatten()) - 1
            pct_err = ((labels - outputs) / labels)**2
            self.err += pct_err.sum()
            self.n_labels += labels.size()[0]

    def eval(self):
        return np.sqrt(self.err / self.n_labels)


def standard_kfold_train(k, baseline, features, target, model):
    predictions = []
    incr = int(len(baseline)/k)
    for i in range(k):
        train_x = baseline.drop(baseline.index[range(incr*(i), incr*(i+1))])
        train_y = baseline.drop(baseline.index[range(incr*(i), incr*(i+1))]
                                )[target]
        test_x = baseline[incr*(i):incr*(i+1)]

        model.fit(train_x[features], train_y)
        pred = model.predict(test_x[features])
        predictions = predictions + pred.tolist()
    if len(baseline) % k != 0:
        train_x = baseline[:incr*(i+1)]
        train_y = baseline[:incr*(i+1)][target]
        test_x = baseline[incr*(i+1):]

        model.fit(train_x[features], train_y)
        pred = model.predict(test_x[features])
        predictions = predictions + pred.tolist()

    baseline[target+'_projections'] = predictions

    return baseline, model