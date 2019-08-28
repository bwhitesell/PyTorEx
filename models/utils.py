import torch


def eval_in_batches(ds, net, criteria, metric=None):
    loss = 0
    total = 0
    correct = 0
    for i, data in enumerate(ds, 0):
        inputs, labels = data
        outputs = net(inputs)
        loss += criteria(net(inputs), labels).item()
        if metric:
            metric.update(outputs, labels)

    metric_eval = metric.eval() if metric else None
    return loss / (i + 1), metric_eval


class Accuracy:
    correct = 0
    total = 0

    def update(self, outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

    def eval(self):
        return self.correct / self.total


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