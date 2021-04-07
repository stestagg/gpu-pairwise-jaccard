from . import metric as _metric


def pairwise_distance(values, metric='jaccard', **kwargs):
    metric_cls = _metric.Metric.get_metric(metric)
    inst = metric_cls(**kwargs)
    return inst.transform(values)

from . import boolean, geometric