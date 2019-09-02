class NormalizeCols:
    norms = {}
    n_cols = 0

    def __init__(self, data):
        if len(data.shape) == 2:
            self.n_cols = data.shape[1]
        else:
            raise ValueError("Can only Normalize Matrices.")

        for col in range(data.shape[1]):
            self.norms[col] = [data[:, col].mean(), data[:, col].std()]

    def apply_norm(self, data):
        for col in range(self.n_cols):
            data[:, col] = (data[:, col] - self.norms[col][0]) / self.norms[col][1]
        return data


def create_mapping(series):
    mpng = {}
    for t, cat in enumerate(series.unique()):
        mpng[cat] = t
    return mpng


class CategoricalMapping:

    def __init__(self, series):
        self.mapping = self.create_mapping(series)

    def map(self, series):
        return series.apply(lambda x: self.mapping[x])

    @staticmethod
    def create_mapping(series):
        mapping = {}
        for t, cat in enumerate(series.unique()):
            mapping[cat] = t
        return mapping
