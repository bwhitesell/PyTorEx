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
