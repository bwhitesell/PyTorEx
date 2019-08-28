import gensim
import torch


class MakeRGB:
    _order = 0
    def __call__(self, item): return item.convert('RGB')


def get_word_2_vec_weights():
    model = gensim.models.KeyedVectors.load_word2vec_format('path/to/file')
    return torch.FloatTensor(model.vectors)