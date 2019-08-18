class MakeRGB:
    _order = 0
    def __call__(self, item): return item.convert('RGB')
