from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import torch


class Hook:
    def __init__(self, module, fn):
        self.hook = module.register_forward_hook(partial(fn, self))
        self.layer_name = module.__repr__()

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


class Hooks:
    def __init__(self, model, fn):
        self.hooks = [Hook(layer, fn) for layer in model.layers()]

    def __iter__(self):
        return iter(self.hooks)

    def __del__(self):
        self.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def remove(self):
        for hook in self.hooks:
            hook.hook.remove()


class MomentHooks(Hooks):
    def __init__(self, model):
        super().__init__(model, self.append_stats)

    @staticmethod
    def append_stats(hook, mod, inp, outp):
        if not hasattr(hook, 'stats'):
            hook.stats = ([], [])
        means, stds = hook.stats
        means.append(outp.data.mean())
        stds.append(outp.data.std())

    def plot(self):
        with self as hooks:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 10))
            for h in hooks:
                ms, ss = h.stats
                ax0.plot(ms)
                ax0.set_title('Layer Means')
                ax1.plot(ss)
                ax1.set_title('Layer Standard Deviations')
            plt.legend([hook.layer_name for hook in hooks], loc="upper left", bbox_to_anchor=(-1, -.05))


class HistHooks(Hooks):
    def __init__(self, model):
        super().__init__(model, self.append_stats)

    @staticmethod
    def append_stats(hook, mod, inp, outp):
        if not hasattr(hook, 'stats'):
            hook.stats = ([], [], [])
        means, stds, hists = hook.stats
        means.append(outp.data.mean().cpu())
        stds.append(outp.data.std().cpu())
        hists.append(outp.data.cpu().histc(40, 0, 10))

    @staticmethod
    def get_hist(h):
        return torch.stack(h.stats[2]).t().float().log1p()

    @staticmethod
    def get_min(h):
        h1 = torch.stack(h.stats[2]).t().float()
        return h1[:2].sum(0) / h1.sum(0)

    def plot_hists(self):
        n_layers = len(self.hooks)
        with self as hooks:
            fig, axes = plt.subplots(round(n_layers / 2), 2, figsize=(15, n_layers * 4))
            for ax, h in zip(axes.flatten(), hooks):
                ax.set_title(h.layer_name)
                ax.imshow(self.get_hist(h), origin='lower')
                ax.axis('off')

    def plot_dead_activations(self):
        n_layers = len(self.hooks)
        with self as hooks:
            fig, axes = plt.subplots(round(n_layers / 2), 2, figsize=(15, n_layers * 4))
            for ax, h in zip(axes.flatten(), hooks):
                ax.set_title(h.layer_name)
                ax.plot(np.array(self.get_min(h)))
                ax.set_ylim(0, 1)
