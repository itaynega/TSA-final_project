import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        # `accelerator` is resolved once in run.py to one of cuda / mps / cpu.
        # Upstream only knew about cuda, which left no way to run on Apple
        # Silicon or on a CPU-only machine.
        accelerator = getattr(self.args, 'accelerator', 'cuda' if self.args.use_gpu else 'cpu')

        if accelerator == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif accelerator == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps (Apple Silicon)')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
