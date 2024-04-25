from tqdm import tqdm
import lightning as L

# Class for preventing the wall of text
# https://github.com/Lightning-AI/pytorch-lightning/issues/15283
class LitProgressBar(L.pytorch.callbacks.TQDMProgressBar):
    def init_validation_tqdm(self):
        # bar = super().init_validation_tqdm()
        bar = tqdm(disable=True,)
        # bar.disable = True
        return bar

    def init_train_tqdm(self):
        bar = tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        bar.bar_format ='{desc} [{rate_fmt}{postfix}]'
        return bar