from __future__ import annotations
import time
from pathlib import Path

class NoopLogger:
    def log(self, **kwargs):
        pass
    def close(self):
        pass

class TBLogger(NoopLogger):
    """
    Backward compatible:
        - logger.log(step=.., loss=.., lr=..)
    Extras you can optionally use:
        - logger.hist("params/wte.weight", tensor, step)
        - logger.text("samples/generation", text, step)
        - logger.image("attn/heatmap", HWC_or_CHW_tensor_or_np, step)
        - logger.graph(model, example_batch)
        - logger.hparams(dict_of_config, dict_of_metrics_once)
        - logger.flush()
    Auto-behavior:
        - If a value in .log(...) is a tensor/ndarray with >1 element, it logs a histogram.
        - If key starts with "text/", logs as text.
    """
    def __init__(self, output_dir:str, flush_secs:int=10, run_name:str|None=None):
        pass

    def log(self, step:Optional[int] = None, **kv:Any):
        pass

    def hist(self, tag:str, values:Any, step:Optional[int]=None, bins:str="tensorflow"):
        pass

    def text(self, tag:str, text:str, step: Optional[int] = None):
        pass

    def image(self, tag:str, img, step:Optional[int]=None):
        pass

    def graph(self, model, example_input):
        pass

    def hparams(self, hparams:Dict[str, Any], metrics_once:Optional[Dict[str, float]]= None):
        pass

    def flush(self):
        pass

    def close(self):
        pass

class WBLogger(NoopLogger):
    def __init__(self, project:str, run_name:str|None=None):
        pass

    def log(self, **kv):
        pass

def init_logger(which:str, output_dir:str="runs/part_4"):
    pass