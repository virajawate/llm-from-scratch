from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]/'part_3'))
import time
import torch
import shutil
import torch.nn as nn

DEF_NAME = 'model_last.pt'

def _is_tb(logger) -> bool:
    return getattr(logger, "w", None) is not None

def 