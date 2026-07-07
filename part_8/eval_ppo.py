from __future__ import annotations
import argparse, torch
from pathlib import Path
from policy import PolicyWithValue
from rollout import RLHFTokenizer, sample_prompts, format_prompt_only

# Reward model
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel

def score_policy(policy_ckpt:str, rm_ckpt:str, bpe_sir:str|None, n:int=16):
    pass

if __name__ == '__main__':
    pass