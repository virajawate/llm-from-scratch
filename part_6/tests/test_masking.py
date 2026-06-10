from collator_sft import SFTCollator
from formatters import Example

def test_masking_sets_prompt_to_ignore():
    col = SFTCollator(block_size=256, bpe_dir="../part_4/runs/part4-demo/tokenizer")
    text = "This is a tiny test."
    x, y = col.collate([(text, "OK")])
    boundary = (f"<s>\n## Instruction : \n{text}\n\n### Response : \n")
    assert (y[0] == -100).sum() > 0