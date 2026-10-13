"""Post-processing of generated text — strips the chat scaffolding.

This has to be the **single canonical implementation**: the string scored by the RL reward
during training and the string scored by evaluation (`pipeline/generate.py`) must be *exactly* the
same, otherwise what training improves and what evaluation measures drift apart. Keeping a
second copy anywhere means one of them gets fixed and the other silently diverges.
"""
from __future__ import annotations

import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)          # drop the thinking channel
_CHAT_TOK = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>")
_ROLE = re.compile(r"^\s*(assistant|user|system)\b", re.IGNORECASE)


def clean_gen(txt: str) -> str:
    """Strip chat scaffolding from generated text: <think></think>, special tokens, leading role
    marker. The model's config.eos_token_id is None, so if generation does not stop at the turn
    end this prevents the next turn leaking in (used together with an explicit eos)."""
    txt = _THINK.sub(" ", txt)
    txt = _CHAT_TOK.sub(" ", txt)
    txt = _ROLE.sub("", txt.strip())
    # Cut everything after im_end (anything leaked from the next turn)
    for marker in ("<|im_start|>", "\nuser\n", "\nassistant\n", "\nsystem\n"):
        if marker in txt:
            txt = txt.split(marker)[0]
    return " ".join(txt.replace("\n", " ").split()).strip()
