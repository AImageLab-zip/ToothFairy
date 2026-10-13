"""Generation — contextual anatomical tokens -> LLM -> per-region report text.

Parts:
  llm_backend — build_llm_backend, VisualPrefixProjector (the visual prefix), chat helpers
  realizer    — Realizer (region-conditioned per-structure decode)
"""
from .llm_backend import (  # noqa: F401
    build_llm_backend, VisualPrefixProjector,
    chat_text_ids, assemble_chat_sample, assemble_chat_prompt, pad_and_stack,
)
from .realizer import (  # noqa: F401
    Realizer, REGION_TYPES, N_REGION_TYPES, region_type_id, region_instruction,
)
