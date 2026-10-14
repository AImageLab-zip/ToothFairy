"""Inference-only shim; the notebook never logs to Weights & Biases."""
class _Config:
    def update(self, *_args, **_kwargs):
        return None
config = _Config()
def init(*_args, **_kwargs):
    return None
def log(*_args, **_kwargs):
    return None
def finish(*_args, **_kwargs):
    return None
