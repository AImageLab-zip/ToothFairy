"""Training — shared epoch loop, align->generate two-stage schedule, reproducibility."""
from .trainer import train, build_loaders, evaluate, save_checkpoint, seed_everything  # noqa: F401
