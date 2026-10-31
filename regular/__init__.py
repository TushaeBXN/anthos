"""The regular model: a plain decoder-only transformer. This is the active foundation; the Anthos architecture in
anthos/ is paused (kept intact, not deleted)."""
from .model import PRESETS, Regular, RegularConfig

__all__ = ["PRESETS", "Regular", "RegularConfig"]
