"""Video-frame preprocessing contracts for model-ready tensors."""

from qpu_xla.video.headless import HeadlessPreprocessReport, Nv12Frame, run_headless_nv12_preprocessing
from qpu_xla.video.preprocess import preprocess_nv12_to_nchw_fp32
from qpu_xla.video.ring import Nv12FrameRing, Nv12FrameSlot

__all__ = [
    "HeadlessPreprocessReport",
    "Nv12Frame",
    "Nv12FrameRing",
    "Nv12FrameSlot",
    "preprocess_nv12_to_nchw_fp32",
    "run_headless_nv12_preprocessing",
]
