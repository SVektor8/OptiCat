"""OptiCat package."""

from .components import (
    ArbitraryWaveformGenerator,
    BandPassFilter,
    CoherentDetector,
    ElectricalNoiseGenerator,
    Fiber,
    IncoherentDetector,
    IntensityModulator,
    Laser,
    LowPassFilter,
    OpticalSplitter,
    Oscilloscope,
    Photodetector,
)
from .core import SuperMan
from .gui import SuperCat
from .signals import DetectorSignal, ElectricalSignal, OpticalSignal

__all__ = [
    "ArbitraryWaveformGenerator",
    "BandPassFilter",
    "CoherentDetector",
    "DetectorSignal",
    "ElectricalNoiseGenerator",
    "ElectricalSignal",
    "Fiber",
    "IncoherentDetector",
    "IntensityModulator",
    "Laser",
    "LowPassFilter",
    "OpticalSignal",
    "OpticalSplitter",
    "Oscilloscope",
    "Photodetector",
    "SuperCat",
    "SuperMan",
]
