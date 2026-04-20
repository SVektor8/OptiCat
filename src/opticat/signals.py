"""Signal data models used across OptiCat."""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go


@dataclass(frozen=True)
class ElectricalSignal:
    t: any
    s: any

    def plot(self, title="Oscilloscope Trace", xlabel="Time [s]", ylabel="Amplitude"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.t, y=np.real_if_close(self.s), mode='lines'))

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=900,
            height=400,
            margin=dict(l=50, r=10, t=30, b=50),
            template='plotly_white',
            showlegend=False
        )

        fig.show()


@dataclass(frozen=True)
class OpticalSignal:
    A: np.ndarray   # complex envelope A(t)
    t: np.ndarray   # time grid [s]
    f0: float       # carrier frequency [Hz]


@dataclass(frozen=True)
class DetectorSignal:
    t: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    i: np.ndarray
    q: np.ndarray
    power: np.ndarray
