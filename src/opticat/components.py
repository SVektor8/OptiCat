"""Optical/electrical components and plotting helpers."""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from .signals import DetectorSignal, ElectricalSignal, OpticalSignal


class BandPassFilter:
    def __init__(self, R=50, L=10e-9, C=1e-12):
        self.R = R
        self.L = L
        self.C = C

    @property
    def omega0(self):
        return 1.0 / np.sqrt(self.L * self.C)

    @property
    def f0(self):
        return self.omega0 / (2 * np.pi)

    def transfer_function(self, omega):
        # H(omega) = i*omega*R*C / (1 - omega^2*L*C + i*omega*R*C)
        num = 1j * omega * self.R * self.C
        den = 1.0 - (omega ** 2) * self.L * self.C + 1j * omega * self.R * self.C
        return num / den

    def apply(self, signal: ElectricalSignal):
        t, vin = signal.t, signal.s
        dt = t[1] - t[0]
        N = len(vin)

        freq = np.fft.fftfreq(N, d=dt)
        omega = 2 * np.pi * freq
        H = self.transfer_function(omega)

        Vin_f = np.fft.fft(vin)
        Vout_f = Vin_f * H
        vout = np.fft.ifft(Vout_f)
        return ElectricalSignal(t, np.real_if_close(vout))

    def plot_frequency_response(self, fmin=0.0, fmax=None, points=2000):
        if fmax is None:
            fmax = 3 * self.f0

        f = np.linspace(fmin, fmax, points)
        omega = 2 * np.pi * f
        H = self.transfer_function(omega)
        mag = np.abs(H)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=mag, mode='lines', name='|H(f)|'))
        fig.add_vline(x=self.f0, line_dash='dash', annotation_text='f0')
        fig.update_layout(
            xaxis_title='Frequency [Hz]',
            yaxis_title='Amplitude |H(f)|',
            width=800,
            height=600,
            margin=dict(l=50, r=10, t=10, b=50),
            template='plotly_white',
            showlegend=False
        )
        fig.show()

class LowPassFilter:
    def __init__(self, R=50, C=1e-12):
        self.R = R
        self.C = C

    def transfer_function(self, omega):
        # Комплексная ПФ 1-го порядка RC НЧ фильтра (сохраняет фазовый сдвиг)
        #return 1.0 / (1.0 + 1j * omega * self.R * self.C)
        return 1.0 / np.sqrt(1.0 + (omega * self.R * self.C)**2)

    def apply(self, signal: ElectricalSignal):
        t, vin = signal.t, signal.s
        dt = t[1] - t[0]
        N = len(vin)
        freq = np.fft.fftfreq(N, d=dt)
        omega = 2 * np.pi * freq

        H = self.transfer_function(omega)
        Vin_f = np.fft.fft(vin)
        Vout_f = Vin_f * H
        vout = np.fft.ifft(Vout_f)
        return ElectricalSignal(t, np.real_if_close(vout))

    def plot_frequency_response(self, fmax=None, points=1000):
        if fmax is None:
            fmax = 5 / (2 * np.pi * self.R * self.C)
        f = np.linspace(0, fmax, points)
        omega = 2 * np.pi * f
        H = self.transfer_function(omega)
        mag = np.abs(H)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=mag, mode='lines'))
        fig.update_layout(
            xaxis_title='Frequency [Hz]',
            yaxis_title='Amplitude |H(f)|',
            width=800,
            height=600,
            margin=dict(l=50, r=10, t=10, b=50),
            template='plotly_white',
            showlegend=False
        )
        fig.show()


class ElectricalNoiseGenerator:
    def __init__(self, noise_std=0.05, mean=0.0, seed=None):
        self.noise_std = noise_std
        self.mean = mean
        self.rng = np.random.default_rng(seed)

    def apply(self, signal: ElectricalSignal) -> ElectricalSignal:
        noise = self.rng.normal(loc=self.mean, scale=self.noise_std, size=len(signal.s))
        noisy = np.real_if_close(signal.s) + noise
        return ElectricalSignal(signal.t, noisy)

class Oscilloscope:
    def __init__(self,
                 width=800,
                 height=600,
                 plot_modes=None,
                 plot_titles=None,
                 eye_slot_duration=None,
                 eye_bit_rate=None,
                 eye_slots=3,
                 eye_max_traces=300):

        self.width = width
        self.height = height

        # параметры, которые приходят из графа
        self.plot_modes = plot_modes or {}
        self.plot_titles = plot_titles or {}

        # параметры глазковой диаграммы
        self.eye_slot_duration = eye_slot_duration
        self.eye_bit_rate = eye_bit_rate
        self.eye_slots = eye_slots
        self.eye_max_traces = eye_max_traces

    def plot(self, trace, title="Oscilloscope Trace",
             xlabel="Time [s]", ylabel="Amplitude"):

        t = trace.t
        signal = np.real_if_close(trace.s)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=signal, mode='lines'))

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=self.width,
            height=self.height,
            margin=dict(l=50, r=10, t=30, b=50),
            template='plotly_white',
            showlegend=False
        )

        fig.show()

    def plot_detector_amp_phase(self, det, unwrap_phase=True,
                                title_prefix="Detector"):

        phase = np.unwrap(det.phase) if unwrap_phase else det.phase

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=det.t, y=det.amplitude,
                                 mode='lines', name='Amplitude'))
        fig.add_trace(go.Scatter(x=det.t, y=phase,
                                 mode='lines', name='Phase [rad]',
                                 yaxis='y2'))

        fig.update_layout(
            title=f"{title_prefix}: Amplitude & Phase",
            xaxis_title='Time [s]',
            yaxis=dict(title='Amplitude'),
            yaxis2=dict(title='Phase [rad]', overlaying='y', side='right'),
            width=self.width,
            height=self.height,
            margin=dict(l=50, r=50, t=30, b=50),
            template='plotly_white',
            showlegend=True
        )

        fig.show()

    def plot_iq(self, det, title="I/Q Traces"):

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=det.t, y=det.i,
                                 mode='lines', name='I'))
        fig.add_trace(go.Scatter(x=det.t, y=det.q,
                                 mode='lines', name='Q'))

        fig.update_layout(
            title=title,
            xaxis_title='Time [s]',
            yaxis_title='Amplitude',
            width=self.width,
            height=self.height,
            margin=dict(l=50, r=10, t=30, b=50),
            template='plotly_white',
            showlegend=True
        )

        fig.show()

    def plot_eye(self, trace,
                 slot_duration=None,
                 bit_rate=None,
                 slots=None,
                 max_traces=None,
                 title="Eye Diagram"):

        # значения по умолчанию берём из конструктора, если не заданы явно
        if slot_duration is None:
            slot_duration = self.eye_slot_duration
        if bit_rate is None:
            bit_rate = self.eye_bit_rate
        if slots is None:
            slots = self.eye_slots
        if max_traces is None:
            max_traces = self.eye_max_traces

        if slots <= 0:
            raise ValueError("slots должен быть > 0")

        t = np.asarray(trace.t)
        signal = np.real(np.real_if_close(trace.s))
        dt = t[1] - t[0]

        # автоматический подбор slot_duration
        if slot_duration is None:
            if bit_rate is not None:
                slot_duration = 1.0 / bit_rate
            else:
                ds = np.abs(np.diff(signal))
                thr = 0.25 * np.max(ds) if len(ds) > 0 else 0.0
                edge_idx = np.where(ds > thr)[0]

                if len(edge_idx) >= 2:
                    est_samples = int(np.median(np.diff(edge_idx)))
                    slot_duration = max(est_samples, 2) * dt
                else:
                    slot_duration = (t[-1] - t[0]) / max(8, slots + 2)

        if slot_duration <= 0:
            raise ValueError("slot_duration должен быть > 0")

        samples_per_slot = max(int(round(slot_duration / dt)), 2)
        window = slots * samples_per_slot

        if window >= len(signal):
            samples_per_slot = max(len(signal) // (slots + 1), 2)
            window = slots * samples_per_slot

        n_segments = len(signal) // samples_per_slot - slots + 1
        if n_segments <= 0:
            raise ValueError("Сигнал слишком короткий для eye-диаграммы")

        if max_traces is not None:
            n_segments = min(n_segments, int(max_traces))

        x = np.arange(window) * dt

        fig = go.Figure()

        for k in range(n_segments):
            i0 = k * samples_per_slot
            i1 = i0 + window
            seg = signal[i0:i1]

            if len(seg) == window:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=seg,
                    mode='lines',
                    line=dict(width=2, color='blue'),
                    opacity=0.3,
                    hoverinfo='skip',
                    showlegend=False
                ))

        for m in range(1, slots):
            fig.add_vline(x=m * samples_per_slot * dt,
                          line_dash='dot')

        fig.update_layout(
            title=f"{title} ({slots} slots)",
            xaxis_title='Time in eye window [s]',
            yaxis_title='Amplitude',
            width=self.width,
            height=self.height,
            margin=dict(l=50, r=10, t=30, b=50),
            template='plotly_white',
            showlegend=False
        )

        fig.show()

class ArbitraryWaveformGenerator:
    def __init__(self, mode="rec", sampling_rate=1e18, length=1,
                 low=0.0, high=1.0, duty=0.5, amp=1.0, phi=0.0, per=1e-12,
                 sequence=None, pulse_shape="rect", gauss_sigma_frac=0.15, **kwargs):
        self.mode = mode
        self.sampling_rate = sampling_rate
        self.length = length
        self.low = low
        self.high = high
        self.duty = duty
        self.amp = amp
        self.phi = phi
        self.per = per
        self.sequence = np.array(sequence) if sequence is not None else np.array([])
        self.pulse_shape = pulse_shape
        self.gauss_sigma_frac = gauss_sigma_frac

        self.t = None
        self.signal = None

        if self.mode == "seq":
            self.length = len(self.sequence) * self.per

    def _gaussian_pulse(self, tau, center, sigma):
        return np.exp(-0.5 * ((tau - center) / sigma) ** 2)

    def generate(self):
        self.t = np.arange(0, self.length, 1 / self.sampling_rate)
        self.signal = np.zeros_like(self.t)

        if self.mode == "rec":
            phase = (self.t + self.phi * self.per) % self.per
            self.signal = np.where(phase < self.duty * self.per, self.high, self.low)

        elif self.mode == "gauss":
            phase = (self.t + self.phi * self.per) % self.per
            center = self.duty * self.per
            sigma = max(self.gauss_sigma_frac * self.per, 1e-18)
            g = self._gaussian_pulse(phase, center, sigma)
            self.signal = self.low + (self.high - self.low) * g

        elif self.mode == "sin":
            self.signal = self.amp * np.sin(2 * np.pi * self.t / self.per + self.phi)

        elif self.mode == "saw":
            samples_per_period = int(self.per * self.sampling_rate)
            rise_samples = int(samples_per_period * self.duty)
            fall_samples = samples_per_period - rise_samples

            if rise_samples > 0:
                rising = np.linspace(self.low, self.high, rise_samples, endpoint=False)
            else:
                rising = np.array([])
            if fall_samples > 0:
                falling = np.linspace(self.high, self.low, fall_samples, endpoint=False)
            else:
                falling = np.array([])
            one_period = np.concatenate([rising, falling])

            n_repeat = int(np.ceil(len(self.t) / samples_per_period))
            signal_full = np.tile(one_period, n_repeat)
            self.signal = signal_full[:len(self.t)]

            shift_samples = int(self.phi / self.per * samples_per_period)
            self.signal = np.roll(self.signal, shift_samples)

        elif self.mode == "seq":
            if len(self.sequence) == 0:
                raise ValueError("Для режима 'seq' нужно задать sequence")

            self.length = len(self.sequence) * self.per
            self.t = np.arange(0, self.length, 1 / self.sampling_rate)

            samples_per_slot = max(int(self.per * self.sampling_rate), 1)
            signal_full = np.zeros(len(self.t)) + self.low

            for i, bit in enumerate(self.sequence):
                if bit <= 0:
                    continue
                i0 = i * samples_per_slot
                i1 = min((i + 1) * samples_per_slot, len(self.t))
                tau = np.arange(i1 - i0) / self.sampling_rate

                if self.pulse_shape == "gauss":
                    center = self.duty * self.per
                    sigma = max(self.gauss_sigma_frac * self.per, 1e-18)
                    pulse = self._gaussian_pulse(tau, center, sigma)
                    slot_signal = self.low + (self.high - self.low) * pulse
                else:
                    slot_signal = np.ones_like(tau) * self.high

                signal_full[i0:i1] = slot_signal

            self.signal = signal_full
        else:
            raise ValueError("mode должен быть 'rec', 'gauss', 'sin', 'saw' или 'seq'")

        return ElectricalSignal(self.t, self.signal)

    def plot(self):
        if self.signal is None or self.t is None:
            raise ValueError("Сначала вызовите generate() для генерации сигнала")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.t, y=self.signal, mode='lines'))
        fig.update_layout(
            xaxis_title='Time [s]',
            yaxis_title='Amplitude',
            margin=dict(l=50, r=10, t=10, b=50),
            width=800,
            height=600,
            showlegend=False,
            template='plotly_white',
        )
        fig.show()

@dataclass
class Laser:
    wavelength: float
    P0: float = 1.0
    linewidth: float = 0.0
    phi0: float = 0.0

    def generate(self, t: np.ndarray) -> "OpticalSignal":
        c = 3e8
        f0 = c / self.wavelength

        dt = t[1] - t[0]

        if self.linewidth > 0:
            dphi_std = np.sqrt(2 * np.pi * self.linewidth * dt)
            dphi = np.random.normal(0.0, dphi_std, size=len(t))
            phi = self.phi0 + np.cumsum(dphi)
        else:
            phi = self.phi0 + np.zeros_like(t)

        A = np.sqrt(self.P0) * np.exp(1j * phi)

        return OpticalSignal(A=A, t=t, f0=f0)

class IncoherentDetector:
    def __init__(self, responsivity=1.0, expose_phase=True):
        self.R = responsivity
        self.expose_phase = expose_phase

    def detect(self, optical: OpticalSignal) -> DetectorSignal:
        power = self.R * np.abs(optical.A) ** 2
        amplitude = np.sqrt(np.maximum(power, 0.0))

        if self.expose_phase:
            # Для мониторинга берём фазу огибающей из модели поля,
            # физически прямой детектор фазу напрямую не измеряет.
            phase = np.angle(optical.A)
        else:
            phase = np.zeros_like(power)

        i = amplitude * np.cos(phase)
        q = amplitude * np.sin(phase)
        return DetectorSignal(t=optical.t, amplitude=amplitude, phase=phase, i=i, q=q, power=power)

class CoherentDetector:
    def __init__(self, responsivity=1.0, lo_power=1.0, lo_phase=0.0):
        self.R = responsivity
        self.lo_power = lo_power
        self.lo_phase = lo_phase

    def detect(self, optical: OpticalSignal, lo: OpticalSignal = None) -> DetectorSignal:
        if lo is None:
            lo_field = np.sqrt(self.lo_power) * np.exp(1j * self.lo_phase) * np.ones_like(optical.A)
        else:
            if len(lo.A) != len(optical.A):
                raise ValueError("LO и signal должны иметь одинаковую длину")
            lo_field = lo.A

        baseband = self.R * optical.A * np.conj(lo_field)
        i = np.real(baseband)
        q = np.imag(baseband)
        amplitude = np.sqrt(i ** 2 + q ** 2)
        phase = np.arctan2(q, i)
        power = i ** 2 + q ** 2

        return DetectorSignal(t=optical.t, amplitude=amplitude, phase=phase, i=i, q=q, power=power)

class Photodetector:
    def __init__(self, responsivity=1.0):
        self.R = responsivity

    def detect(self, optical: OpticalSignal):
        I = self.R * np.abs(optical.A) ** 2
        return ElectricalSignal(optical.t, I)

class IntensityModulator:
    def __init__(self, Vpi=1.0, bias=0.5):
        self.Vpi = Vpi
        self.bias = bias

    def apply(self, signal: OpticalSignal, drive: ElectricalSignal):
        # Нелинейность MZM через cos-плечо сохраняет комплексную фазу носителя.
        m = np.cos(0.5 * np.pi * (drive.s / self.Vpi + self.bias))
        A_out = signal.A * m
        return OpticalSignal(A=A_out, t=signal.t, f0=signal.f0)

@dataclass
class Fiber:
    alpha_db_per_km: float
    D: float
    wavelength: float
    length: float

    def propagate(self, signal: OpticalSignal) -> OpticalSignal:
        A = signal.A.copy()
        t = signal.t
        dt = t[1] - t[0]
        N = len(A)

        f = np.fft.fftfreq(N, d=dt)
        omega = 2 * np.pi * f

        c = 3e8
        lam = self.wavelength
        D_SI = self.D * 1e-6
        beta2 = -(D_SI * lam ** 2) / (2 * np.pi * c)

        H_disp = np.exp(-1j * 0.5 * beta2 * omega ** 2 * self.length)

        A_w = np.fft.fft(A)
        A_w *= H_disp
        A = np.fft.ifft(A_w)

        alpha_lin = (self.alpha_db_per_km / 4.343) * 1e-3
        A *= np.exp(-alpha_lin * self.length / 2)

        return OpticalSignal(A=A, t=t, f0=signal.f0)




class OpticalSplitter:
    def __init__(self, ratio=0.5):
        self.ratio = float(ratio)

    def split(self, signal: OpticalSignal):
        r = np.clip(self.ratio, 0.0, 1.0)
        a1 = np.sqrt(r) * signal.A
        a2 = np.sqrt(1.0 - r) * signal.A
        return (
            OpticalSignal(A=a1, t=signal.t, f0=signal.f0),
            OpticalSignal(A=a2, t=signal.t, f0=signal.f0),
        )

