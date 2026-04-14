import numpy as np
import plotly.graph_objects as go


class ArbitraryWaveformGenerator:
    def __init__(self, mode="rec", sampling_rate=1e18, length=1,
                 low=0.0, high=1.0, duty=0.5, amp=1.0, phi=0.0, per=1e-12,
                 sequence=None, **kwargs):
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

        self.t = None
        self.signal = None

        if self.mode == "seq":
            self.length = len(self.sequence) * self.per

    def generate(self):
        # создаём временную ось
        self.t = np.arange(0, self.length, 1 / self.sampling_rate)
        self.signal = np.zeros_like(self.t)

        if self.mode == "rec":
            # квадратный сигнал: level = high в течение duty*per, else low
            phase = (self.t + self.phi * self.per) % self.per
            self.signal = np.where(phase < self.duty * self.per, self.high, self.low)

        elif self.mode == "sin":
            # синус: amp * sin(2*pi*t/per + phi)
            self.signal = self.amp * np.sin(2 * np.pi * self.t / self.per + self.phi)

        elif self.mode == "saw":
            # Количество сэмплов на период
            samples_per_period = int(self.per * self.sampling_rate)
            rise_samples = int(samples_per_period * self.duty)
            fall_samples = samples_per_period - rise_samples

            # Генерируем один период
            if rise_samples > 0:
                rising = np.linspace(self.low, self.high, rise_samples, endpoint=False)
            else:
                rising = np.array([])
            if fall_samples > 0:
                falling = np.linspace(self.high, self.low, fall_samples, endpoint=False)
            else:
                falling = np.array([])
            one_period = np.concatenate([rising, falling])

            # Повторяем период до длины сигнала
            n_repeat = int(np.ceil(len(self.t) / samples_per_period))
            signal_full = np.tile(one_period, n_repeat)
            self.signal = signal_full[:len(self.t)]

            # Сдвиг по phi (фаза)
            shift_samples = int(self.phi / self.per * samples_per_period)
            self.signal = np.roll(self.signal, shift_samples)


        elif self.mode == "seq":

            if len(self.sequence) == 0:
                raise ValueError("Для режима 'seq' нужно задать sequence")

            self.length = len(self.sequence) * self.per
            self.t = np.arange(0, self.length, 1 / self.sampling_rate)

            samples_per_slot = max(int(self.per * self.sampling_rate), 1)

            signal_full = np.repeat(self.sequence, samples_per_slot)


            self.signal = np.where(signal_full[:len(self.t)] > 0, self.high, self.low)
        else:
            raise ValueError("mode должен быть 'rec', 'sin', 'saw' или 'seq'")

        return self.signal

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