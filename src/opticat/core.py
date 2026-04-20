"""Interactive graph editor and simulation core."""

import ast
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython import get_ipython
from IPython.display import clear_output, display
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    import ipywidgets as widgets
except Exception:
    widgets = None

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
from .signals import DetectorSignal, ElectricalSignal, OpticalSignal


class SuperMan:
    BOX_W = 0.18
    BOX_H = 0.09
    PORT_R = 0.0075

    def _ensure_interactive_backend(self):
        backend = matplotlib.get_backend().lower()
        interactive = any(k in backend for k in ("widget", "nbagg", "notebook", "qt", "tk", "macosx"))
        if interactive:
            return True

        ip = get_ipython()
        if ip is not None:
            for mode in ("widget", "notebook"):
                try:
                    ip.run_line_magic("matplotlib", mode)
                    backend = matplotlib.get_backend().lower()
                    interactive = any(k in backend for k in ("widget", "nbagg", "notebook", "qt", "tk", "macosx"))
                    if interactive:
                        return True
                except Exception:
                    continue

        self._backend_warning = (
            "Нужен интерактивный backend matplotlib для drag&drop. "
            "Установите ipympl (`python3 -m pip install ipympl`) и перезапустите kernel."
        )
        return False

    def __init__(self, use_notebook_backend=True):
        if widgets is None:
            raise ImportError("Для SuperMan нужен ipywidgets. Установите: python3 -m pip install ipywidgets")

        self._backend_warning = ""
        self._interactive_backend = self._ensure_interactive_backend() if use_notebook_backend else True

        self.inline_show = True
        self.show_ids = True
        self.pan_mode = False
        self.delete_edge_mode = False

        self.nodes = {}
        self.edges = []
        self._node_counter = {}

        self.selected_node = None
        self.selected_port = None
        self._dragging = None
        self._drag_offset = (0.0, 0.0)
        self._connect_from = None
        self._panning = False
        self._pan_start = None
        self._pan_start_px = None
        self._pan_xlim = None
        self._pan_ylim = None

        self._last_outputs = {}
        self._figures = []
        self.logs = []

        self.device_specs = {
            "ArbitraryWaveformGenerator": {
                "class": ArbitraryWaveformGenerator,
                "defaults": {
                    "mode": "seq",
                    "sampling_rate": 5e13,
                    "per": 1e-10,
                    "sequence": [1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0],
                    "pulse_shape": "rec",
                    "gauss_sigma_frac": 0.18,
                    "low": 0.0,
                    "high": 1.0,
                },
                "in_ports": [],
                "out_ports": [("el_out", "ElOut", "el")],
                "runner": self._run_awg,
            },
            "Laser": {
                "class": Laser,
                "defaults": {"wavelength": 1550e-9, "P0": 1.0, "linewidth": 0.0, "phi0": 0.0},
                "in_ports": [],
                "out_ports": [("opt_out", "OptOut", "opt")],
                "runner": self._run_laser,
            },
            "IntensityModulator": {
                "class": IntensityModulator,
                "defaults": {"Vpi": 1.0, "bias": 0.5},
                "in_ports": [("opt_in", "OptIn", "opt"), ("el_in", "ElIn", "el")],
                "out_ports": [("opt_out", "OptOut", "opt")],
                "runner": self._run_mzm,
            },
            "OpticalSplitter": {
                "class": OpticalSplitter,
                "defaults": {"ratio": 0.5},
                "in_ports": [("opt_in", "OptIn", "opt")],
                "out_ports": [("opt_out1", "OptOut1", "opt"), ("opt_out2", "OptOut2", "opt")],
                "runner": self._run_splitter,
            },
            "Fiber": {
                "class": Fiber,
                "defaults": {"alpha_db_per_km": 0.2, "D": 16.0, "wavelength": 1550e-9, "length": 5e3},
                "in_ports": [("opt_in", "OptIn", "opt")],
                "out_ports": [("opt_out", "OptOut", "opt")],
                "runner": self._run_fiber,
            },
            "Photodetector": {
                "class": Photodetector,
                "defaults": {"responsivity": 1.0},
                "in_ports": [("opt_in", "OptIn", "opt")],
                "out_ports": [("el_out", "ElOut", "el")],
                "runner": self._run_photodetector,
            },
            "BandPassFilter": {
                "class": BandPassFilter,
                "defaults": {"R": 50.0, "L": 10e-9, "C": 1e-12},
                "in_ports": [("el_in", "ElIn", "el")],
                "out_ports": [("el_out", "ElOut", "el")],
                "runner": self._run_apply_single,
            },
            "LowPassFilter": {
                "class": LowPassFilter,
                "defaults": {"R": 50.0, "C": 45e-15},
                "in_ports": [("el_in", "ElIn", "el")],
                "out_ports": [("el_out", "ElOut", "el")],
                "runner": self._run_apply_single,
            },
            "ElectricalNoiseGenerator": {
                "class": ElectricalNoiseGenerator,
                "defaults": {"noise_std": 2.0, "mean": 0.0, "seed": 42},
                "in_ports": [("el_in", "ElIn", "el")],
                "out_ports": [("el_out", "ElOut", "el")],
                "runner": self._run_apply_single,
            },
            "IncoherentDetector": {
                "class": IncoherentDetector,
                "defaults": {"responsivity": 1.0, "expose_phase": True},
                "in_ports": [("opt_in", "OptIn", "opt")],
                "out_ports": [("det_out", "DetOut", "det")],
                "runner": self._run_detect_single,
            },
            "CoherentDetector": {
                "class": CoherentDetector,
                "defaults": {"responsivity": 1.0, "lo_power": 1.0, "lo_phase": 0.0},
                "in_ports": [("opt_in", "OptIn", "opt")],
                "out_ports": [("det_out", "DetOut", "det")],
                "runner": self._run_detect_single,
            },
            "Oscilloscope": {
                "class": Oscilloscope,
                "defaults": {
                    "width": 900,
                    "height": 450,
                    "plot_modes": {
                        "time1": "time", "time2": "time", "time3": "time", "time4": "time", "time5": "time", "time6": "time",
                        "eye1": "eye", "eye2": "eye", "eye3": "eye", "eye4": "eye", "eye5": "eye", "eye6": "eye",
                        "custom1": "auto", "custom2": "auto", "custom3": "auto", "custom4": "auto",
                    },
                    "plot_titles": {
                        "time1": "AWG (Clean)",
                        "time2": "After Noise Generator",
                        "time3": "After LowPassFilter",
                        "time4": "Time4",
                        "time5": "Time5",
                        "time6": "Time6",
                        "eye1": "Eye Diagram @ 7GHz LPF",
                        "eye2": "Eye Diagram @ Final LPF",
                        "eye3": "Eye3",
                        "eye4": "Eye4",
                        "eye5": "Eye5",
                        "eye6": "Eye6",
                        "custom1": "Photodetector Output",
                        "custom2": "Custom2",
                        "custom3": "Custom3",
                        "custom4": "Custom4",
                    },
                    "eye_slot_duration": 100e-12,
                    "eye_bit_rate": None,
                    "eye_slots": 3,
                    "eye_max_traces": 80,
                },
                "in_ports": [
                    ("time1", "T1", "any", "top"), ("time2", "T2", "any", "top"), ("time3", "T3", "any", "top"),
                    ("time4", "T4", "any", "top"), ("time5", "T5", "any", "top"), ("time6", "T6", "any", "top"),
                    ("eye1", "E1", "any", "bottom"), ("eye2", "E2", "any", "bottom"), ("eye3", "E3", "any", "bottom"),
                    ("eye4", "E4", "any", "bottom"), ("eye5", "E5", "any", "bottom"), ("eye6", "E6", "any", "bottom"),
                    ("custom1", "C1", "any", "left"), ("custom2", "C2", "any", "left"), ("custom3", "C3", "any", "left"), ("custom4", "C4", "any", "left"),
                ],
                "out_ports": [],
                "runner": self._run_scope,
            },
        }

        self._build_widgets()
        self._build_canvas()
        self._load_default_demo()

    def _log(self, level, message):
        ts = datetime.now().strftime("%H:%M:%S")
        self.logs.append({"time": ts, "level": level, "message": str(message)})
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]

    def _load_default_demo(self):
        self.nodes.clear()
        self.edges.clear()
        self._node_counter.clear()

        self.nodes["awg"] = {
            "id": "awg", "type": "ArbitraryWaveformGenerator",
            "params": {
                "mode": "seq", "sampling_rate": 5e13, "per": 1e-10,
                "sequence": [1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0],
                "pulse_shape": "rec", "gauss_sigma_frac": 0.18, "low": 0.0, "high": 1.0,
            },
            "x": 0.07, "y": 0.80,
        }
        self.nodes["lpf7"] = {"id": "lpf7", "type": "LowPassFilter", "params": {"R": 50, "C": 455e-15}, "x": 0.28, "y": 0.80}
        self.nodes["ng"] = {"id": "ng", "type": "ElectricalNoiseGenerator", "params": {"noise_std": 2, "mean": 0.0, "seed": 42}, "x": 0.49, "y": 0.80}
        self.nodes["lpf1"] = {"id": "lpf1", "type": "LowPassFilter", "params": {"R": 50, "C": 45e-15}, "x": 0.70, "y": 0.80}
        self.nodes["osc"] = {
            "id": "osc", "type": "Oscilloscope",
            "params": {
                "width": 900, "height": 450,
                "plot_modes": {"time1": "time", "time2": "time", "time3": "time", "time4": "time", "time5": "time", "time6": "time", "eye1": "eye", "eye2": "eye", "eye3": "eye", "eye4": "eye", "eye5": "eye", "eye6": "eye", "custom1": "auto", "custom2": "auto", "custom3": "auto", "custom4": "auto"},
                "plot_titles": {
                    "time1": "AWG (Clean)", "time2": "After Noise Generator", "time3": "After LowPassFilter",
                    "eye1": "Eye Diagram @ 7GHz LPF", "eye2": "Eye Diagram @ Final LPF", "custom1": "Photodetector Output",
                },
                "eye_slot_duration": 100e-12, "eye_bit_rate": None, "eye_slots": 3, "eye_max_traces": 80,
            },
            "x": 0.70, "y": 0.60,
        }
        self.nodes["laser"] = {"id": "laser", "type": "Laser", "params": {"wavelength": 1550e-9, "P0": 1.0, "linewidth": 0.0, "phi0": 0.0}, "x": 0.07, "y": 0.36}
        self.nodes["mzm"] = {"id": "mzm", "type": "IntensityModulator", "params": {"Vpi": 1.0, "bias": 0.5}, "x": 0.32, "y": 0.33}
        self.nodes["pd"] = {"id": "pd", "type": "Photodetector", "params": {"responsivity": 1.0}, "x": 0.57, "y": 0.33}

        def edge(sn, sp, dn, dp):
            self.edges.append({"src_node": sn, "src_port": sp, "dst_node": dn, "dst_port": dp})

        edge("awg", "el_out", "lpf7", "el_in")
        edge("lpf7", "el_out", "ng", "el_in")
        edge("ng", "el_out", "lpf1", "el_in")

        edge("awg", "el_out", "osc", "time1")
        edge("ng", "el_out", "osc", "time2")
        edge("lpf1", "el_out", "osc", "time3")
        edge("lpf7", "el_out", "osc", "eye1")
        edge("lpf1", "el_out", "osc", "eye2")

        edge("laser", "opt_out", "mzm", "opt_in")
        edge("lpf1", "el_out", "mzm", "el_in")
        edge("mzm", "opt_out", "pd", "opt_in")
        edge("pd", "el_out", "osc", "custom1")

        self.selected_node = "awg"
        self.selected_port = None
        self._sync_code_boxes()
        self._log("INFO", "Loaded default demo scheme")

    def _build_widgets(self):
        self.device_select = widgets.Dropdown(options=list(self.device_specs.keys()), description="Прибор")
        self.add_btn = widgets.Button(description="Добавить", button_style="success")
        self.connect_btn = widgets.ToggleButton(description="Connect", value=False)
        self.delete_edge_btn = widgets.ToggleButton(description="Delete edge", value=False)
        self.pan_btn = widgets.ToggleButton(description="Pan", value=False)
        self.del_btn = widgets.Button(description="Удалить узел", button_style="warning")
        self.run_btn = widgets.Button(description="Run", button_style="primary")
        self.zoom_in_btn = widgets.Button(description="+")
        self.zoom_out_btn = widgets.Button(description="-")
        self.reset_zoom_btn = widgets.Button(description="r")
        self.show_ids_toggle = widgets.Checkbox(value=True, description="Show IDs")

        self.save_checkbox = widgets.Checkbox(value=False, description="Сохранять графики")
        self.save_dir = widgets.Text(value="plots", description="Папка")

        self.node_code = widgets.Textarea(value="", layout=widgets.Layout(width="100%", height="120px"), description="Node code")
        self.apply_code_btn = widgets.Button(description="Применить код", button_style="info")
        self.scheme_code = widgets.Textarea(value="", layout=widgets.Layout(width="100%", height="190px"), description="Scheme", disabled=True)

        self.status = widgets.HTML(value="<b>Status:</b> ready")
        self.output = widgets.Output()

        self.add_btn.on_click(self._on_add)
        self.del_btn.on_click(self._on_delete)
        self.connect_btn.observe(self._on_connect_toggle, names="value")
        self.delete_edge_btn.observe(self._on_delete_edge_toggle, names="value")
        self.pan_btn.observe(self._on_pan_toggle, names="value")
        self.apply_code_btn.on_click(self._on_apply_node_code)
        self.run_btn.on_click(self._on_run)
        self.zoom_in_btn.on_click(lambda _: self._zoom(0.9))
        self.zoom_out_btn.on_click(lambda _: self._zoom(1.1))
        self.reset_zoom_btn.on_click(self._on_reset_zoom)
        self.show_ids_toggle.observe(self._on_show_ids_toggle, names="value")

    def _build_canvas(self):
        self.fig, self.ax = plt.subplots(figsize=(13, 7))
        self.default_xlim = (0.0, 1.0)
        self.default_ylim = (0.0, 1.0)
        self.ax.set_xlim(*self.default_xlim)
        self.ax.set_ylim(*self.default_ylim)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("SuperMan: drag приборы, connect по портам")
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

    def ui(self):
        toolbar = widgets.HBox([
            self.device_select, self.add_btn, self.connect_btn, self.delete_edge_btn, self.pan_btn,
            self.del_btn, self.run_btn, self.zoom_in_btn, self.zoom_out_btn, self.reset_zoom_btn,
            self.show_ids_toggle, self.save_checkbox, self.save_dir,
        ])
        editor = widgets.VBox([self.node_code, self.apply_code_btn, self.scheme_code, self.status])
        display(toolbar)
        if self._backend_warning:
            display(widgets.HTML(value=f"<span style='color:#b00020'><b>Warning:</b> {self._backend_warning}</span>"))
        display(editor)
        display(self.output)
        display(self.fig)
        self._redraw()

    def _on_show_ids_toggle(self, change):
        self.show_ids = bool(change["new"])
        self._redraw()

    def _on_delete_edge_toggle(self, change):
        self.delete_edge_mode = bool(change["new"])
        if self.delete_edge_mode:
            self.connect_btn.value = False
        self.status.value = "<b>Status:</b> delete-edge mode ON" if self.delete_edge_mode else "<b>Status:</b> delete-edge mode OFF"

    def _on_pan_toggle(self, change):
        self.pan_mode = bool(change["new"])
        self.status.value = "<b>Status:</b> pan mode ON" if self.pan_mode else "<b>Status:</b> pan mode OFF"

    def _zoom(self, scale):
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        cx = (cur_xlim[0] + cur_xlim[1]) / 2
        cy = (cur_ylim[0] + cur_ylim[1]) / 2
        new_w = (cur_xlim[1] - cur_xlim[0]) * scale
        new_h = (cur_ylim[1] - cur_ylim[0]) * scale
        self.ax.set_xlim(cx - new_w / 2, cx + new_w / 2)
        self.ax.set_ylim(cy - new_h / 2, cy + new_h / 2)
        self.fig.canvas.draw_idle()

    def _on_reset_zoom(self, _):
        self.ax.set_xlim(*self.default_xlim)
        self.ax.set_ylim(*self.default_ylim)
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self._zoom(0.9 if (event.button == "up") else 1.1)

    def _new_node_id(self, device_name):
        base = ''.join(ch for ch in device_name if ch.isupper()).lower() or device_name.lower()
        self._node_counter[base] = self._node_counter.get(base, 0) + 1
        return f"{base}{self._node_counter[base]}"

    def _short_label(self, device_type):
        return {
            "ArbitraryWaveformGenerator": "AWG", "Laser": "LAS", "IntensityModulator": "MZM",
            "OpticalSplitter": "SPL", "Fiber": "FIB", "Photodetector": "PD",
            "BandPassFilter": "BPF", "LowPassFilter": "LPF", "ElectricalNoiseGenerator": "NSE",
            "IncoherentDetector": "IDT", "CoherentDetector": "CDT", "Oscilloscope": "OSC",
        }.get(device_type, device_type[:6].upper())

    def _port_type_ok(self, out_type, in_type):
        return out_type == in_type or out_type == "any" or in_type == "any"

    def _node_ports(self, node_id):
        node = self.nodes[node_id]
        spec = self.device_specs[node["type"]]
        in_ports = spec.get("in_ports", [])
        out_ports = spec.get("out_ports", [])
        ports = []

        def parse_meta(meta, default_side):
            if len(meta) >= 4:
                return meta[0], meta[1], meta[2], meta[3]
            return meta[0], meta[1], meta[2], default_side

        def positions(count, a, b):
            if count <= 0:
                return []
            if count == 1:
                return [(a + b) / 2]
            step = (b - a) / (count + 1)
            return [a + step * (i + 1) for i in range(count)]

        def place(group, direction):
            for side in ("left", "right", "top", "bottom"):
                subset = [m for m in group if parse_meta(m, "left" if direction == "in" else "right")[3] == side]
                if not subset:
                    continue

                if side in ("left", "right"):
                    ys = positions(len(subset), node["y"], node["y"] + self.BOX_H)
                    for meta, py in zip(subset, ys):
                        pid, label, ptype, _ = parse_meta(meta, "left" if direction == "in" else "right")
                        px = node["x"] if side == "left" else node["x"] + self.BOX_W
                        ports.append({"node": node_id, "id": pid, "label": label, "type": ptype, "dir": direction, "x": px, "y": py, "side": side})
                else:
                    xs = positions(len(subset), node["x"], node["x"] + self.BOX_W)
                    for meta, px in zip(subset, xs):
                        pid, label, ptype, _ = parse_meta(meta, "left" if direction == "in" else "right")
                        py = node["y"] + self.BOX_H if side == "top" else node["y"]
                        ports.append({"node": node_id, "id": pid, "label": label, "type": ptype, "dir": direction, "x": px, "y": py, "side": side})

        place(in_ports, "in")
        place(out_ports, "out")
        return ports

    def _port_xy(self, node_id, port_id, direction):
        for p in self._node_ports(node_id):
            if p["id"] == port_id and p["dir"] == direction:
                return p["x"], p["y"]
        return None, None

    def _hit_test_node(self, x, y):
        if x is None or y is None:
            return None
        for node_id, node in reversed(list(self.nodes.items())):
            if node["x"] <= x <= node["x"] + self.BOX_W and node["y"] <= y <= node["y"] + self.BOX_H:
                return node_id
        return None

    def _hit_test_port(self, x, y):
        if x is None or y is None:
            return None
        rr = (self.PORT_R * 1.8) ** 2
        for node_id in reversed(list(self.nodes.keys())):
            for p in self._node_ports(node_id):
                if (x - p["x"]) ** 2 + (y - p["y"]) ** 2 <= rr:
                    return p
        return None

    @staticmethod
    def _dist_point_to_segment(px, py, x1, y1, x2, y2):
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return math.hypot(px - x1, py - y1)
        c2 = vx * vx + vy * vy
        if c2 <= c1:
            return math.hypot(px - x2, py - y2)
        b = c1 / c2
        bx, by = x1 + b * vx, y1 + b * vy
        return math.hypot(px - bx, py - by)

    def _hit_test_edge(self, x, y, threshold=0.02):
        if x is None or y is None:
            return None
        best = None
        best_d = 1e9
        for e in self.edges:
            x1, y1 = self._port_xy(e["src_node"], e["src_port"], "out")
            x2, y2 = self._port_xy(e["dst_node"], e["dst_port"], "in")
            if x1 is None or x2 is None:
                continue
            d = self._dist_point_to_segment(x, y, x1, y1, x2, y2)
            if d < best_d:
                best_d = d
                best = e
        return best if best_d <= threshold else None

    def _remove_edge(self, edge):
        if edge in self.edges:
            self.edges.remove(edge)
            self.status.value = f"<b>Status:</b> удалена связь {edge['src_node']}.{edge['src_port']} -> {edge['dst_node']}.{edge['dst_port']}"
            self._log("INFO", f"Edge removed: {edge['src_node']}.{edge['src_port']} -> {edge['dst_node']}.{edge['dst_port']}")
            self._sync_code_boxes()
            self._redraw()

    def _on_add(self, _):
        device_name = self.device_select.value
        node_id = self._new_node_id(device_name)
        defaults = dict(self.device_specs[device_name]["defaults"])
        n = len(self.nodes)
        x = 0.06 + (n % 4) * 0.21
        y = 0.84 - (n // 4) * 0.14
        y = max(0.06, y)
        self.nodes[node_id] = {"id": node_id, "type": device_name, "params": defaults, "x": x, "y": y}
        self.selected_node = node_id
        self.selected_port = None
        self.status.value = f"<b>Status:</b> добавлен {node_id} ({device_name})"
        self._log("INFO", f"Node added: {node_id} ({device_name})")
        self._sync_code_boxes()
        self._redraw()

    def _on_delete(self, _):
        if not self.selected_node:
            self.status.value = "<b>Status:</b> выберите прибор для удаления"
            self._log("WARN", "Delete node requested but nothing selected")
            return
        node_id = self.selected_node
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges if e["src_node"] != node_id and e["dst_node"] != node_id]
        self.selected_node = None
        self.selected_port = None
        self._connect_from = None
        self.status.value = f"<b>Status:</b> удален {node_id}"
        self._log("INFO", f"Node removed: {node_id}")
        self._sync_code_boxes()
        self._redraw()

    def _on_connect_toggle(self, change):
        self._connect_from = None
        self.selected_port = None
        if bool(change["new"]):
            self.delete_edge_mode = False
            if hasattr(self, 'delete_edge_btn'):
                self.delete_edge_btn.value = False
            self.status.value = "<b>Status:</b> connect mode ON: OUT -> IN"
        else:
            self.status.value = "<b>Status:</b> connect mode OFF"
        self._redraw()

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return

        if self.delete_edge_mode and event.button == 1:
            hit = self._hit_test_edge(event.xdata, event.ydata)
            if hit is not None:
                self._remove_edge(hit)
                return

        if event.button == 3:
            hit = self._hit_test_edge(event.xdata, event.ydata)
            if hit is not None:
                self._remove_edge(hit)
                return

        if self.pan_mode and event.button == 1:
            self._panning = True
            self._pan_start = (event.xdata, event.ydata)
            self._pan_start_px = (event.x, event.y)
            self._pan_xlim = self.ax.get_xlim()
            self._pan_ylim = self.ax.get_ylim()
            return

        port = self._hit_test_port(event.xdata, event.ydata)
        if self.connect_btn.value:
            if port is None:
                return
            self.selected_node = port["node"]
            self.selected_port = (port["node"], port["id"], port["dir"])
            self._sync_code_boxes()

            if self._connect_from is None:
                if port["dir"] != "out":
                    self.status.value = "<b>Status:</b> первым выберите OUT"
                    self._log("WARN", "Connect mode: first port is not OUT")
                    self._redraw()
                    return
                self._connect_from = port
                self.status.value = f"<b>Status:</b> source {port['node']}.{port['label']} выбран"
                self._redraw()
                return

            src, dst = self._connect_from, port
            if dst["dir"] != "in":
                self.status.value = "<b>Status:</b> вторым выберите IN"
                self._log("WARN", "Connect mode: second port is not IN")
                self._redraw()
                return
            if src["node"] == dst["node"]:
                self.status.value = "<b>Status:</b> нельзя соединять порты одного узла"
                self._log("WARN", "Rejected self-edge")
                self._connect_from = None
                self._redraw()
                return
            if not self._port_type_ok(src["type"], dst["type"]):
                self.status.value = f"<b>Status:</b> несовместимые порты: {src['type']} -> {dst['type']}"
                self._log("WARN", f"Incompatible ports: {src['type']} -> {dst['type']}")
                self._connect_from = None
                self._redraw()
                return

            edge = {"src_node": src["node"], "src_port": src["id"], "dst_node": dst["node"], "dst_port": dst["id"]}
            if edge not in self.edges:
                self.edges.append(edge)
                self.status.value = f"<b>Status:</b> связь {src['node']}.{src['label']} -> {dst['node']}.{dst['label']}"
                self._log("INFO", f"Edge added: {src['node']}.{src['id']} -> {dst['node']}.{dst['id']}")
            else:
                self.status.value = "<b>Status:</b> такая связь уже есть"
            self._connect_from = None
            self._sync_code_boxes()
            self._redraw()
            return

        nid = self._hit_test_node(event.xdata, event.ydata)
        if not nid:
            return
        self.selected_node = nid
        self.selected_port = None
        self._sync_code_boxes()
        node = self.nodes[nid]
        self._dragging = nid
        self._drag_offset = (event.xdata - node["x"], event.ydata - node["y"])

    def _on_motion(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        if self._panning and self._pan_start_px is not None:
            dx_px = event.x - self._pan_start_px[0]
            dy_px = event.y - self._pan_start_px[1]

            bbox = self.ax.bbox
            if bbox.width > 0 and bbox.height > 0:
                xpp = (self._pan_xlim[1] - self._pan_xlim[0]) / bbox.width
                ypp = (self._pan_ylim[1] - self._pan_ylim[0]) / bbox.height
                dx = dx_px * xpp
                dy = dy_px * ypp
                self.ax.set_xlim(self._pan_xlim[0] - dx, self._pan_xlim[1] - dx)
                self.ax.set_ylim(self._pan_ylim[0] - dy, self._pan_ylim[1] - dy)
                self.fig.canvas.draw_idle()
            return

        if self._dragging is None:
            return
        nid = self._dragging
        dx, dy = self._drag_offset
        self.nodes[nid]["x"] = float(event.xdata - dx)
        self.nodes[nid]["y"] = float(event.ydata - dy)
        self._redraw()

    def _on_release(self, event):
        self._dragging = None
        self._panning = False
        self._pan_start_px = None

    def _node_kind(self, node_type):
        spec = self.device_specs.get(node_type, {})
        ptypes = set()
        for m in spec.get("in_ports", []) + spec.get("out_ports", []):
            if len(m) >= 3:
                ptypes.add(m[2])
        has_opt = "opt" in ptypes
        has_el = "el" in ptypes or "det" in ptypes
        if has_opt and has_el:
            return "mixed"
        if has_opt:
            return "opt"
        return "el"

    def _node_face_colors(self, node_type):
        kind = self._node_kind(node_type)
        if kind == "opt":
            return ("#2563EB", None)
        if kind == "el":
            return ("#16A34A", None)
        return ("#16A34A", "#2563EB")

    def _port_color(self, ptype):
        if ptype == "opt":
            return "#2563EB"
        if ptype in ("el", "det"):
            return "#16A34A"
        return "#6B7280"

    def _redraw(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('auto')
        self.ax.set_title("SuperMan: drag приборы, connect по портам")

        for e in self.edges:
            sx, sy = self._port_xy(e["src_node"], e["src_port"], "out")
            dx, dy = self._port_xy(e["dst_node"], e["dst_port"], "in")
            if sx is None or dx is None:
                continue
            self.ax.add_patch(FancyArrowPatch((sx, sy), (dx, dy), arrowstyle="->", mutation_scale=11, lw=1.7, color="#355C7D"))

        for nid, node in self.nodes.items():
            c1, c2 = self._node_face_colors(node["type"])
            edge_color = "#F59E0B" if nid == self.selected_node else "#1B1B1B"
            box = FancyBboxPatch(
                (node["x"], node["y"]), self.BOX_W, self.BOX_H,
                boxstyle="round,pad=0.002,rounding_size=0.02",
                edgecolor=edge_color, facecolor='none' if c2 else c1, lw=1.6, zorder=3
            )
            self.ax.add_patch(box)

            if c2:
                grad = np.linspace(0, 1, 128).reshape(1, -1)
                cmap = LinearSegmentedColormap.from_list("node_mix", [c1, c2])
                im = self.ax.imshow(
                    grad,
                    extent=(node["x"], node["x"] + self.BOX_W, node["y"], node["y"] + self.BOX_H),
                    origin='lower', cmap=cmap, aspect='auto', zorder=2
                )
                im.set_clip_path(box)

            txt = self._short_label(node["type"]) if not self.show_ids else f"{self._short_label(node['type'])}\n{nid}"
            self.ax.text(node["x"] + self.BOX_W / 2, node["y"] + self.BOX_H / 2, txt, ha="center", va="center", fontsize=8, fontweight='bold', color='white', zorder=4)

            for p in self._node_ports(nid):
                pcolor = self._port_color(p["type"])
                if self._connect_from and p["node"] == self._connect_from["node"] and p["id"] == self._connect_from["id"]:
                    pcolor = "#DC2626"
                if self.selected_port and (p["node"], p["id"], p["dir"]) == self.selected_port:
                    pcolor = "#DC2626"

                self.ax.add_patch(plt.Circle((p["x"], p["y"]), self.PORT_R, color=pcolor, ec="#111", lw=0.8, zorder=5))

                if p["side"] == "left":
                    self.ax.text(p["x"] - 0.014, p["y"], p["label"], va="center", ha="right", fontsize=7)
                elif p["side"] == "right":
                    self.ax.text(p["x"] + 0.014, p["y"], p["label"], va="center", ha="left", fontsize=7)
                elif p["side"] == "top":
                    self.ax.text(p["x"], p["y"] + 0.014, p["label"], va="bottom", ha="center", fontsize=7)
                else:
                    self.ax.text(p["x"], p["y"] - 0.014, p["label"], va="top", ha="center", fontsize=7)

        self.fig.canvas.draw_idle()

    def _node_line(self, nid):
        node = self.nodes[nid]
        params = ", ".join([f"{k}={repr(v)}" for k, v in node["params"].items()])
        return f"{nid} = {node['type']}({params})"

    def _edge_line(self, e):
        return f"connect({e['src_node']}.{e['src_port']} -> {e['dst_node']}.{e['dst_port']})"

    def _scheme_lines(self):
        lines = ["# --- Auto-generated by SuperMan ---", ""]
        for nid in self.nodes:
            lines.append(self._node_line(nid))
        lines.append("")
        for e in self.edges:
            lines.append(self._edge_line(e))
        return lines

    def _sync_code_boxes(self):
        self.scheme_code.value = "\n".join(self._scheme_lines())
        self.node_code.value = self._node_line(self.selected_node) if (self.selected_node in self.nodes) else ""

    def _safe_eval(self, expr):
        return eval(expr, {"__builtins__": {}}, {"np": np, "None": None, "True": True, "False": False})

    def _prune_invalid_edges(self):
        valid = []
        for e in self.edges:
            if e["src_node"] not in self.nodes or e["dst_node"] not in self.nodes:
                continue
            src_ports = {p[0] for p in self.device_specs[self.nodes[e["src_node"]]["type"]].get("out_ports", [])}
            dst_ports = {p[0] for p in self.device_specs[self.nodes[e["dst_node"]]["type"]].get("in_ports", [])}
            if e["src_port"] in src_ports and e["dst_port"] in dst_ports:
                valid.append(e)
        self.edges = valid

    def _on_apply_node_code(self, _):
        if not self.selected_node:
            self.status.value = "<b>Status:</b> выберите прибор"
            return
        line = self.node_code.value.strip()
        try:
            tree = ast.parse(line, mode="exec")
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
                raise ValueError("Ожидается id = Class(...)")
            assign = tree.body[0]
            if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
                raise ValueError("Некорректная левая часть")
            new_id = assign.targets[0].id
            call = assign.value
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                raise ValueError("Правая часть должна быть вызовом класса")
            class_name = call.func.id
            if class_name not in self.device_specs:
                raise ValueError(f"Неизвестный класс {class_name}")

            params = {}
            for kw in call.keywords:
                if kw.arg is None:
                    raise ValueError("**kwargs не поддержаны")
                params[kw.arg] = self._safe_eval(ast.get_source_segment(line, kw.value))

            old_id = self.selected_node
            nd = self.nodes.pop(old_id)
            nd["id"] = new_id
            nd["type"] = class_name
            nd["params"] = params
            self.nodes[new_id] = nd

            new_edges = []
            for e in self.edges:
                ne = dict(e)
                if ne["src_node"] == old_id:
                    ne["src_node"] = new_id
                if ne["dst_node"] == old_id:
                    ne["dst_node"] = new_id
                if ne not in new_edges:
                    new_edges.append(ne)
            self.edges = new_edges
            self._prune_invalid_edges()
            self.selected_node = new_id
            self.status.value = f"<b>Status:</b> параметры {new_id} обновлены"
            self._log("INFO", f"Node code updated: {new_id}")
            self._sync_code_boxes()
            self._redraw()
        except Exception as ex:
            self.status.value = f"<b>Status:</b> ошибка кода: {ex}"
            self._log("ERROR", f"Node code apply failed: {ex}")

    def _incoming_map(self, nid, outputs):
        inc = {}
        for e in self.edges:
            if e["dst_node"] != nid:
                continue
            sig = outputs.get(e["src_node"], {}).get(e["src_port"])
            if sig is None:
                continue
            inc.setdefault(e["dst_port"], []).append(sig)
        return inc

    def _first(self, values, typ):
        for v in values:
            if isinstance(v, typ):
                return v
        return None

    def _first_on_port(self, inc, port_name, typ):
        return self._first(inc.get(port_name, []), typ)

    def _first_any(self, inc, typ):
        for lst in inc.values():
            v = self._first(lst, typ)
            if v is not None:
                return v
        return None

    def _toposort(self):
        indeg = {nid: 0 for nid in self.nodes}
        for e in self.edges:
            if e["src_node"] in indeg and e["dst_node"] in indeg:
                indeg[e["dst_node"]] += 1
        q = [n for n, d in indeg.items() if d == 0]
        out = []
        edges = list(self.edges)
        while q:
            n = q.pop(0)
            out.append(n)
            for e in list(edges):
                if e["src_node"] == n:
                    indeg[e["dst_node"]] -= 1
                    edges.remove(e)
                    if indeg[e["dst_node"]] == 0:
                        q.append(e["dst_node"])
        if len(out) < len(self.nodes):
            out.extend([n for n in self.nodes if n not in out])
            self._log("WARN", "Cycle detected, run order fallback used")
        return out

    def _build_instance(self, nid):
        node = self.nodes[nid]
        return self.device_specs[node["type"]]["class"](**node["params"])

    def _normalize_output(self, nid, value):
        out_ports = self.device_specs[self.nodes[nid]["type"]].get("out_ports", [])
        if isinstance(value, dict):
            return value
        if not out_ports:
            return {}
        return {out_ports[0][0]: value}

    def _run_awg(self, inst, inc, params):
        return inst.generate()

    def _run_laser(self, inst, inc, params):
        ref_e = self._first_any(inc, ElectricalSignal)
        t = ref_e.t if ref_e is not None else np.arange(0, 20e-12, 1e-14)
        return inst.generate(t)

    def _run_mzm(self, inst, inc, params):
        optical = self._first_on_port(inc, "opt_in", OpticalSignal) or self._first_any(inc, OpticalSignal)
        drive = self._first_on_port(inc, "el_in", ElectricalSignal) or self._first_any(inc, ElectricalSignal)
        if optical is None or drive is None:
            raise ValueError("IntensityModulator требует OptIn и ElIn")
        return inst.apply(optical, drive)

    def _run_splitter(self, inst, inc, params):
        optical = self._first_on_port(inc, "opt_in", OpticalSignal) or self._first_any(inc, OpticalSignal)
        if optical is None:
            raise ValueError("OpticalSplitter требует OptIn")
        out1, out2 = inst.split(optical)
        return {"opt_out1": out1, "opt_out2": out2}

    def _run_fiber(self, inst, inc, params):
        optical = self._first_on_port(inc, "opt_in", OpticalSignal) or self._first_any(inc, OpticalSignal)
        if optical is None:
            raise ValueError("Fiber требует OptIn")
        return inst.propagate(optical)

    def _run_photodetector(self, inst, inc, params):
        optical = self._first_on_port(inc, "opt_in", OpticalSignal) or self._first_any(inc, OpticalSignal)
        if optical is None:
            raise ValueError("Photodetector требует OptIn")
        return inst.detect(optical)

    def _run_apply_single(self, inst, inc, params):
        e = self._first_on_port(inc, "el_in", ElectricalSignal) or self._first_any(inc, ElectricalSignal)
        if e is None:
            raise ValueError(f"{inst.__class__.__name__} требует ElIn")
        return inst.apply(e)

    def _run_detect_single(self, inst, inc, params):
        optical = self._first_on_port(inc, "opt_in", OpticalSignal) or self._first_any(inc, OpticalSignal)
        if optical is None:
            raise ValueError(f"{inst.__class__.__name__} требует OptIn")
        return inst.detect(optical)

    def _plot_electrical(self, sig, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sig.t, y=np.real_if_close(sig.s), mode='lines'))
        fig.update_layout(title=title, xaxis_title='t [s]', yaxis_title='Amplitude', template='plotly_white')
        if self.inline_show:
            fig.show()
        return fig

    def _plot_optical_power(self, sig, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sig.t, y=np.abs(sig.A) ** 2, mode='lines'))
        fig.update_layout(title=title, xaxis_title='t [s]', yaxis_title='|A|^2', template='plotly_white')
        if self.inline_show:
            fig.show()
        return fig

    def _plot_detector_iq(self, sig, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sig.t, y=sig.i, mode='lines', name='I'))
        fig.add_trace(go.Scatter(x=sig.t, y=sig.q, mode='lines', name='Q'))
        fig.update_layout(title=title, xaxis_title='t [s]', yaxis_title='Amplitude', template='plotly_white')
        if self.inline_show:
            fig.show()
        return fig

    def _plot_detector_amp_phase(self, sig, title):
        phase = np.unwrap(sig.phase)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sig.t, y=sig.amplitude, mode='lines', name='Amplitude'))
        fig.add_trace(go.Scatter(x=sig.t, y=phase, mode='lines', name='Phase [rad]', yaxis='y2'))
        fig.update_layout(title=title, xaxis_title='t [s]', yaxis=dict(title='Amplitude'), yaxis2=dict(title='Phase [rad]', overlaying='y', side='right'), template='plotly_white')
        if self.inline_show:
            fig.show()
        return fig

    def _plot_eye(self, trace: ElectricalSignal, title, slot_duration=None, bit_rate=None, slots=3, max_traces=300):
        t = np.asarray(trace.t)
        signal = np.real(np.real_if_close(trace.s))
        dt = t[1] - t[0]

        if slot_duration is None:
            if bit_rate is not None:
                slot_duration = 1.0 / bit_rate
            else:
                ds = np.abs(np.diff(signal))
                thr = 0.25 * np.max(ds) if len(ds) > 0 else 0.0
                edge_idx = np.where(ds > thr)[0]
                if len(edge_idx) >= 2:
                    slot_duration = max(int(np.median(np.diff(edge_idx))), 2) * dt
                else:
                    slot_duration = (t[-1] - t[0]) / max(8, slots + 2)

        samples_per_slot = max(int(round(slot_duration / dt)), 2)
        window = slots * samples_per_slot
        if window >= len(signal):
            samples_per_slot = max(len(signal) // (slots + 1), 2)
            window = slots * samples_per_slot

        n_segments = len(signal) // samples_per_slot - slots + 1
        n_segments = max(n_segments, 1)
        if max_traces is not None:
            n_segments = min(n_segments, int(max_traces))

        max_points_per_trace = 1200
        step = max(1, int(np.ceil(window / max_points_per_trace)))
        x = np.arange(0, window, step) * dt

        fig = go.Figure()
        for k in range(n_segments):
            i0 = k * samples_per_slot
            i1 = i0 + window
            seg = signal[i0:i1]
            if len(seg) == window:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=seg[::step],
                    mode='lines',
                    opacity=0.33,
                    line=dict(width=2, color='blue'),
                    showlegend=False,
                    hoverinfo='skip'
                ))

        for m in range(1, slots):
            fig.add_vline(x=m * samples_per_slot * dt, line_dash='dot', line_color='gray')

        fig.update_layout(title=title, xaxis_title='Time in eye window [s]', yaxis_title='Amplitude', template='plotly_white')
        if self.inline_show:
            fig.show()
        return fig

    def _save_fig(self, fig, filename):
        os.makedirs(self.save_dir.value or "plots", exist_ok=True)
        path = os.path.join(self.save_dir.value or "plots", filename)
        fig.write_html(path)
        return path

    def _plot_outputs(self, nid, out_map):
        for port, value in out_map.items():
            fig = None
            if isinstance(value, ElectricalSignal):
                fig = self._plot_electrical(value, f"{nid}.{port}: ElectricalSignal")
            elif isinstance(value, OpticalSignal):
                fig = self._plot_optical_power(value, f"{nid}.{port}: OpticalSignal power")
            elif isinstance(value, DetectorSignal):
                fig = self._plot_detector_iq(value, f"{nid}.{port}: Detector I/Q")
            if fig is None:
                continue
            self._figures.append((f"{nid}.{port}", fig))
            if self.save_checkbox.value:
                self._save_fig(fig, f"{nid}.{port}.html")

    def _mode_for_scope_port(self, node_params, port_name):
        modes = node_params.get("plot_modes", {})
        if isinstance(modes, dict):
            return str(modes.get(port_name, "auto")).lower()
        return str(node_params.get("plot_mode", "auto")).lower()

    def _title_for_scope_port(self, node_params, port_name, fallback):
        titles = node_params.get("plot_titles", {})
        if isinstance(titles, dict):
            return str(titles.get(port_name, fallback))
        return fallback

    def _run_scope(self, inst, inc, params):
        port_labels = {}
        for meta in self.device_specs["Oscilloscope"]["in_ports"]:
            pname = meta[0]
            plabel = meta[1] if len(meta) >= 2 else pname
            port_labels[pname] = plabel

        plotted = 0
        for port_name, vals in inc.items():
            if not vals:
                continue

            sig = self._first(vals, (ElectricalSignal, OpticalSignal, DetectorSignal))
            if sig is None:
                continue

            mode = self._mode_for_scope_port(params, port_name)
            title = str(port_labels.get(port_name, port_name))
            fig = None

            if mode == "auto":
                if isinstance(sig, ElectricalSignal):
                    fig = self._plot_electrical(sig, title)
                elif isinstance(sig, OpticalSignal):
                    fig = self._plot_optical_power(sig, title)
                elif isinstance(sig, DetectorSignal):
                    fig = self._plot_detector_iq(sig, title)
            elif mode == "time":
                if isinstance(sig, ElectricalSignal):
                    fig = self._plot_electrical(sig, title)
                elif isinstance(sig, OpticalSignal):
                    fig = self._plot_optical_power(sig, title + " (power)")
            elif mode == "power":
                if isinstance(sig, OpticalSignal):
                    fig = self._plot_optical_power(sig, title)
                elif isinstance(sig, DetectorSignal):
                    fig = self._plot_electrical(ElectricalSignal(sig.t, sig.power), title)
            elif mode == "iq" and isinstance(sig, DetectorSignal):
                fig = self._plot_detector_iq(sig, title)
            elif mode == "amp_phase" and isinstance(sig, DetectorSignal):
                fig = self._plot_detector_amp_phase(sig, title)
            elif mode == "eye" and isinstance(sig, ElectricalSignal):
                fig = self._plot_eye(
                    sig,
                    title,
                    slot_duration=params.get("eye_slot_duration"),
                    bit_rate=params.get("eye_bit_rate"),
                    slots=int(params.get("eye_slots", 3)),
                    max_traces=int(params.get("eye_max_traces", 250)),
                )

            if fig is not None:
                self._figures.append((f"scope.{port_name}", fig))
                plotted += 1
                if self.save_checkbox.value:
                    self._save_fig(fig, f"scope.{port_name}.html")
            else:
                self._log("WARN", f"Scope {port_name}: mode '{mode}' incompatible with signal")

        if plotted == 0:
            self._log("WARN", "Oscilloscope: no connected/compatible inputs to plot")
        return {}

    def export_py(self, path="scheme_export.py"):
        order = self._toposort()
        lines = []
        lines.append("import numpy as np")
        lines.append("import plotly.graph_objects as go")
        lines.append("from opticat import *")
        lines.append("")
        lines.append("# Generated by SuperMan.export_py")
        lines.append("nodes_params = {}")
        for nid in self.nodes:
            lines.append(f"nodes_params['{nid}'] = {repr(self.nodes[nid]['params'])}")
        lines.append("")
        for nid in self.nodes:
            node = self.nodes[nid]
            ptxt = ", ".join([f"{k}={repr(v)}" for k, v in node["params"].items()])
            lines.append(f"{nid} = {node['type']}({ptxt})")
        lines.append("")
        lines.append("edges = [")
        for e in self.edges:
            lines.append(f"    ('{e['src_node']}', '{e['src_port']}', '{e['dst_node']}', '{e['dst_port']}'),")
        lines.append("]")
        lines.append(f"order = {repr(order)}")
        lines.append("")
        lines.append("instances = {")
        for nid in self.nodes:
            lines.append(f"    '{nid}': {nid},")
        lines.append("}")
        lines.append("outputs = {}")
        lines.append("")
        lines.append("def _first(values, typ):")
        lines.append("    for v in values:")
        lines.append("        if isinstance(v, typ):")
        lines.append("            return v")
        lines.append("    return None")
        lines.append("")
        lines.append("def incoming(node):")
        lines.append("    inc = {}")
        lines.append("    for s, sp, d, dp in edges:")
        lines.append("        if d != node:")
        lines.append("            continue")
        lines.append("        sig = outputs.get(s, {}).get(sp)")
        lines.append("        if sig is None:")
        lines.append("            continue")
        lines.append("        inc.setdefault(dp, []).append(sig)")
        lines.append("    return inc")
        lines.append("")
        lines.append("def normalize(nid, value):")
        lines.append("    if isinstance(value, dict):")
        lines.append("        return value")
        lines.append("    out = {")
        for nid in self.nodes:
            out_ports = self.device_specs[self.nodes[nid]["type"]].get("out_ports", [])
            first = repr(out_ports[0][0]) if out_ports else "None"
            lines.append(f"        '{nid}': {first},")
        lines.append("    }")
        lines.append("    p = out.get(nid)")
        lines.append("    return {} if p is None else {p: value}")
        lines.append("")
        lines.append("for nid in order:")
        lines.append("    inst = instances[nid]")
        lines.append("    inc = incoming(nid)")
        lines.append("    tname = type(inst).__name__")
        lines.append("    if tname == 'ArbitraryWaveformGenerator':")
        lines.append("        raw = inst.generate()")
        lines.append("    elif tname == 'Laser':")
        lines.append("        ref = None")
        lines.append("        for arr in inc.values():")
        lines.append("            ref = _first(arr, ElectricalSignal)")
        lines.append("            if ref is not None: break")
        lines.append("        t = ref.t if ref is not None else np.arange(0, 20e-12, 1e-14)")
        lines.append("        raw = inst.generate(t)")
        lines.append("    elif tname == 'IntensityModulator':")
        lines.append("        opt = _first(inc.get('opt_in', []), OpticalSignal)")
        lines.append("        dr = _first(inc.get('el_in', []), ElectricalSignal)")
        lines.append("        raw = inst.apply(opt, dr)")
        lines.append("    elif tname == 'OpticalSplitter':")
        lines.append("        opt = _first(inc.get('opt_in', []), OpticalSignal)")
        lines.append("        o1, o2 = inst.split(opt)")
        lines.append("        raw = {'opt_out1': o1, 'opt_out2': o2}")
        lines.append("    elif tname == 'Fiber':")
        lines.append("        raw = inst.propagate(_first(inc.get('opt_in', []), OpticalSignal))")
        lines.append("    elif tname == 'Photodetector':")
        lines.append("        raw = inst.detect(_first(inc.get('opt_in', []), OpticalSignal))")
        lines.append("    elif tname in ('BandPassFilter','LowPassFilter','ElectricalNoiseGenerator'):")
        lines.append("        raw = inst.apply(_first(inc.get('el_in', []), ElectricalSignal))")
        lines.append("    elif tname in ('IncoherentDetector','CoherentDetector'):")
        lines.append("        raw = inst.detect(_first(inc.get('opt_in', []), OpticalSignal))")
        lines.append("    elif tname == 'Oscilloscope':")
        lines.append("        prm = nodes_params[nid]")
        lines.append("        modes = prm.get('plot_modes', {})")
        lines.append("        titles = prm.get('plot_titles', {})")
        lines.append("        for p in ['time1','time2','time3','time4','time5','time6','eye1','eye2','eye3','eye4','eye5','eye6','custom1','custom2','custom3','custom4']:")
        lines.append("            vals = inc.get(p, [])")
        lines.append("            sig = vals[0] if vals else None")
        lines.append("            if sig is None: continue")
        lines.append("            md = str(modes.get(p, 'auto')).lower()")
        lines.append("            tt = str(p)")
        lines.append("            if md in ('auto','time') and isinstance(sig, ElectricalSignal):")
        lines.append("                sig.plot(title=tt)")
        lines.append("            elif md == 'eye' and isinstance(sig, ElectricalSignal):")
        lines.append("                inst.plot_eye(sig, title=tt, slot_duration=prm.get('eye_slot_duration'), bit_rate=prm.get('eye_bit_rate'), slots=int(prm.get('eye_slots',3)), max_traces=int(prm.get('eye_max_traces',250)))")
        lines.append("            elif isinstance(sig, DetectorSignal):")
        lines.append("                inst.plot_iq(sig, title=tt)")
        lines.append("            elif isinstance(sig, OpticalSignal):")
        lines.append("                ElectricalSignal(sig.t, np.abs(sig.A)**2).plot(title=tt)")
        lines.append("        raw = {}")
        lines.append("    else:")
        lines.append("        raise RuntimeError(f'Unsupported node type {tname}')")
        lines.append("    outputs[nid] = normalize(nid, raw)")
        lines.append("")
        lines.append("print('Run completed. Nodes:', ', '.join(order))")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self._log("INFO", f"Exported python script: {path}")
        return path

    def _on_run(self, _):
        self._sync_code_boxes()
        self._figures.clear()
        outputs = {}
        order = self._toposort()
        self._log("INFO", f"Run started. Order: {' -> '.join(order) if order else '(empty)'}")

        with self.output:
            clear_output(wait=True)
            print("Run order:", " -> ".join(order) if order else "(empty)")
            for nid in order:
                node = self.nodes[nid]
                inc = self._incoming_map(nid, outputs)
                inst = self._build_instance(nid)
                runner = self.device_specs[node["type"]]["runner"]
                try:
                    raw = runner(inst, inc, node["params"])
                    out_map = self._normalize_output(nid, raw)
                    outputs[nid] = out_map
                    print(f"[ok] {nid} ({node['type']})")
                    self._log("INFO", f"[ok] {nid} ({node['type']})")
                except Exception as ex:
                    outputs[nid] = {}
                    print(f"[fail] {nid} ({node['type']}): {ex}")
                    self._log("ERROR", f"[fail] {nid} ({node['type']}): {ex}")

            if self.save_checkbox.value and self._figures:
                print(f"Saved html plots into: {self.save_dir.value or 'plots'}")

        self._last_outputs = outputs
        self.status.value = "<b>Status:</b> run completed"
        self._log("INFO", "Run completed")

