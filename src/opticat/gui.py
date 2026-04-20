"""Desktop GUI wrapper for SuperMan using tkinter."""

import os
import tempfile
import webbrowser
from datetime import datetime

from .core import SuperMan


class SuperCat:
    def __init__(self):
        try:
            import tkinter as tk
            from tkinter import ttk
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as ex:
            raise ImportError("Для SuperCat нужен tkinter (и backend tkagg matplotlib).") from ex

        self.tk = tk
        self.ttk = ttk
        self.FigureCanvasTkAgg = FigureCanvasTkAgg

        self.core = SuperMan(use_notebook_backend=False)
        self.core.inline_show = False

        self.root = None
        self.status_var = None
        self.device_var = None
        self.link_mode_var = None
        self.pan_var = None
        self.show_ids_var = None
        self.save_var = None
        self.save_dir_var = None

        self.log_info_var = None
        self.log_warn_var = None
        self.log_error_var = None

        self.node_code_text = None
        self.scheme_text = None
        self.log_text = None

        self.history_canvas = None
        self.history_inner = None
        self._history = []

    def _clean_status(self, html_text):
        s = str(html_text)
        for old, new in [("<b>", ""), ("</b>", ""), ("<br>", " "), ("&nbsp;", " ")]:
            s = s.replace(old, new)
        return s

    def _set_text(self, widget, text):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', text)

    def _set_text_readonly(self, widget, text):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', text)
        widget.configure(state='disabled')

    def _sync_to_core(self):
        self.core.device_select.value = self.device_var.get()

        mode = self.link_mode_var.get() if self.link_mode_var is not None else "off"
        self.core.connect_btn.value = (mode == "connect")
        self.core.delete_edge_mode = (mode == "delete")

        self.core.pan_mode = bool(self.pan_var.get())
        self.core.show_ids = bool(self.show_ids_var.get())
        self.core.save_checkbox.value = bool(self.save_var.get())
        self.core.save_dir.value = self.save_dir_var.get().strip() or "plots"
        node_code = self.node_code_text.get('1.0', 'end').strip()
        if node_code:
            self.core.node_code.value = node_code

    def _refresh_logs(self):
        if self.log_text is None:
            return
        allowed = set()
        if self.log_info_var.get():
            allowed.add("INFO")
        if self.log_warn_var.get():
            allowed.add("WARN")
        if self.log_error_var.get():
            allowed.add("ERROR")

        lines = []
        for r in self.core.logs[-350:]:
            if r["level"] in allowed:
                lines.append(f"[{r['time']}] {r['level']}: {r['message']}")
        self._set_text_readonly(self.log_text, "\n".join(lines))

    def _sync_from_core(self, force=False):
        self.status_var.set(self._clean_status(self.core.status.value))

        scheme = self.core.scheme_code.value
        if self.scheme_text.get('1.0', 'end').strip() != scheme.strip():
            self._set_text_readonly(self.scheme_text, scheme)

        focus_widget = None
        if self.root is not None:
            try:
                focus_widget = self.root.focus_get()
            except Exception:
                focus_widget = None

        current_node_code = self.core.node_code.value
        if force or focus_widget != self.node_code_text:
            if self.node_code_text.get('1.0', 'end').strip() != current_node_code.strip():
                self._set_text(self.node_code_text, current_node_code)

        self._refresh_logs()

    def _fig_to_png(self, fig, width, height):
        try:
            return fig.to_image(format='png', width=int(width), height=int(height), scale=1)
        except Exception as ex:
            self.core._log("WARN", f"PNG preview unavailable ({ex}). Install kaleido: python3 -m pip install kaleido")
            return None

    def _capture_plot_history(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for name, fig in self.core._figures:
            self._history.append({
                "time": ts,
                "name": name,
                "fig": fig,
                "thumb": None,
                "full": None,
                "html": None,
                "thumb_img": None,
                "full_img": None,
            })
        if len(self._history) > 400:
            self._history = self._history[-400:]

    def _save_temp_html(self, fig):
        try:
            fd, path = tempfile.mkstemp(prefix='supercat_plot_', suffix='.html')
            os.close(fd)
            fig.write_html(path)
            return path
        except Exception as ex:
            self.core._log("WARN", f"Temp html export failed: {ex}")
            return None

    def _open_plot_full(self, entry):
        tk = self.tk
        win = tk.Toplevel(self.root)
        win.title(f"Plot: {entry['name']}")
        win.geometry('600x160')

        top = self.ttk.Frame(win)
        top.pack(fill='x')
        self.ttk.Button(top, text='Закрыть', command=win.destroy).pack(side='right', padx=8, pady=6)

        body = self.ttk.Frame(win)
        body.pack(fill='both', expand=True)

        tk.Label(body, text='Открываю интерактивный график в браузере...').pack(padx=8, pady=12)

        if entry.get("html") is None:
            fig = entry.get("fig")
            if fig is not None:
                entry["html"] = self._save_temp_html(fig)

        if entry.get("html"):
            webbrowser.open(f"file://{entry['html']}")
        else:
            tk.Label(body, text='Не удалось подготовить HTML графика').pack(padx=8, pady=6)

    def _clear_history(self):
        self._history = []
        self._refresh_history_panel()
        self.core._log("INFO", "Plot history cleared")
        self._refresh_logs()

    def _refresh_history_panel(self):
        if self.history_inner is None:
            return
        tk = self.tk
        for w in self.history_inner.winfo_children():
            w.destroy()

        if not self._history:
            tk.Label(self.history_inner, text='Графиков пока нет', anchor='w', justify='left').pack(fill='x', padx=6, pady=6)
            return

        for entry in reversed(self._history):
            card = self.ttk.Frame(self.history_inner)
            card.pack(fill='x', padx=5, pady=4)
            self.ttk.Label(card, text=f"{entry['time']}  {entry['name']}").pack(fill='x', anchor='w')
            self.ttk.Button(card, text='Открыть график', command=lambda en=entry: self._open_plot_full(en)).pack(anchor='w', pady=(2, 4))

    def _on_add(self):
        self._sync_to_core()
        self.core._on_add(None)
        self._sync_from_core(force=True)

    def _on_delete(self):
        self._sync_to_core()
        self.core._on_delete(None)
        self._sync_from_core(force=True)

    def _on_link_mode_change(self):
        self._sync_to_core()
        mode = self.link_mode_var.get() if self.link_mode_var is not None else "off"
        if mode == "connect":
            self.status_var.set("Status: link mode CONNECT")
        elif mode == "delete":
            self.status_var.set("Status: link mode DELETE")
        else:
            self.status_var.set("Status: link mode OFF")

    def _on_save_node_code(self):
        self._sync_to_core()
        self.core._log("INFO", "Node code saved")
        self.status_var.set("Status: node code saved")
        self._refresh_logs()

    def _on_apply_node_code(self):
        self._sync_to_core()
        self.core._on_apply_node_code(None)
        self._sync_from_core(force=True)

    def _on_run(self):
        self._sync_to_core()
        self.core._on_run(None)
        n = len(self.core._figures)
        self._capture_plot_history()
        self._refresh_history_panel()
        self._sync_from_core(force=True)
        if n > 0:
            self.status_var.set(f"Status: run completed, plots: {n}")
        else:
            self.status_var.set("Status: run completed, plots: 0 (check logs)")

    def _on_zoom_in(self):
        self.core._zoom(0.9)

    def _on_zoom_out(self):
        self.core._zoom(1.1)

    def _on_reset_zoom(self):
        self.core._on_reset_zoom(None)

    def _on_export_py(self):
        self._sync_to_core()
        out = self.core.export_py("scheme_export.py")
        self.status_var.set(f"Status: exported to {out}")
        self._refresh_logs()

    def _poll(self):
        if self.root is None:
            return
        self._sync_from_core(force=False)
        self.root.after(250, self._poll)

    def open(self, title='SuperCat'):
        tk = self.tk
        ttk = self.ttk

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry('1800x1000')

        main_paned = ttk.Panedwindow(self.root, orient='horizontal')
        main_paned.pack(fill='both', expand=True)

        left_outer = ttk.Frame(main_paned)
        right = ttk.Frame(main_paned)
        main_paned.add(left_outer, weight=7)
        main_paned.add(right, weight=5)

        left_paned = ttk.Panedwindow(left_outer, orient='horizontal')
        left_paned.pack(fill='both', expand=True)

        hist_frame = ttk.Frame(left_paned)
        scheme_frame = ttk.Frame(left_paned)
        left_paned.add(hist_frame, weight=2)
        left_paned.add(scheme_frame, weight=8)

        hist_top = ttk.Frame(hist_frame)
        hist_top.pack(fill='x', padx=6, pady=(6, 2))
        ttk.Label(hist_top, text='История графиков').pack(side='left')
        ttk.Button(hist_top, text='Clear history', command=self._clear_history).pack(side='right')

        self.history_canvas = tk.Canvas(hist_frame, highlightthickness=0)
        hs = ttk.Scrollbar(hist_frame, orient='vertical', command=self.history_canvas.yview)
        self.history_canvas.configure(yscrollcommand=hs.set)
        hs.pack(side='right', fill='y')
        self.history_canvas.pack(side='left', fill='both', expand=True)

        self.history_inner = ttk.Frame(self.history_canvas)
        win_id = self.history_canvas.create_window((0, 0), window=self.history_inner, anchor='nw')
        self.history_inner.bind('<Configure>', lambda _e: self.history_canvas.configure(scrollregion=self.history_canvas.bbox('all')))
        self.history_canvas.bind('<Configure>', lambda e: self.history_canvas.itemconfig(win_id, width=e.width))

        scheme_tools = ttk.Frame(scheme_frame)
        scheme_tools.pack(fill='x', padx=6, pady=(6, 2))
        ttk.Label(scheme_tools, text='View:').pack(side='left')
        ttk.Button(scheme_tools, text='-', width=3, command=self._on_zoom_out).pack(side='left', padx=2)
        ttk.Button(scheme_tools, text='r', width=3, command=self._on_reset_zoom).pack(side='left', padx=2)
        ttk.Button(scheme_tools, text='+', width=3, command=self._on_zoom_in).pack(side='left', padx=2)
        ttk.Checkbutton(scheme_tools, text='pan', variable=self.pan_var, command=lambda: self._sync_to_core()).pack(side='left', padx=10)

        canvas = self.FigureCanvasTkAgg(self.core.fig, master=scheme_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        top = ttk.Frame(right)
        top.pack(fill='x', padx=8, pady=8)

        self.device_var = tk.StringVar(value=list(self.core.device_specs.keys())[0])
        self.link_mode_var = tk.StringVar(value='connect')
        self.pan_var = tk.BooleanVar(value=False)
        self.show_ids_var = tk.BooleanVar(value=True)
        self.save_var = tk.BooleanVar(value=False)
        self.save_dir_var = tk.StringVar(value='plots')
        self.status_var = tk.StringVar(value='Status: ready')

        self.log_info_var = tk.BooleanVar(value=True)
        self.log_warn_var = tk.BooleanVar(value=True)
        self.log_error_var = tk.BooleanVar(value=True)

        ttk.Label(top, text='Прибор').grid(row=0, column=0, sticky='w')
        ttk.Combobox(top, textvariable=self.device_var, values=list(self.core.device_specs.keys()), state='readonly', width=32).grid(row=0, column=1, columnspan=5, sticky='ew', padx=4)

        ttk.Button(top, text='Добавить', command=self._on_add).grid(row=1, column=0, sticky='ew', pady=4)
        ttk.Button(top, text='Удалить узел', command=self._on_delete).grid(row=1, column=1, sticky='ew', pady=4)
        links = ttk.Frame(top)
        links.grid(row=1, column=2, columnspan=2, sticky='w')
        ttk.Radiobutton(links, text='Connect', value='connect', variable=self.link_mode_var, command=self._on_link_mode_change).pack(side='left')
        ttk.Radiobutton(links, text='Delete', value='delete', variable=self.link_mode_var, command=self._on_link_mode_change).pack(side='left', padx=6)
        ttk.Radiobutton(links, text='Off', value='off', variable=self.link_mode_var, command=self._on_link_mode_change).pack(side='left')
        ttk.Checkbutton(top, text='Show IDs', variable=self.show_ids_var, command=lambda: self._sync_to_core() or self.core._redraw()).grid(row=1, column=4, sticky='w', pady=4)
        ttk.Button(top, text='Run', command=self._on_run).grid(row=1, column=5, sticky='ew', pady=4)

        ttk.Label(top, text='').grid(row=2, column=0)
        ttk.Label(top, text='').grid(row=2, column=1)
        ttk.Button(top, text='Export.py', command=self._on_export_py).grid(row=2, column=2, sticky='ew', pady=4)
        ttk.Checkbutton(top, text='Save plots', variable=self.save_var).grid(row=2, column=3, sticky='w', pady=4)

        ttk.Label(top, text='Папка').grid(row=3, column=0, sticky='e', pady=4)
        ttk.Entry(top, textvariable=self.save_dir_var).grid(row=3, column=1, columnspan=5, sticky='ew', pady=4)

        for c in range(6):
            top.columnconfigure(c, weight=1)

        right_paned = ttk.Panedwindow(right, orient='vertical')
        right_paned.pack(fill='both', expand=True, padx=8, pady=6)

        node_frame = ttk.LabelFrame(right_paned, text='Node code')
        scheme_frame_r = ttk.LabelFrame(right_paned, text='Scheme code')
        logs_frame = ttk.LabelFrame(right_paned, text='Logs')
        right_paned.add(node_frame, weight=3)
        right_paned.add(scheme_frame_r, weight=4)
        right_paned.add(logs_frame, weight=3)

        node_btns = ttk.Frame(node_frame)
        node_btns.pack(fill='x', pady=4)
        ttk.Button(node_btns, text='Save', command=self._on_save_node_code).pack(side='left')
        ttk.Button(node_btns, text='Применить код', command=self._on_apply_node_code).pack(side='right')
        self.node_code_text = tk.Text(node_frame, wrap='word')
        self.node_code_text.pack(fill='both', expand=True)

        self.scheme_text = tk.Text(scheme_frame_r, wrap='word', state='disabled')
        self.scheme_text.pack(fill='both', expand=True)

        logs_top = ttk.Frame(logs_frame)
        logs_top.pack(fill='x')
        ttk.Checkbutton(logs_top, text='INFO', variable=self.log_info_var, command=self._refresh_logs).pack(side='left')
        ttk.Checkbutton(logs_top, text='WARN', variable=self.log_warn_var, command=self._refresh_logs).pack(side='left')
        ttk.Checkbutton(logs_top, text='ERROR', variable=self.log_error_var, command=self._refresh_logs).pack(side='left')

        self.log_text = tk.Text(logs_frame, wrap='word', state='disabled')
        self.log_text.pack(fill='both', expand=True)

        ttk.Label(right, textvariable=self.status_var, anchor='w').pack(fill='x', padx=8, pady=(2, 8))

        self._refresh_history_panel()
        self._sync_to_core()
        self._sync_from_core(force=True)

        self.core.fig.canvas.mpl_connect('button_release_event', lambda e: self._sync_from_core(force=True))
        self.core.fig.canvas.mpl_connect('button_press_event', lambda e: self._sync_from_core(force=True))

        self._poll()
        self.root.mainloop()


