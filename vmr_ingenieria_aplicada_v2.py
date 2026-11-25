import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon, Circle, Arc, FancyArrowPatch
from estructuras import Estructura2D, ConversorUnidades


def _fmt_num(val, dec=2):
    if abs(val) < 0.5 * (10 ** (-dec)):
        val = 0.0
    s = f"{val:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def _x_nodo(nodo):
    if isinstance(nodo, (tuple, list, np.ndarray)):
        return float(next(iter(nodo)))
    return float(nodo)


def _conv_l(val, ul):
    if hasattr(ConversorUnidades, "de_base_l"):
        return ConversorUnidades.de_base_l(val, ul)
    if hasattr(ConversorUnidades, "desde_base_l"):
        return ConversorUnidades.desde_base_l(val, ul)
    factores = {"m": 1.0, "cm": 100.0, "mm": 1000.0, "ft": 1.0 / 0.3048, "in": 1.0 / 0.0254}
    return val * factores.get(ul, 1.0)


def _a_base_l(val, ul):
    if hasattr(ConversorUnidades, "a_base_l"):
        return ConversorUnidades.a_base_l(val, ul)
    factores = {"m": 1.0, "cm": 0.01, "mm": 0.001, "ft": 0.3048, "in": 0.0254}
    return val * factores.get(ul, 1.0)


def _conv_f(val, uf):
    if hasattr(ConversorUnidades, "de_base_f"):
        return ConversorUnidades.de_base_f(val, uf)
    if hasattr(ConversorUnidades, "desde_base_f"):
        return ConversorUnidades.desde_base_f(val, uf)
    factores = {"kN": 1.0, "tf": 1.0 / 9.80665, "kgf": 1000.0 / 9.80665, "lbf": 1000.0 / 4.4482216, "kip": 1.0 / 4.4482216}
    return val * factores.get(uf, 1.0)


def _conv_m(val, uf, ul):
    if hasattr(ConversorUnidades, "de_base_m"):
        return ConversorUnidades.de_base_m(val, uf, ul)
    if hasattr(ConversorUnidades, "desde_base_m"):
        return ConversorUnidades.desde_base_m(val, uf, ul)
    return _conv_l(_conv_f(val, uf), ul)


class VMRIngenieriaAplicadaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VMR Ingeniería Aplicada v2.0 - Análisis Estructural I | Ing. Víctor Martínez Rodríguez")
        self.root.geometry("1300x820")
        self.root.minsize(1120, 700)

        self.u_fuerza = tk.StringVar(value="kN")
        self.u_long = tk.StringVar(value="m")
        self.u_long_prev = "m"
        self.longitud_viga = tk.DoubleVar(value=30.0)
        self.E_val = tk.DoubleVar(value=2.1e7)
        self.I_val = tk.DoubleVar(value=0.005)
        self.A_val = tk.DoubleVar(value=1.0)

        self.apoyos = []
        self.rotulas = []
        self.puntuales = []
        self.momentos = []
        self.distribuidas = []

        self.modelo_calculado = None
        self.curva_global = None
        self.mapa_letras = {}
        self.x_consulta = tk.DoubleVar(value=0.0)

        self._construir_ui()
        self._cargar_ejemplo_inicial()

    def _construir_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Header.TFrame", background="#14213d")
        style.configure("Header.TLabel", background="#14213d", foreground="white", font=("Segoe UI", 12, "bold"))
        style.configure("SubHeader.TLabel", background="#14213d", foreground="#fca311", font=("Segoe UI", 10, "bold"))
        style.configure("Tool.TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.configure("Calc.TButton", font=("Segoe UI", 10, "bold"), padding=8, foreground="#003049")

        header = ttk.Frame(self.root, style="Header.TFrame", padding=(14, 10))
        header.pack(fill="x")
        ttk.Label(header, text="VMR INGENIERÍA APLICADA  |  Análisis Estructural I", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="Autor: Ing. Víctor Martínez Rodríguez", style="SubHeader.TLabel").pack(side="right")

        cfg = ttk.LabelFrame(self.root, text="Configuración General y Unidades", padding=(10, 6))
        cfg.pack(fill="x", padx=10, pady=(6, 4))

        ttk.Label(cfg, text="Fuerza:").pack(side="left", padx=(0, 4))
        cb_f = ttk.Combobox(cfg, textvariable=self.u_fuerza, values=["kN", "tf", "kgf", "lbf", "kip"], width=6, state="readonly")
        cb_f.pack(side="left", padx=(0, 12))
        cb_f.bind("<<ComboboxSelected>>", lambda e: self._al_cambiar_unidad_fuerza())

        ttk.Label(cfg, text="Longitud:").pack(side="left", padx=(0, 4))
        cb_l = ttk.Combobox(cfg, textvariable=self.u_long, values=["m", "cm", "mm", "ft", "in"], width=6, state="readonly")
        cb_l.pack(side="left", padx=(0, 12))
        cb_l.bind("<<ComboboxSelected>>", lambda e: self._al_cambiar_unidad_longitud())

        ttk.Label(cfg, text="Longitud Total Viga (L):").pack(side="left", padx=(0, 4))
        ttk.Entry(cfg, textvariable=self.longitud_viga, width=8).pack(side="left", padx=(0, 4))
        ttk.Button(cfg, text="Aplicar L", command=self._actualizar_longitud).pack(side="left", padx=(0, 14))

        ttk.Label(cfg, text="E:").pack(side="left", padx=(0, 4))
        ttk.Entry(cfg, textvariable=self.E_val, width=10).pack(side="left", padx=(0, 10))
        ttk.Label(cfg, text="I:").pack(side="left", padx=(0, 4))
        ttk.Entry(cfg, textvariable=self.I_val, width=8).pack(side="left", padx=(0, 10))

        tools = ttk.Frame(self.root, padding=(10, 4))
        tools.pack(fill="x")

        ttk.Button(tools, text="▲ + Apoyo", style="Tool.TButton", command=self._dlg_apoyo).pack(side="left", padx=3)
        ttk.Button(tools, text="○ + Rótula", style="Tool.TButton", command=self._dlg_rotula).pack(side="left", padx=3)
        ttk.Button(tools, text="⬇ + Carga Puntual", style="Tool.TButton", command=self._dlg_puntual).pack(side="left", padx=3)
        ttk.Button(tools, text="↻ + Momento", style="Tool.TButton", command=self._dlg_momento).pack(side="left", padx=3)
        ttk.Button(tools, text="📶 + Carga Distribuida", style="Tool.TButton", command=self._dlg_distribuida).pack(side="left", padx=3)
        ttk.Button(tools, text="🗑 Limpiar Todo", style="Tool.TButton", command=self._limpiar_todo).pack(side="left", padx=(14, 3))
        ttk.Button(tools, text="🖼 Exportar Lámina PNG", style="Tool.TButton", command=self._exportar_png).pack(side="left", padx=3)
        ttk.Button(tools, text="▶   CALCULAR ESTRUCTURA", style="Calc.TButton", command=self._calcular).pack(side="right", padx=3)

        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=4)

        left_frame = ttk.LabelFrame(paned, text="Elementos Definidos (Doble clic para eliminar)", padding=6)
        paned.add(left_frame, weight=1)

        self.lista_items = tk.Listbox(left_frame, font=("Consolas", 9), height=25, selectmode="single")
        self.lista_items.pack(fill="both", expand=True)
        self.lista_items.bind("<Double-Button-1>", self._eliminar_seleccionado)
        ttk.Button(left_frame, text="Eliminar Seleccionado", command=self._eliminar_seleccionado).pack(fill="x", pady=(6, 0))

        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=4)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill="both", expand=True)

        self.tab_esquema = ttk.Frame(self.notebook)
        self.tab_diagramas = ttk.Frame(self.notebook)
        self.tab_memoria = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_esquema, text=" 📐 Esquema, Nudos y Reacciones ")
        self.notebook.add(self.tab_diagramas, text=" 📊 Diagramas N, V, M y Deflexión ")
        self.notebook.add(self.tab_memoria, text=" 📝 Memoria de Cálculo (Paso a Paso) ")

        self.fig_esq, self.ax_esq = plt.subplots(figsize=(9, 4.5), dpi=100)
        self.canvas_esq = FigureCanvasTkAgg(self.fig_esq, master=self.tab_esquema)
        self.canvas_esq.get_tk_widget().pack(fill="both", expand=True)

        self.fig_diag, self.axes_diag = plt.subplots(4, 1, figsize=(9, 6.8), dpi=100, sharex=True)
        self.canvas_diag = FigureCanvasTkAgg(self.fig_diag, master=self.tab_diagramas)
        self.canvas_diag.get_tk_widget().pack(fill="both", expand=True)

        scroll_y = ttk.Scrollbar(self.tab_memoria, orient="vertical")
        self.txt_memoria = tk.Text(self.tab_memoria, font=("Consolas", 10), wrap="none", yscrollcommand=scroll_y.set)
        scroll_y.config(command=self.txt_memoria.yview)
        scroll_y.pack(side="right", fill="y")
        self.txt_memoria.pack(fill="both", expand=True)

        insp_frame = ttk.LabelFrame(self.root, text=" 🔍 Inspector de Sección Exacta a cualquier distancia x ", padding=(10, 6))
        insp_frame.pack(fill="x", padx=10, pady=(2, 8))

        ttk.Label(insp_frame, text="Distancia x =").pack(side="left", padx=(0, 4))
        self.ent_x_insp = ttk.Entry(insp_frame, textvariable=self.x_consulta, width=9)
        self.ent_x_insp.pack(side="left", padx=(0, 4))
        ttk.Button(insp_frame, text="Ir a x", command=self._actualizar_inspector).pack(side="left", padx=(0, 12))

        self.slider_x = ttk.Scale(insp_frame, from_=0.0, to=30.0, variable=self.x_consulta, command=self._on_slider_move)
        self.slider_x.pack(side="left", fill="x", expand=True, padx=(0, 12))

        self.lbl_res_seccion = ttk.Label(
            insp_frame,
            text="Calcule la estructura para inspeccionar N(x), V(x), M(x) y δ(x)",
            font=("Consolas", 10, "bold"),
            foreground="#003049"
        )
        self.lbl_res_seccion.pack(side="right")

    def _al_cambiar_unidad_fuerza(self):
        self.modelo_calculado = None
        self.curva_global = None
        self._refrescar_lista()
        self._redibujar_esquema()

    def _al_cambiar_unidad_longitud(self):
        nueva_ul = self.u_long.get()
        antigua_ul = self.u_long_prev
        if nueva_ul == antigua_ul:
            return

        conv = lambda val: round(_conv_l(_a_base_l(val, antigua_ul), nueva_ul), 6)

        try:
            L_actual = float(self.longitud_viga.get())
            L_nueva = conv(L_actual)
            self.longitud_viga.set(L_nueva)
            self.slider_x.config(to=L_nueva)
        except ValueError:
            pass

        for a in self.apoyos:
            a['x'] = conv(a['x'])
        self.rotulas = [conv(r) for r in self.rotulas]
        for p in self.puntuales:
            p['x'] = conv(p['x'])
        for m in self.momentos:
            m['x'] = conv(m['x'])
        for d in self.distribuidas:
            d['x1'] = conv(d['x1'])
            d['x2'] = conv(d['x2'])

        self.x_consulta.set(round(conv(float(self.x_consulta.get())), 3))
        self.u_long_prev = nueva_ul
        self.modelo_calculado = None
        self.curva_global = None
        self._refrescar_lista()
        self._redibujar_esquema()

    def _cargar_ejemplo_inicial(self):
        self.u_long.set("m")
        self.u_long_prev = "m"
        self.longitud_viga.set(30.0)
        self.slider_x.config(to=30.0)
        self.apoyos = [
            {'x': 10.0, 'tipo': 'movil', 'ang': 0.0},
            {'x': 30.0, 'tipo': 'empotrado', 'ang': 0.0}
        ]
        self.rotulas = []
        self.distribuidas = [
            {'x1': 0.0, 'x2': 10.0, 'w1': -1.5, 'w2': -1.5, 'ang': -90.0}
        ]
        self.puntuales = []
        self.momentos = []
        self._refrescar_lista()
        self._calcular()

    def _actualizar_longitud(self):
        try:
            L = float(self.longitud_viga.get())
            if L <= 0:
                raise ValueError
            self.slider_x.config(to=L)
            self.modelo_calculado = None
            self.curva_global = None
            self._redibujar_esquema()
        except ValueError:
            messagebox.showerror("Dato inválido", "La longitud L debe ser un número mayor a 0.")

    def _limpiar_todo(self):
        self.apoyos.clear()
        self.rotulas.clear()
        self.puntuales.clear()
        self.momentos.clear()
        self.distribuidas.clear()
        self.modelo_calculado = None
        self.curva_global = None
        self._refrescar_lista()
        self._redibujar_esquema()

    def _refrescar_lista(self):
        self.lista_items.delete(0, tk.END)
        uf, ul = self.u_fuerza.get(), self.u_long.get()
        for i, a in enumerate(self.apoyos):
            self.lista_items.insert(tk.END, f"[APOYO] #{i}: {a['tipo'].upper()} en x={_fmt_num(a['x'], 2)}{ul}")
        for i, r in enumerate(self.rotulas):
            self.lista_items.insert(tk.END, f"[RÓTULA] #{i}: Articulación en x={_fmt_num(r, 2)}{ul}")
        for i, p in enumerate(self.puntuales):
            self.lista_items.insert(tk.END, f"[PUNTUAL] #{i}: P={_fmt_num(abs(p['P']), 2)}{uf} en x={_fmt_num(p['x'], 2)}{ul}")
        for i, m in enumerate(self.momentos):
            self.lista_items.insert(tk.END, f"[MOMENTO] #{i}: M={_fmt_num(m['M'], 2)}{uf}·{ul} en x={_fmt_num(m['x'], 2)}{ul}")
        for i, d in enumerate(self.distribuidas):
            self.lista_items.insert(
                tk.END,
                f"[DISTRIB] #{i}: w1={_fmt_num(d['w1'], 2)} a w2={_fmt_num(d['w2'], 2)}{uf}/{ul} [{_fmt_num(d['x1'], 2)}..{_fmt_num(d['x2'], 2)}{ul}]"
            )

    def _eliminar_seleccionado(self, event=None):
        sel = self.lista_items.curselection()
        if not sel:
            return
        texto = self.lista_items.get(sel)
        _, resto = texto.split("#", 1)
        idx_str, _ = resto.split(":", 1)
        idx = int(idx_str)

        if texto.startswith("[APOYO]"):
            self.apoyos.pop(idx)
        elif texto.startswith("[RÓTULA]"):
            self.rotulas.pop(idx)
        elif texto.startswith("[PUNTUAL]"):
            self.puntuales.pop(idx)
        elif texto.startswith("[MOMENTO]"):
            self.momentos.pop(idx)
        elif texto.startswith("[DISTRIB]"):
            self.distribuidas.pop(idx)

        self.modelo_calculado = None
        self.curva_global = None
        self._refrescar_lista()
        self._redibujar_esquema()

    def _dlg_apoyo(self):
        win = tk.Toplevel(self.root)
        win.title("Agregar Apoyo")
        win.geometry("330x185")
        win.transient(self.root)
        win.grab_set()

        v_x = tk.DoubleVar(value=0.0)
        v_tipo = tk.StringVar(value="fijo")
        v_ang = tk.DoubleVar(value=0.0)

        ttk.Label(win, text=f"Posición x [{self.u_long.get()}]:").grid(row=0, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(win, textvariable=v_x, width=12).grid(row=0, column=1, padx=10, pady=6)
        ttk.Label(win, text="Tipo de apoyo:").grid(row=1, column=0, padx=10, pady=6, sticky="w")
        ttk.Combobox(win, textvariable=v_tipo, values=["fijo", "movil", "empotrado"], width=10, state="readonly").grid(row=1, column=1, padx=10, pady=6)
        ttk.Label(win, text="Ángulo (°):").grid(row=2, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(win, textvariable=v_ang, width=12).grid(row=2, column=1, padx=10, pady=6)

        def ok():
            x = float(v_x.get())
            if x < 0 or x > float(self.longitud_viga.get()):
                messagebox.showerror("Error", "Posición fuera de la viga.", parent=win)
                return
            self.apoyos.append({'x': x, 'tipo': v_tipo.get(), 'ang': float(v_ang.get())})
            self.modelo_calculado = None
            self.curva_global = None
            self._refrescar_lista()
            self._redibujar_esquema()
            win.destroy()

        ttk.Button(win, text="Agregar Apoyo", command=ok).grid(row=3, column=0, columnspan=2, pady=12)

    def _dlg_rotula(self):
        win = tk.Toplevel(self.root)
        win.title("Agregar Rótula")
        win.geometry("310x120")
        win.transient(self.root)
        win.grab_set()

        v_x = tk.DoubleVar(value=round(float(self.longitud_viga.get()) / 2.0, 2))
        ttk.Label(win, text=f"Posición x [{self.u_long.get()}]:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ttk.Entry(win, textvariable=v_x, width=12).grid(row=0, column=1, padx=10, pady=10)

        def ok():
            x = float(v_x.get())
            if x <= 0 or x >= float(self.longitud_viga.get()):
                messagebox.showerror("Error", "La rótula debe estar en el interior de la viga.", parent=win)
                return
            self.rotulas.append(x)
            self.modelo_calculado = None
            self.curva_global = None
            self._refrescar_lista()
            self._redibujar_esquema()
            win.destroy()

        ttk.Button(win, text="Agregar Rótula", command=ok).grid(row=1, column=0, columnspan=2, pady=8)

    def _dlg_puntual(self):
        win = tk.Toplevel(self.root)
        win.title("Agregar Carga Puntual")
        win.geometry("330x185")
        win.transient(self.root)
        win.grab_set()

        v_x = tk.DoubleVar(value=round(float(self.longitud_viga.get()) / 2.0, 2))
        v_p = tk.DoubleVar(value=10.0)
        v_ang = tk.DoubleVar(value=-90.0)

        ttk.Label(win, text=f"Posición x [{self.u_long.get()}]:").grid(row=0, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(win, textvariable=v_x, width=12).grid(row=0, column=1, padx=10, pady=6)
        ttk.Label(win, text=f"Magnitud P [{self.u_fuerza.get()}]:").grid(row=1, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(win, textvariable=v_p, width=12).grid(row=1, column=1, padx=10, pady=6)
        ttk.Label(win, text="Ángulo (-90° abajo):").grid(row=2, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(win, textvariable=v_ang, width=12).grid(row=2, column=1, padx=10, pady=6)

        def ok():
            x = float(v_x.get())
            if x < 0 or x > float(self.longitud_viga.get()):
                messagebox.showerror("Error", "Posición fuera de la viga.", parent=win)
                return
            self.puntuales.append({'x': x, 'P': abs(float(v_p.get())), 'ang': float(v_ang.get())})
            self.modelo_calculado = None
            self.curva_global = None
            self._refrescar_lista()
            self._redibujar_esquema()
            win.destroy()

        ttk.Button(win, text="Agregar Carga Puntual", command=ok).grid(row=3, column=0, columnspan=2, pady=12)

    def _dlg_momento(self):
        win = tk.Toplevel(self.root)
        win.title("Agregar Momento Concentrado")
        win.geometry("330x155")
        win.transient(self.root)
        win.grab_set()

        v_x = tk.DoubleVar(value=round(float(self.longitud_viga.get()) / 2.0, 2))
        v_m = tk.DoubleVar(value=15.0)

        ttk.Label(win, text=f"Posición x [{self.u_long.get()}]:").grid(row=0, column=0, padx=10, pady=8, sticky="w")
        ttk.Entry(win, textvariable=v_x, width=12).grid(row=0, column=1, padx=10, pady=8)
        ttk.Label(win, text="Momento M (+ antihorario):").grid(row=1, column=0, padx=10, pady=8, sticky="w")
        ttk.Entry(win, textvariable=v_m, width=12).grid(row=1, column=1, padx=10, pady=8)

        def ok():
            x = float(v_x.get())
            if x < 0 or x > float(self.longitud_viga.get()):
                messagebox.showerror("Error", "Posición fuera de la viga.", parent=win)
                return
            self.momentos.append({'x': x, 'M': float(v_m.get())})
            self.modelo_calculado = None
            self.curva_global = None
            self._refrescar_lista()
            self._redibujar_esquema()
            win.destroy()

        ttk.Button(win, text="Agregar Momento", command=ok).grid(row=2, column=0, columnspan=2, pady=12)

    def _dlg_distribuida(self):
        win = tk.Toplevel(self.root)
        win.title("Agregar Carga Distribuida")
        win.geometry("350x235")
        win.transient(self.root)
        win.grab_set()

        L = float(self.longitud_viga.get())
        v_x1 = tk.DoubleVar(value=0.0)
        v_x2 = tk.DoubleVar(value=L)
        v_w1 = tk.DoubleVar(value=10.0)
        v_w2 = tk.DoubleVar(value=10.0)
        v_ang = tk.DoubleVar(value=-90.0)

        ttk.Label(win, text=f"Inicio x1 [{self.u_long.get()}]:").grid(row=0, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(win, textvariable=v_x1, width=12).grid(row=0, column=1, padx=10, pady=6)
        ttk.Label(win, text=f"Fin x2 [{self.u_long.get()}]:").grid(row=1, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(win, textvariable=v_x2, width=12).grid(row=1, column=1, padx=10, pady=6)
        ttk.Label(win, text=f"Intensidad w1 [{self.u_fuerza.get()}/{self.u_long.get()}]:").grid(row=2, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(win, textvariable=v_w1, width=12).grid(row=2, column=1, padx=10, pady=6)
        ttk.Label(win, text=f"Intensidad w2 [{self.u_fuerza.get()}/{self.u_long.get()}]:").grid(row=3, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(win, textvariable=v_w2, width=12).grid(row=3, column=1, padx=10, pady=6)
        ttk.Label(win, text="Ángulo (-90° abajo):").grid(row=4, column=0, padx=10, pady=6, sticky="w")
        ttk.Entry(win, textvariable=v_ang, width=12).grid(row=4, column=1, padx=10, pady=6)

        def ok():
            x1, x2 = float(v_x1.get()), float(v_x2.get())
            if x1 < 0 or x2 > L or x2 <= x1:
                messagebox.showerror("Error", "Rango [x1, x2] inválido.", parent=win)
                return
            self.distribuidas.append({'x1': x1, 'x2': x2, 'w1': -abs(float(v_w1.get())), 'w2': -abs(float(v_w2.get())), 'ang': float(v_ang.get())})
            self.modelo_calculado = None
            self.curva_global = None
            self._refrescar_lista()
            self._redibujar_esquema()
            win.destroy()

        ttk.Button(win, text="Agregar Carga Distribuida", command=ok).grid(row=5, column=0, columnspan=2, pady=12)

    def _obtener_puntos_singulares(self):
        L = float(self.longitud_viga.get())
        coords = {0.0, round(L, 6)}
        for a in self.apoyos:
            coords.add(round(a['x'], 6))
        for r in self.rotulas:
            coords.add(round(r, 6))
        for p in self.puntuales:
            coords.add(round(p['x'], 6))
        for m in self.momentos:
            coords.add(round(m['x'], 6))
        for d in self.distribuidas:
            coords.add(round(d['x1'], 6))
            coords.add(round(d['x2'], 6))

        lista_x = sorted(coords)
        letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        mapa = {}
        for idx, x_val in enumerate(lista_x):
            if idx < len(letras):
                mapa[x_val] = letras[idx]
            else:
                mapa[x_val] = f"N{idx+1}"
        return lista_x, mapa

    def _dibujar_momento_curvo(self, ax, x, val_m, L, color, texto, dy_label=0.0):
        r_m = 0.062 * L
        diam_m = 2.0 * r_m
        if val_m >= 0:
            theta1, theta2 = 45.0, 300.0
            ang_punta = np.radians(theta2)
            px = x + r_m * np.cos(ang_punta)
            py = r_m * np.sin(ang_punta)
            tx = -np.sin(ang_punta)
            ty = np.cos(ang_punta)
        else:
            theta1, theta2 = 240.0, 135.0
            ang_punta = np.radians(theta1)
            px = x + r_m * np.cos(ang_punta)
            py = r_m * np.sin(ang_punta)
            tx = np.sin(ang_punta)
            ty = -np.cos(ang_punta)

        arco = Arc(
            (x, 0), width=diam_m, height=diam_m, angle=0,
            theta1=theta1, theta2=theta2, color=color, lw=2.8, zorder=10
        )
        ax.add_patch(arco)

        step = 0.015 * L
        flecha = FancyArrowPatch(
            (px - tx * step, py - ty * step),
            (px + tx * step * 0.45, py + ty * step * 0.45),
            arrowstyle="-|>", mutation_scale=18, color=color, lw=2.6, zorder=11
        )
        ax.add_patch(flecha)

        ax.text(
            x, r_m + 0.028 * L + dy_label, texto,
            color=color, fontsize=10.5, fontweight="bold", ha="center", zorder=12
        )

    def _dibujar_simbolo_apoyo(self, ax, x, tipo, ang_deg, L):
        h = 0.08 * L
        w = 0.045 * L
        c_fill = "#7cc6fe"
        c_edge = "#1d3557"
        if tipo == "fijo":
            tri = Polygon([[x, 0], [x - w, -h], [x + w, -h]], closed=True, facecolor=c_fill, edgecolor=c_edge, lw=1.6, zorder=6)
            ax.add_patch(tri)
            ax.plot([x - 1.45 * w, x + 1.45 * w], [-h, -h], color=c_edge, lw=2.0, zorder=6)
            for k in np.linspace(-1.2 * w, 1.2 * w, 7):
                ax.plot([x + k, x + k - 0.015 * L], [-h, -h - 0.025 * L], color=c_edge, lw=1.1, zorder=5)
        elif tipo in ("movil", "móvil"):
            h_tri = 0.72 * h
            r_rod = 0.14 * h
            tri = Polygon([[x, 0], [x - w, -h_tri], [x + w, -h_tri]], closed=True, facecolor=c_fill, edgecolor=c_edge, lw=1.6, zorder=6)
            ax.add_patch(tri)
            ax.plot([x - 1.25 * w, x + 1.25 * w], [-h_tri, -h_tri], color=c_edge, lw=1.6, zorder=6)
            y_rod = -h_tri - r_rod
            for dx in [-0.7 * w, 0.0, 0.7 * w]:
                rod = Circle((x + dx, y_rod), radius=r_rod, facecolor="white", edgecolor=c_edge, lw=1.4, zorder=7)
                ax.add_patch(rod)
            y_suelo = y_rod - r_rod
            ax.plot([x - 1.45 * w, x + 1.45 * w], [y_suelo, y_suelo], color=c_edge, lw=2.0, zorder=6)
            for k in np.linspace(-1.2 * w, 1.2 * w, 7):
                ax.plot([x + k, x + k - 0.015 * L], [y_suelo, y_suelo - 0.025 * L], color=c_edge, lw=1.1, zorder=5)
        elif tipo == "empotrado":
            lado = -1.0 if x <= 1e-6 else 1.0
            hw = 0.095 * L
            ancho_muro = 0.025 * L
            rect = Polygon(
                [[x, -hw], [x + lado * ancho_muro, -hw], [x + lado * ancho_muro, hw], [x, hw]],
                closed=True, facecolor="#dce1e9", edgecolor="none", zorder=5
            )
            ax.add_patch(rect)
            ax.plot([x, x], [-hw, hw], color=c_edge, lw=3.0, zorder=7)
            for yk in np.linspace(-hw + 0.015 * L, hw - 0.015 * L, 9):
                ax.plot([x, x + lado * ancho_muro], [yk, yk - 0.022 * L], color=c_edge, lw=1.2, zorder=6)

    def _redibujar_esquema(self):
        self.ax_esq.clear()
        L = float(self.longitud_viga.get())
        uf, ul = self.u_fuerza.get(), self.u_long.get()
        lista_x, self.mapa_letras = self._obtener_puntos_singulares()
        self.ax_esq.plot([0, L], [0, 0], color="#d62828", lw=5.5, solid_capstyle="butt", zorder=5)
        for x_val in lista_x:
            letra = self.mapa_letras[x_val]
            self.ax_esq.plot(x_val, 0, "ko", markersize=4.5, zorder=8)
            self.ax_esq.text(
                x_val, -0.028 * L, letra,
                ha="center", va="top", fontsize=11, fontweight="bold", color="#14213d",
                bbox=dict(boxstyle="circle,pad=0.24", facecolor="#fff3b0", edgecolor="#14213d", lw=1.2),
                zorder=11
            )
        for a in self.apoyos:
            self._dibujar_simbolo_apoyo(self.ax_esq, a['x'], a['tipo'], a['ang'], L)
        for rx in self.rotulas:
            self.ax_esq.plot(rx, 0, "o", mfc="white", mec="#14213d", mew=2.2, markersize=10, zorder=9)
        max_w = max([max(abs(d['w1']), abs(d['w2'])) for d in self.distribuidas], default=1.0)
        h_max_w = 0.17 * L
        for d in self.distribuidas:
            x1, x2 = d['x1'], d['x2']
            h1 = (abs(d['w1']) / max_w) * h_max_w
            h2 = (abs(d['w2']) / max_w) * h_max_w
            poly = Polygon(
                [[x1, 0], [x1, h1], [x2, h2], [x2, 0]], closed=True,
                facecolor="#2a9d8f", alpha=0.20, edgecolor="#2a9d8f", lw=1.8, zorder=3
            )
            self.ax_esq.add_patch(poly)
            n_fl = max(3, int(round((x2 - x1) / (L / 14.0))))
            xs_fl = np.linspace(x1, x2, n_fl)
            hs_fl = np.linspace(h1, h2, n_fl)
            for xf, hf in zip(xs_fl, hs_fl):
                if hf > 0.015 * L:
                    self.ax_esq.annotate(
                        "", xy=(xf, 0.008 * L), xytext=(xf, hf),
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color="#2a9d8f", lw=1.6)
                    )
            self.ax_esq.text(
                x1, h1 + 0.02 * L, f"{_fmt_num(abs(d['w1']), 2)} {uf}/{ul}",
                color="#1b4332", fontsize=10, fontweight="bold"
            )
            if abs(d['w2'] - d['w1']) > 1e-3:
                self.ax_esq.text(
                    x2, h2 + 0.02 * L, f"{_fmt_num(abs(d['w2']), 2)} {uf}/{ul}",
                    color="#1b4332", fontsize=10, fontweight="bold"
                )
        for p in self.puntuales:
            xp = p['x']
            rad = np.radians(p['ang'])
            dx = -np.cos(rad) * 0.17 * L
            dy = -np.sin(rad) * 0.17 * L
            self.ax_esq.annotate(
                "", xy=(xp, 0.005 * L), xytext=(xp + dx, dy),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=18, color="#e63946", lw=2.8), zorder=10
            )
            self.ax_esq.text(
                xp + dx, dy + 0.018 * L, f"P={_fmt_num(p['P'], 2)} {uf}",
                color="#e63946", fontsize=10.5, fontweight="bold", ha="center"
            )
        for m in self.momentos:
            xm = m['x']
            self._dibujar_momento_curvo(
                self.ax_esq, xm, m['M'], L, color="#7209b7",
                texto=f"M={_fmt_num(abs(m['M']), 2)} {uf}·{ul}"
            )
        if self.modelo_calculado is not None:
            est = self.modelo_calculado
            nodos_ord = sorted(est.nodos.keys())
            mapa_idx = {nid: i for i, nid in enumerate(nodos_ord)}
            for nid in sorted(est.apoyos.keys()):
                idx = mapa_idx[nid] * 3
                x_ap = _conv_l(_x_nodo(est.nodos[nid]), ul)
                letra = self.mapa_letras.get(round(x_ap, 6), str(nid))
                rx = _conv_f(est.R.item(idx), uf)
                ry = _conv_f(est.R.item(idx + 1), uf)
                mz = _conv_m(est.R.item(idx + 2), uf, ul)
                if abs(ry) > 1e-3:
                    if ry > 0:
                        self.ax_esq.annotate(
                            "", xy=(x_ap, -0.092 * L), xytext=(x_ap, -0.23 * L),
                            arrowprops=dict(arrowstyle="-|>", mutation_scale=20, color="#f77f00", lw=3.0), zorder=10
                        )
                    else:
                        self.ax_esq.annotate(
                            "", xy=(x_ap, -0.23 * L), xytext=(x_ap, -0.092 * L),
                            arrowprops=dict(arrowstyle="-|>", mutation_scale=20, color="#f77f00", lw=3.0), zorder=10
                        )
                    self.ax_esq.text(
                        x_ap, -0.27 * L, f"R{letra}y = {_fmt_num(abs(ry), 2)} {uf}", color="#d62828",
                        fontsize=10.5, fontweight="bold", ha="center"
                    )
                if abs(rx) > 1e-3:
                    dir_x = 1.0 if rx > 0 else -1.0
                    self.ax_esq.annotate(
                        "", xy=(x_ap - 0.02 * L * dir_x, 0.03 * L),
                        xytext=(x_ap - 0.16 * L * dir_x, 0.03 * L),
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=18, color="#f77f00", lw=2.8), zorder=10
                    )
                    self.ax_esq.text(
                        x_ap - 0.09 * L * dir_x, 0.06 * L, f"R{letra}x = {_fmt_num(abs(rx), 2)} {uf}",
                        color="#d62828", fontsize=10, fontweight="bold", ha="center"
                    )
                if abs(mz) > 1e-3:
                    self._dibujar_momento_curvo(
                        self.ax_esq, x_ap, mz, L, color="#f77f00",
                        texto=f"M{letra} = {_fmt_num(abs(mz), 2)} {uf}·{ul}", dy_label=0.035 * L
                    )
        self.ax_esq.set_xlim(-0.15 * L, 1.15 * L)
        self.ax_esq.set_ylim(-0.32 * L, 0.30 * L)
        self.ax_esq.set_xlabel(f"Posición a lo largo de la viga [{ul}]", fontsize=10.5, fontweight="bold")
        self.ax_esq.tick_params(axis="x", labelsize=10)
        self.ax_esq.set_yticks([])
        self.ax_esq.grid(True, axis="x", linestyle="--", alpha=0.4)
        self.ax_esq.set_title("Esquema Estructural, Nudos (A, B, C...), Cargas y Reacciones", fontsize=12, fontweight="bold", pad=10)
        self.fig_esq.tight_layout()
        self.canvas_esq.draw()

    def _calcular(self):
        if not self.apoyos:
            messagebox.showwarning("Sin apoyos", "Agregue al menos un apoyo antes de calcular.")
            return
        L = float(self.longitud_viga.get())
        uf, ul = self.u_fuerza.get(), self.u_long.get()
        E, I, A = float(self.E_val.get()), float(self.I_val.get()), float(self.A_val.get())
        lista_x, self.mapa_letras = self._obtener_puntos_singulares()
        est = Estructura2D(u_fuerza=uf, u_long=ul)
        mapa_nodos = {}
        for idx, x_val in enumerate(lista_x):
            nid = idx + 1
            mapa_nodos[x_val] = nid
            est.agregar_nodo(nid, x_val, 0.0)
        for a in self.apoyos:
            nid = mapa_nodos[round(a['x'], 6)]
            est.agregar_apoyo(nid, tipo=a['tipo'], angulo_deg=a['ang'])
        for p in self.puntuales:
            nid = mapa_nodos[round(p['x'], 6)]
            est.agregar_carga_nodal_inclinada(nid, magnitud=p['P'], angulo_deg=p['ang'])
        for m in self.momentos:
            nid = mapa_nodos[round(m['x'], 6)]
            est.agregar_carga_nodal(nid, mz=m['M'])
        rot_set = {round(r, 6) for r in self.rotulas}
        for k in range(len(lista_x) - 1):
            xa, xb = lista_x[k], lista_x[k + 1]
            ni, nj = mapa_nodos[xa], mapa_nodos[xb]
            art_i = (xa in rot_set)
            art_j = (xb in rot_set)
            w1_tot, w2_tot, ang_c = 0.0, 0.0, None
            for d in self.distribuidas:
                if d['x1'] <= xa + 1e-6 and d['x2'] >= xb - 1e-6:
                    span = d['x2'] - d['x1']
                    wa = d['w1'] + ((xa - d['x1']) / span) * (d['w2'] - d['w1'])
                    wb = d['w1'] + ((xb - d['x1']) / span) * (d['w2'] - d['w1'])
                    w1_tot += abs(wa)
                    w2_tot += abs(wb)
                    ang_c = d['ang']
            est.agregar_elemento(
                k + 1, ni, nj, E=E, A=A, I=I, w1=w1_tot, w2=w2_tot,
                angulo_carga_deg=ang_c, art_i=art_i, art_j=art_j
            )
        try:
            est.resolver()
        except Exception as ex:
            messagebox.showerror("Estructura Inestable / Error", f"No se pudo resolver el sistema: {ex}")
            return
        self.modelo_calculado = est
        self._construir_curvas_globales(est, lista_x)
        self._redibujar_esquema()
        self._dibujar_diagramas()
        self._llenar_memoria_academica(est, lista_x)
        self._actualizar_inspector()

    def _construir_curvas_globales(self, est, lista_x):
        xs_tot, N_tot, V_tot, M_tot, D_tot = [], [], [], [], []
        ul, uf = self.u_long.get(), self.u_fuerza.get()
        nodos_ord = sorted(est.nodos.keys())
        mapa_idx = {nid: i for i, nid in enumerate(nodos_ord)}
        for elem in est.elementos:
            it_res = iter(est.resultados[elem['id']])
            f_int = next(it_res)
            L_m = next(it_res)
            wx1, wx2, wy1, wy2 = next(it_res)
            x1_m = _x_nodo(est.nodos[elem['ni']])
            xs = np.linspace(0.0, L_m, 100)
            dwx, dwy = wx2 - wx1, wy2 - wy1
            Ni, Vi, Mi = -f_int.item(0), f_int.item(1), -f_int.item(2)
            N_x = Ni - (wx1 * xs + (dwx / (2.0 * L_m)) * (xs ** 2))
            V_x = Vi + wy1 * xs + (dwy / (2.0 * L_m)) * (xs ** 2)
            M_x = Mi + Vi * xs + 0.5 * wy1 * (xs ** 2) + (dwy / (6.0 * L_m)) * (xs ** 3)
            idx_i = mapa_idx[elem['ni']] * 3
            idx_j = mapa_idx[elem['nj']] * 3
            v1, th1 = est.U.item(idx_i + 1), est.U.item(idx_i + 2)
            v2, th2 = est.U.item(idx_j + 1), est.U.item(idx_j + 2)
            xi = xs / L_m
            N1 = 1.0 - 3.0 * (xi ** 2) + 2.0 * (xi ** 3)
            N2 = L_m * (xi - 2.0 * (xi ** 2) + xi ** 3)
            N3 = 3.0 * (xi ** 2) - 2.0 * (xi ** 3)
            N4 = L_m * (-(xi ** 2) + xi ** 3)
            defl_m = N1 * v1 + N2 * th1 + N3 * v2 + N4 * th2
            xs_tot.append(_conv_l(x1_m + xs, ul))
            N_tot.append(_conv_f(N_x, uf))
            V_tot.append(_conv_f(V_x, uf))
            M_tot.append(_conv_m(M_x, uf, ul))
            D_tot.append(_conv_l(defl_m, ul))
        self.curva_global = (
            np.concatenate(xs_tot),
            np.concatenate(N_tot),
            np.concatenate(V_tot),
            np.concatenate(M_tot),
            np.concatenate(D_tot)
        )

    def _dibujar_diagramas(self):
        if self.curva_global is None:
            return
        xs, Ns, Vs, Ms, Ds = self.curva_global
        uf, ul = self.u_fuerza.get(), self.u_long.get()
        L = float(self.longitud_viga.get())
        datos = [
            (Ns, f"Axial N [{uf}]", "#1f77b4", False),
            (Vs, f"Cortante V [{uf}]", "#2a9d8f", False),
            (Ms, f"Momento M [{uf}·{ul}]", "#d62828", True),
            (Ds, f"Deflexión δ [{ul}]", "#7209b7", False)
        ]
        for ax, (ys, titulo, col, inv) in zip(self.axes_diag, datos):
            ax.clear()
            ax.plot([0, L], [0, 0], "k-", lw=1.2)
            ax.plot(xs, ys, color=col, lw=2.0)
            ax.fill_between(xs, 0, ys, color=col, alpha=0.22)
            ax.set_ylabel(titulo, fontsize=9.5, fontweight="bold")
            ax.tick_params(axis="both", labelsize=9.5)
            ax.grid(True, linestyle="--", alpha=0.4)
            if inv:
                ax.invert_yaxis()
            idx_max = int(np.argmax(np.abs(ys)))
            if abs(ys.item(idx_max)) > 1e-4:
                ax.plot(xs.item(idx_max), ys.item(idx_max), "o", color=col, markersize=6)
                ax.text(
                    xs.item(idx_max), ys.item(idx_max),
                    f" {_fmt_num(ys.item(idx_max), 2)}",
                    color=col, fontsize=10, fontweight="bold"
                )
        x_cur = float(self.x_consulta.get())
        for ax in self.axes_diag:
            ax.axvline(x_cur, color="#fca311", linestyle="--", lw=1.8)
        self.axes_diag[-1].set_xlabel(f"Posición x [{ul}]", fontsize=10.5, fontweight="bold")
        self.fig_diag.tight_layout()
        self.canvas_diag.draw()

    def _on_slider_move(self, val):
        self.x_consulta.set(round(float(val), 3))
        self._actualizar_inspector()

    def _actualizar_inspector(self):
        if self.curva_global is None:
            return
        L = float(self.longitud_viga.get())
        x_q = max(0.0, min(L, float(self.x_consulta.get())))
        self.x_consulta.set(round(x_q, 3))
        xs, Ns, Vs, Ms, Ds = self.curva_global
        n_v = float(np.interp(x_q, xs, Ns))
        v_v = float(np.interp(x_q, xs, Vs))
        m_v = float(np.interp(x_q, xs, Ms))
        d_v = float(np.interp(x_q, xs, Ds))
        uf, ul = self.u_fuerza.get(), self.u_long.get()
        self.lbl_res_seccion.config(
            text=f"x={_fmt_num(x_q, 3)}{ul}  |  N={_fmt_num(n_v, 2)}{uf}  |  V={_fmt_num(v_v, 2)}{uf}  |  M={_fmt_num(m_v, 2)}{uf}·{ul}  |  δ={_fmt_num(d_v, 5)}{ul}"
        )
        self._dibujar_diagramas()

    def _llenar_memoria_academica(self, est, lista_x):
        self.txt_memoria.delete("1.0", tk.END)
        uf, ul = self.u_fuerza.get(), self.u_long.get()
        sep = "=" * 78
        sub = "-" * 78
        L_lines = [
            sep,
            " VMR INGENIERÍA APLICADA  -  MEMORIA DE CÁLCULO ESTRUCTURAL (PASO A PASO)",
            " Autor: Ing. Víctor Martínez Rodríguez  |  Método Matricial de la Rigidez",
            sep,
            f"Unidades de trabajo: Fuerza = [{uf}], Longitud = [{ul}]",
            f"Propiedades del material y sección: E = {_fmt_num(self.E_val.get(), 2)} {uf}/{ul}^2, I = {_fmt_num(self.I_val.get(), 6)} {ul}^4",
            "",
            "PASO 1: IDENTIFICACIÓN DE NUDOS Y CONDICIONES DE APOYO",
            sub
        ]
        nodos_ord = sorted(est.nodos.keys())
        mapa_idx = {nid: i for i, nid in enumerate(nodos_ord)}
        for nid in nodos_ord:
            x_u = _conv_l(_x_nodo(est.nodos[nid]), ul)
            letra = self.mapa_letras.get(round(x_u, 6), str(nid))
            desc_ap = "Nudo libre"
            if nid in est.apoyos:
                desc_ap = f"Apoyo {est.apoyos[nid]['tipo'].upper()}"
            L_lines.append(f"  Nudo {letra} (Nodo interno #{nid}): x = {_fmt_num(x_u, 3):>10s} {ul}   ->   {desc_ap}")
        L_lines.extend(["", "PASO 2: DISCRETIZACIÓN POR TRAMOS Y RIGIDEZ A FLEXIÓN", sub])
        for elem in est.elementos:
            x1 = _conv_l(_x_nodo(est.nodos[elem['ni']]), ul)
            x2 = _conv_l(_x_nodo(est.nodos[elem['nj']]), ul)
            la = self.mapa_letras.get(round(x1, 6), str(elem['ni']))
            lb = self.mapa_letras.get(round(x2, 6), str(elem['nj']))
            long_t = x2 - x1
            L_lines.append(f"  Tramo {la}{lb} (Longitud L = {_fmt_num(long_t, 3)} {ul}):")
            L_lines.append(f"    Rigidez a flexión (EI/L) = {_fmt_num(elem['E']*elem['I']/long_t, 3)} {uf}·{ul}")
        L_lines.extend(["", "PASO 3: DESPLAZAMIENTOS Y GIROS NODALES CALCULADOS ([K]{U} = {F})", sub])
        for nid in nodos_ord:
            idx = mapa_idx[nid] * 3
            x_u = _conv_l(_x_nodo(est.nodos[nid]), ul)
            letra = self.mapa_letras.get(round(x_u, 6), str(nid))
            ux = _conv_l(est.U.item(idx), ul)
            uy = _conv_l(est.U.item(idx + 1), ul)
            rz = est.U.item(idx + 2)
            L_lines.append(
                f"  Nudo {letra}:  Δx = {_fmt_num(ux, 6):>12s} {ul}  |  Δy = {_fmt_num(uy, 6):>12s} {ul}  |  θz = {_fmt_num(rz, 6):>12s} rad"
            )
        L_lines.extend(["", "PASO 4: REACCIONES EN LOS APOYOS Y VERIFICACIÓN DE EQUILIBRIO", sub])
        for nid in sorted(est.apoyos.keys()):
            idx = mapa_idx[nid] * 3
            x_u = _conv_l(_x_nodo(est.nodos[nid]), ul)
            letra = self.mapa_letras.get(round(x_u, 6), str(nid))
            rx = _conv_f(est.R.item(idx), uf)
            ry = _conv_f(est.R.item(idx + 1), uf)
            mz = _conv_m(est.R.item(idx + 2), uf, ul)
            tipo = est.apoyos[nid]['tipo'].upper()
            L_lines.append(
                f"  Apoyo {letra} ({tipo:9s}):  R{letra}x = {_fmt_num(rx, 3):>11s} {uf}  |  R{letra}y = {_fmt_num(ry, 3):>11s} {uf}  |  M{letra} = {_fmt_num(mz, 3):>11s} {uf}·{ul}"
            )
        L_lines.extend(["", "PASO 5: ESFUERZOS INTERNOS EN EXTREMOS DE CADA TRAMO", sub])
        for elem in est.elementos:
            f_int = next(iter(est.resultados[elem['id']]))
            x1 = _conv_l(_x_nodo(est.nodos[elem['ni']]), ul)
            x2 = _conv_l(_x_nodo(est.nodos[elem['nj']]), ul)
            la = self.mapa_letras.get(round(x1, 6), str(elem['ni']))
            lb = self.mapa_letras.get(round(x2, 6), str(elem['nj']))
            Ni = _conv_f(-f_int.item(0), uf)
            Vi = _conv_f(f_int.item(1), uf)
            Mi = _conv_m(-f_int.item(2), uf, ul)
            Nj = _conv_f(f_int.item(3), uf)
            Vj = _conv_f(-f_int.item(4), uf)
            Mj = _conv_m(f_int.item(5), uf, ul)
            L_lines.append(f"  Tramo {la}{lb}:")
            L_lines.append(f"    En {la}:  N = {_fmt_num(Ni, 3):>11s} {uf}  |  V = {_fmt_num(Vi, 3):>11s} {uf}  |  M = {_fmt_num(Mi, 3):>11s} {uf}·{ul}")
            L_lines.append(f"    En {lb}:  N = {_fmt_num(Nj, 3):>11s} {uf}  |  V = {_fmt_num(Vj, 3):>11s} {uf}  |  M = {_fmt_num(Mj, 3):>11s} {uf}·{ul}")
        L_lines.append(sep)
        self.txt_memoria.insert("1.0", "\n".join(L_lines))

    def _exportar_png(self):
        ruta = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Imagen PNG", "*.png")], title="Guardar Lámina VMR Ingeniería Aplicada")
        if not ruta:
            return
        tab_actual = self.notebook.index(self.notebook.select())
        if tab_actual == 0:
            self.fig_esq.savefig(ruta, dpi=300)
        else:
            self.fig_diag.savefig(ruta, dpi=300)
        messagebox.showinfo("Exportado", f"Lámina guardada en alta resolución (300 DPI): {ruta}")


def main():
    root = tk.Tk()
    app = VMRIngenieriaAplicadaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
