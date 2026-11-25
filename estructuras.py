import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# 1. CONVERSOR DE UNIDADES (SISTEMA INTERNACIONAL E INGLES)
# ==============================================================================
class ConversorUnidades:
    FUERZA = {
        'kN': 1.0,
        'tf': 9.80665,
        'kgf': 0.00980665,
        'lbf': 0.0044482216,
        'kip': 4.4482216153
    }
    LONGITUD = {
        'm': 1.0,
        'cm': 0.01,
        'mm': 0.001,
        'ft': 0.3048,
        'in': 0.0254
    }

    @classmethod
    def a_base_f(cls, val, u):
        return val * cls.FUERZA[u]

    @classmethod
    def a_base_l(cls, val, u):
        return val * cls.LONGITUD[u]

    @classmethod
    def a_base_m(cls, val, uf, ul):
        return val * cls.FUERZA[uf] * cls.LONGITUD[ul]

    @classmethod
    def a_base_w(cls, val, uf, ul):
        return val * cls.FUERZA[uf] / cls.LONGITUD[ul]

    @classmethod
    def de_base_f(cls, val, u):
        return val / cls.FUERZA[u]

    @classmethod
    def de_base_l(cls, val, u):
        return val / cls.LONGITUD[u]

    @classmethod
    def de_base_m(cls, val, uf, ul):
        return val / (cls.FUERZA[uf] * cls.LONGITUD[ul])


# ==============================================================================
# 2. MOTOR DE ANÁLISIS ESTRUCTURAL 2D - VMR ESTRUCTURAL (v3.0)
# ==============================================================================
class Estructura2D:
    def __init__(self, u_fuerza='kN', u_long='m'):
        self.u_f = u_fuerza
        self.u_l = u_long
        self.nodos = {}
        self.elementos = []
        self.apoyos = {}
        self.cargas_nodales = {}
        self.resultados = {}
        self.U = None
        self.R = None
        self._mapa_nodos = {}

    def agregar_nodo(self, id_nodo, x, y, u_long=None):
        ul = u_long or self.u_l
        self.nodos[id_nodo] = (
            ConversorUnidades.a_base_l(x, ul),
            ConversorUnidades.a_base_l(y, ul)
        )
        if id_nodo not in self.cargas_nodales:
            self.cargas_nodales[id_nodo] = np.zeros(3, dtype=float)

    def agregar_apoyo(self, id_nodo, tipo='empotrado', angulo_deg=0.0):
        t = tipo.lower()
        if t == 'empotrado':
            rest = (True, True, True)
        elif t == 'fijo':
            rest = (True, True, False)
        elif t in ('movil', 'móvil', 'rodillo'):
            rest = (False, True, False)
        else:
            raise ValueError(f"Tipo de apoyo desconocido: {tipo}")
        self.apoyos[id_nodo] = {
            'tipo': t,
            'rest': rest,
            'angulo': np.radians(angulo_deg)
        }

    def agregar_apoyo_en_distancia(self, id_elem, distancia, tipo='movil', angulo_deg=0.0, u_long=None):
        ul = u_long or self.u_l
        d = ConversorUnidades.a_base_l(distancia, ul)
        elem = next((e for e in self.elementos if e['id'] == id_elem), None)
        if elem is None:
            raise ValueError(f"Elemento {id_elem} no encontrado.")
        x1, y1 = self.nodos[elem['ni']]
        x2, y2 = self.nodos[elem['nj']]
        L = np.hypot(x2 - x1, y2 - y1)
        if d <= 1e-6 or d >= L - 1e-6:
            raise ValueError("La distancia debe estar estrictamente entre 0 y L.")
        nuevo_id = max(self.nodos.keys()) + 1
        r = d / L
        self.agregar_nodo(nuevo_id, x1 + r * (x2 - x1), y1 + r * (y2 - y1), u_long='m')
        wm = elem['w1'] + r * (elem['w2'] - elem['w1'])
        e1 = dict(elem, id=f"{elem['id']}_1", nj=nuevo_id, w2=wm, art_j=False,
                  puntuales=[p for p in elem['puntuales'] if p['a'] < d],
                  momentos=[m for m in elem['momentos'] if m['a'] < d])
        e2 = dict(elem, id=f"{elem['id']}_2", ni=nuevo_id, w1=wm, art_i=False,
                  puntuales=[dict(p, a=p['a'] - d) for p in elem['puntuales'] if p['a'] >= d],
                  momentos=[dict(m, a=m['a'] - d) for m in elem['momentos'] if m['a'] >= d])
        idx = self.elementos.index(elem)
        self.elementos.pop(idx)
        self.elementos.insert(idx, e1)
        self.elementos.insert(idx + 1, e2)
        self.agregar_apoyo(nuevo_id, tipo=tipo, angulo_deg=angulo_deg)
        return nuevo_id

    def agregar_elemento(self, id_elem, ni, nj, E=2.1e7, A=0.05, I=0.001,
                         w1=0.0, w2=None, angulo_carga_deg=None,
                         art_i=False, art_j=False, u_fuerza=None, u_long=None):
        uf = u_fuerza or self.u_f
        ul = u_long or self.u_l
        if w2 is None:
            w2 = w1
        self.elementos.append({
            'id': id_elem, 'ni': ni, 'nj': nj, 'E': E, 'A': A, 'I': I,
            'w1': ConversorUnidades.a_base_w(w1, uf, ul),
            'w2': ConversorUnidades.a_base_w(w2, uf, ul),
            'ang_carga': np.radians(angulo_carga_deg) if angulo_carga_deg is not None else None,
            'art_i': art_i, 'art_j': art_j, 'puntuales': [], 'momentos': []
        })

    def agregar_carga_puntual_barra(self, id_elem, magnitud, distancia, angulo_deg=None, u_fuerza=None, u_long=None):
        uf = u_fuerza or self.u_f
        ul = u_long or self.u_l
        elem = next((e for e in self.elementos if e['id'] == id_elem), None)
        if elem is None:
            raise ValueError(f"Elemento {id_elem} no encontrado.")
        elem['puntuales'].append({
            'P': ConversorUnidades.a_base_f(magnitud, uf),
            'a': ConversorUnidades.a_base_l(distancia, ul),
            'ang': np.radians(angulo_deg) if angulo_deg is not None else None
        })

    def agregar_momento_barra(self, id_elem, momento, distancia, u_fuerza=None, u_long=None):
        uf = u_fuerza or self.u_f
        ul = u_long or self.u_l
        elem = next((e for e in self.elementos if e['id'] == id_elem), None)
        if elem is None:
            raise ValueError(f"Elemento {id_elem} no encontrado.")
        elem['momentos'].append({
            'M': ConversorUnidades.a_base_m(momento, uf, ul),
            'a': ConversorUnidades.a_base_l(distancia, ul)
        })

    def agregar_carga_nodal(self, id_nodo, fx=0.0, fy=0.0, mz=0.0, u_fuerza=None, u_long=None):
        uf = u_fuerza or self.u_f
        ul = u_long or self.u_l
        self.cargas_nodales[id_nodo] += np.array([
            ConversorUnidades.a_base_f(fx, uf),
            ConversorUnidades.a_base_f(fy, uf),
            ConversorUnidades.a_base_m(mz, uf, ul)
        ], dtype=float)

    def agregar_carga_nodal_inclinada(self, id_nodo, magnitud, angulo_deg, mz=0.0, u_fuerza=None, u_long=None):
        rad = np.radians(angulo_deg)
        self.agregar_carga_nodal(id_nodo, fx=magnitud * np.cos(rad), fy=magnitud * np.sin(rad),
                                 mz=mz, u_fuerza=u_fuerza, u_long=u_long)

    def _formular_elemento(self, elem):
        x1, y1 = self.nodos[elem['ni']]
        x2, y2 = self.nodos[elem['nj']]
        L = np.hypot(x2 - x1, y2 - y1)
        c = (x2 - x1) / L
        s = (y2 - y1) / L

        w1, w2 = elem['w1'], elem['w2']
        if elem['ang_carga'] is None:
            wx1, wx2, wy1, wy2 = 0.0, 0.0, w1, w2
        else:
            th = elem['ang_carga']
            ca, sa = np.cos(th), np.sin(th)
            px = ca * c + sa * s
            py = -ca * s + sa * c
            wx1, wx2, wy1, wy2 = w1 * px, w2 * px, w1 * py, w2 * py

        dwx, dwy = wx2 - wx1, wy2 - wy1
        Ni_emp = -(wx1 * L / 2.0 + dwx * L / 6.0)
        Nj_emp = -(wx1 * L / 2.0 + dwx * L / 3.0)
        Vi_emp = -(wy1 * L / 2.0 + 3.0 * dwy * L / 20.0)
        Vj_emp = -(wy1 * L / 2.0 + 7.0 * dwy * L / 20.0)
        Mi_emp = -(wy1 * (L**2) / 12.0 + dwy * (L**2) / 30.0)
        Mj_emp =  (wy1 * (L**2) / 12.0 + dwy * (L**2) / 20.0)

        puntuales_loc = []
        for p in elem['puntuales']:
            a, b = p['a'], L - p['a']
            if p['ang'] is None:
                Px, Py = 0.0, p['P']
            else:
                ca, sa = np.cos(p['ang']), np.sin(p['ang'])
                Px = p['P'] * (ca * c + sa * s)
                Py = p['P'] * (-ca * s + sa * c)
            puntuales_loc.append((a, Px, Py))
            Ni_emp -= Px * (b / L)
            Nj_emp -= Px * (a / L)
            Vi_emp -= Py * (b**2) * (3*a + b) / (L**3)
            Vj_emp -= Py * (a**2) * (a + 3*b) / (L**3)
            Mi_emp -= Py * a * (b**2) / (L**2)
            Mj_emp += Py * (a**2) * b / (L**2)

        for m in elem['momentos']:
            a, b, M0 = m['a'], L - m['a'], m['M']
            Vi_emp -= 6.0 * M0 * a * b / (L**3)
            Vj_emp += 6.0 * M0 * a * b / (L**3)
            Mi_emp -= M0 * b * (2*a - b) / (L**2)
            Mj_emp -= M0 * a * (2*b - a) / (L**2)

        f_emp = np.array([Ni_emp, Vi_emp, Mi_emp, Nj_emp, Vj_emp, Mj_emp], dtype=float)
        E, A, I = elem['E'], elem['A'], elem['I']
        EA_L, EI_L3, EI_L2, EI_L = E * A / L, E * I / (L**3), E * I / (L**2), E * I / L

        k_loc = np.array([
            [ EA_L,       0.0,      0.0, -EA_L,       0.0,      0.0],
            [  0.0,  12*EI_L3,  6*EI_L2,   0.0, -12*EI_L3,  6*EI_L2],
            [  0.0,   6*EI_L2,   4*EI_L,   0.0,  -6*EI_L2,   2*EI_L],
            [-EA_L,       0.0,      0.0,  EA_L,       0.0,      0.0],
            [  0.0, -12*EI_L3, -6*EI_L2,   0.0,  12*EI_L3, -6*EI_L2],
            [  0.0,   6*EI_L2,   2*EI_L,   0.0,  -6*EI_L2,   4*EI_L]
        ], dtype=float)

        art_i, art_j = elem['art_i'], elem['art_j']
        if art_i and art_j:
            v_corr = (Mi_emp + Mj_emp) / L
            k_loc = np.array([
                [ EA_L, 0.0, 0.0, -EA_L, 0.0, 0.0],
                [  0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
                [  0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
                [-EA_L, 0.0, 0.0,  EA_L, 0.0, 0.0],
                [  0.0, 0.0, 0.0,   0.0, 0.0, 0.0],
                [  0.0, 0.0, 0.0,   0.0, 0.0, 0.0]
            ], dtype=float)
            f_emp = np.array([Ni_emp, Vi_emp - v_corr, 0.0, Nj_emp, Vj_emp + v_corr, 0.0], dtype=float)
        elif art_i or art_j:
            gdl_rot = 2 if art_i else 5
            act = [i for i in range(6) if i != gdl_rot]
            k_aa = k_loc[np.ix_(act, act)]
            k_ab = k_loc[act, gdl_rot:gdl_rot+1]
            k_ba = k_loc[gdl_rot:gdl_rot+1, act]
            k_bb = k_loc[gdl_rot, gdl_rot]
            k_cond = k_aa - (k_ab @ k_ba) / k_bb
            f_cond = f_emp[act] - (k_ab[:, 0] / k_bb) * f_emp[gdl_rot]
            k_loc = np.zeros((6, 6), dtype=float)
            k_loc[np.ix_(act, act)] = k_cond
            f_emp = np.zeros(6, dtype=float)
            f_emp[act] = f_cond

        T = np.eye(6, dtype=float)
        T[0:2, 0:2] = np.array([[c, s], [-s, c]])
        T[3:5, 3:5] = np.array([[c, s], [-s, c]])
        return k_loc, T.T @ k_loc @ T, f_emp, T.T @ f_emp, T, L, (wx1, wx2, wy1, wy2), puntuales_loc

    def resolver(self, imprimir=True):
        self.resultados = {}
        nodos_ord = sorted(self.nodos.keys())
        mapa = {nid: i for i, nid in enumerate(nodos_ord)}
        self._mapa_nodos = mapa
        ngdl = 3 * len(nodos_ord)
        K = np.zeros((ngdl, ngdl), dtype=float)
        F_nodal = np.zeros(ngdl, dtype=float)
        F_emp_tot = np.zeros(ngdl, dtype=float)

        for nid, cargas in self.cargas_nodales.items():
            idx = mapa[nid] * 3
            F_nodal[idx:idx+3] = cargas

        F_sis = F_nodal.copy()
        datos_elem = []
        for elem in self.elementos:
            k_loc, k_g, f_emp_l, f_emp_g, T, L, w_comp, p_loc = self._formular_elemento(elem)
            datos_elem.append((elem, k_loc, f_emp_l, T, L, w_comp, p_loc))
            i1, i2 = mapa[elem['ni']] * 3, mapa[elem['nj']] * 3
            gdl = [i1, i1+1, i1+2, i2, i2+1, i2+2]
            F_sis[gdl] -= f_emp_g
            F_emp_tot[gdl] += f_emp_g
            K[np.ix_(gdl, gdl)] += k_g

        T_ap = np.eye(ngdl, dtype=float)
        restringidos = []
        for nid, ap in self.apoyos.items():
            idx = mapa[nid] * 3
            th = ap['angulo']
            if abs(th) > 1e-9:
                ca, sa = np.cos(th), np.sin(th)
                T_ap[idx:idx+2, idx:idx+2] = np.array([[ca, sa], [-sa, ca]])
            for offset, bloqueado in enumerate(ap['rest']):
                if bloqueado:
                    restringidos.append(idx + offset)

        libres = [i for i in range(ngdl) if i not in restringidos]
        K_rot = T_ap @ K @ T_ap.T
        F_rot = T_ap @ F_sis
        U_rot = np.zeros(ngdl, dtype=float)
        if libres:
            U_rot[libres] = np.linalg.solve(K_rot[np.ix_(libres, libres)], F_rot[libres])

        self.U = T_ap.T @ U_rot
        self.R = K @ self.U + F_emp_tot - F_nodal

        for elem, k_loc, f_emp_l, T, L, w_comp, p_loc in datos_elem:
            i1, i2 = mapa[elem['ni']] * 3, mapa[elem['nj']] * 3
            gdl = [i1, i1+1, i1+2, i2, i2+1, i2+2]
            u_loc = T @ self.U[gdl]
            f_int = k_loc @ u_loc + f_emp_l
            self.resultados[elem['id']] = (f_int, L, w_comp, p_loc, elem['momentos'], u_loc, elem['E'], elem['I'])

        if imprimir:
            self._imprimir_reporte(nodos_ord, mapa)

    def _imprimir_reporte(self, nodos_ord, mapa):
        print("=" * 68)
        print(" VMR ESTRUCTURAL | Unidades: [" + self.u_f + "], [" + self.u_l + "]")
        print("=" * 68)
        print("\n1. DESPLAZAMIENTOS NODALES:")
        for nid in nodos_ord:
            idx = mapa[nid] * 3
            ux = ConversorUnidades.de_base_l(self.U.item(idx), self.u_l)
            uy = ConversorUnidades.de_base_l(self.U.item(idx+1), self.u_l)
            rz = self.U.item(idx+2)
            print(f"  Nodo {nid:2d}: Ux = {ux:10.6f} {self.u_l} | Uy = {uy:10.6f} {self.u_l} | Rz = {rz:10.6f} rad")

        print("\n2. REACCIONES EN LOS APOYOS (EJES GLOBALES X, Y):")
        for nid in sorted(self.apoyos.keys()):
            idx = mapa[nid] * 3
            rx = ConversorUnidades.de_base_f(self.R.item(idx), self.u_f)
            ry = ConversorUnidades.de_base_f(self.R.item(idx+1), self.u_f)
            mz = ConversorUnidades.de_base_m(self.R.item(idx+2), self.u_f, self.u_l)
            tipo = self.apoyos[nid]['tipo'].upper()
            print(f"  Nodo {nid:2d} ({tipo:9s}): Rx = {rx:9.3f} {self.u_f} | Ry = {ry:9.3f} {self.u_f} | Mz = {mz:9.3f} {self.u_f}*{self.u_l}")

        print("\n3. FUERZAS INTERNAS EN EXTREMOS (EJES LOCALES):")
        for elem in self.elementos:
            f_int = self.resultados[elem['id']][0]
            Ni = ConversorUnidades.de_base_f(f_int.item(0), self.u_f)
            Vi = ConversorUnidades.de_base_f(f_int.item(1), self.u_f)
            Mi = ConversorUnidades.de_base_m(f_int.item(2), self.u_f, self.u_l)
            Nj = ConversorUnidades.de_base_f(f_int.item(3), self.u_f)
            Vj = ConversorUnidades.de_base_f(f_int.item(4), self.u_f)
            Mj = ConversorUnidades.de_base_m(f_int.item(5), self.u_f, self.u_l)
            print(f"  Elemento {elem['id']} (N{elem['ni']} -> N{elem['nj']}):")
            print(f"    Inicio (i): N = {Ni:9.3f} {self.u_f} | V = {Vi:9.3f} {self.u_f} | M = {Mi:9.3f} {self.u_f}*{self.u_l}")
            print(f"    Final  (j): N = {Nj:9.3f} {self.u_f} | V = {Vj:9.3f} {self.u_f} | M = {Mj:9.3f} {self.u_f}*{self.u_l}")
        print("=" * 68)

    # --------------------------------------------------------------------------
    # INSPECTOR DE SECCIÓN EXACTA A CUALQUIER DISTANCIA x
    # --------------------------------------------------------------------------
    def consultar_seccion(self, id_elem, distancia, u_long=None, imprimir=True):
        """
        Calcula N(x), V(x), M(x), giro y deflexion transversal local a distancia x del inicio de id_elem.
        """
        ul = u_long or self.u_l
        x = ConversorUnidades.a_base_l(distancia, ul)
        if id_elem not in self.resultados:
            raise ValueError(f"Elemento {id_elem} no encontrado o estructura sin resolver.")

        f_int, L, (wx1, wx2, wy1, wy2), p_loc, moms, u_loc, E, I = self.resultados[id_elem]
        if x < -1e-6 or x > L + 1e-6:
            raise ValueError(f"La distancia debe estar entre 0 y {ConversorUnidades.de_base_l(L, ul):.3f} {ul}.")
        x = np.clip(x, 0.0, L)

        dwx, dwy = wx2 - wx1, wy2 - wy1
        Ni, Vi, Mi = -f_int.item(0), f_int.item(1), -f_int.item(2)

        N_val = Ni - (wx1 * x + (dwx / (2.0 * L)) * (x**2))
        V_val = Vi + wy1 * x + (dwy / (2.0 * L)) * (x**2)
        M_val = Mi + Vi * x + 0.5 * wy1 * (x**2) + (dwy / (6.0 * L)) * (x**3)

        # Integracion de giro y deflexion local usando funciones de Hermite + cargas
        xi = x / L
        v1, th1, v2, th2 = u_loc.item(1), u_loc.item(2), u_loc.item(4), u_loc.item(5)
        N1 = 1 - 3*(xi**2) + 2*(xi**3)
        N2 = L * (xi - 2*(xi**2) + (xi**3))
        N3 = 3*(xi**2) - 2*(xi**3)
        N4 = L * (-(xi**2) + (xi**3))
        defl = N1*v1 + N2*th1 + N3*v2 + N4*th2

        dN1 = (-6*xi + 6*(xi**2)) / L
        dN2 = 1 - 4*xi + 3*(xi**2)
        dN3 = (6*xi - 6*(xi**2)) / L
        dN4 = -2*xi + 3*(xi**2)
        giro = dN1*v1 + dN2*th1 + dN3*v2 + dN4*th2

        for a, Px, Py in p_loc:
            if x >= a:
                N_val -= Px
                V_val += Py
                M_val += Py * (x - a)

        for m in moms:
            if x >= m['a']:
                M_val -= m['M']

        res = {
            'x': ConversorUnidades.de_base_l(x, self.u_l),
            'N': ConversorUnidades.de_base_f(N_val, self.u_f),
            'V': ConversorUnidades.de_base_f(V_val, self.u_f),
            'M': ConversorUnidades.de_base_m(M_val, self.u_f, self.u_l),
            'giro_rad': giro,
            'deflexion': ConversorUnidades.de_base_l(defl, self.u_l)
        }

        if imprimir:
            print(f"\n>>> CONSULTA EXACTA EN ELEMENTO {id_elem} A x = {res['x']:.3f} {self.u_l}:")
            print(f"    Fuerza Axial N(x)    = {res['N']:10.3f} {self.u_f}")
            print(f"    Fuerza Cortante V(x) = {res['V']:10.3f} {self.u_f}")
            print(f"    Momento Flector M(x) = {res['M']:10.3f} {self.u_f}*{self.u_l}")
            print(f"    Deflexion local d(x) = {res['deflexion']:10.6f} {self.u_l}")
        return res

    def consultar_x_global(self, X_global, u_long=None, imprimir=True):
        """Busca automaticamente que elemento horizontal cruza la coordenada X_global y consulta esa seccion."""
        ul = u_long or self.u_l
        Xg = ConversorUnidades.a_base_l(X_global, ul)
        for elem in self.elementos:
            x1, _ = self.nodos[elem['ni']]
            x2, _ = self.nodos[elem['nj']]
            xmin, xmax = min(x1, x2), max(x1, x2)
            if xmin - 1e-6 <= Xg <= xmax + 1e-6 and abs(x2 - x1) > 1e-6:
                d_loc = abs(Xg - x1)
                return self.consultar_seccion(elem['id'], ConversorUnidades.de_base_l(d_loc, ul), u_long=ul, imprimir=imprimir)
        raise ValueError(f"No hay ningun elemento horizontal en X = {X_global} {ul}.")

    def graficar_diagramas(self, escala_N=None, escala_V=None, escala_M=None, n_puntos=120):
        curvas = {}
        max_N, max_V, max_M = 1e-9, 1e-9, 1e-9
        xs_coords = [ConversorUnidades.de_base_l(p[0], self.u_l) for p in self.nodos.values()]
        ys_coords = [ConversorUnidades.de_base_l(p[1], self.u_l) for p in self.nodos.values()]
        dim_modelo = max(max(xs_coords) - min(xs_coords), max(ys_coords) - min(ys_coords), 1.0)

        for elem in self.elementos:
            f_int, L, (wx1, wx2, wy1, wy2), p_loc, moms, _, _, _ = self.resultados[elem['id']]
            xs = np.linspace(0.0, L, n_puntos)
            dwx, dwy = wx2 - wx1, wy2 - wy1
            Ni, Vi, Mi = -f_int.item(0), f_int.item(1), -f_int.item(2)
            N_x = Ni - (wx1 * xs + (dwx / (2.0 * L)) * (xs**2))
            V_x = Vi + wy1 * xs + (dwy / (2.0 * L)) * (xs**2)
            M_x = Mi + Vi * xs + 0.5 * wy1 * (xs**2) + (dwy / (6.0 * L)) * (xs**3)
            for a, Px, Py in p_loc:
                mask = xs >= a
                N_x[mask] -= Px
                V_x[mask] += Py
                M_x[mask] += Py * (xs[mask] - a)
            for m in moms:
                mask = xs >= m['a']
                M_x[mask] -= m['M']
            N_out = ConversorUnidades.de_base_f(N_x, self.u_f)
            V_out = ConversorUnidades.de_base_f(V_x, self.u_f)
            M_out = ConversorUnidades.de_base_m(M_x, self.u_f, self.u_l)
            curvas[elem['id']] = (xs, N_out, V_out, M_out, L)
            max_N = max(max_N, float(np.max(np.abs(N_out))))
            max_V = max(max_V, float(np.max(np.abs(V_out))))
            max_M = max(max_M, float(np.max(np.abs(M_out))))

        esc_N = escala_N if escala_N is not None else (0.18 * dim_modelo / max_N)
        esc_V = escala_V if escala_V is not None else (0.18 * dim_modelo / max_V)
        esc_M = escala_M if escala_M is not None else (0.18 * dim_modelo / max_M)

        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
        titulos = [
            f"Fuerza Axial N [{self.u_f}]",
            f"Fuerza Cortante V [{self.u_f}]",
            f"Momento Flector M [{self.u_f}*{self.u_l}]"
        ]
        escalas = (esc_N, esc_V, esc_M)
        colores = ('#1f77b4', '#2ca02c', '#d62728')

        for idx_d, ax in enumerate(axes):
            ax.set_title(titulos[idx_d], fontweight='bold')
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_xlabel(f"X [{self.u_l}]")
            ax.set_ylabel(f"Y [{self.u_l}]")
            for nid, (x, y) in self.nodos.items():
                xo, yo = ConversorUnidades.de_base_l(x, self.u_l), ConversorUnidades.de_base_l(y, self.u_l)
                if nid in self.apoyos:
                    t = self.apoyos[nid]['tipo']
                    mk = 's' if t == 'empotrado' else ('^' if t == 'fijo' else 'o')
                    ax.plot(xo, yo, marker=mk, color='darkorange', markersize=8, zorder=6)
                else:
                    ax.plot(xo, yo, 'ko', markersize=4, zorder=5)

            for elem in self.elementos:
                x1, y1 = self.nodos[elem['ni']]
                x2, y2 = self.nodos[elem['nj']]
                xs, N_out, V_out, M_out, L = curvas[elem['id']]
                c, s = (x2 - x1) / L, (y2 - y1) / L
                nx, ny = -s, c
                vals, sig = (N_out, 1.0) if idx_d == 0 else ((V_out, 1.0) if idx_d == 1 else (M_out, -1.0))
                xb = ConversorUnidades.de_base_l(x1 + xs * c, self.u_l)
                yb = ConversorUnidades.de_base_l(y1 + xs * s, self.u_l)
                xd = xb + sig * vals * escalas[idx_d] * nx
                yd = yb + sig * vals * escalas[idx_d] * ny
                x1_o, y1_o = ConversorUnidades.de_base_l(x1, self.u_l), ConversorUnidades.de_base_l(y1, self.u_l)
                x2_o, y2_o = ConversorUnidades.de_base_l(x2, self.u_l), ConversorUnidades.de_base_l(y2, self.u_l)
                ax.plot([x1_o, x2_o], [y1_o, y2_o], 'k-', linewidth=2, zorder=4)
                if elem['art_i']:
                    ax.plot(x1_o + 0.06 * (x2_o - x1_o), y1_o + 0.06 * (y2_o - y1_o), 'o', mfc='white', mec='black', markersize=6, zorder=7)
                if elem['art_j']:
                    ax.plot(x2_o - 0.06 * (x2_o - x1_o), y2_o - 0.06 * (y2_o - y1_o), 'o', mfc='white', mec='black', markersize=6, zorder=7)
                ax.fill(np.concatenate([xb, xd[::-1]]), np.concatenate([yb, yd[::-1]]), color=colores[idx_d], alpha=0.25)
                ax.plot(xd, yd, color=colores[idx_d], linewidth=1.5)
                idx_max = int(np.argmax(np.abs(vals)))
                for p_i in {0, len(vals) - 1, idx_max}:
                    v_p = vals.item(p_i)
                    if abs(v_p) > 1e-3:
                        ax.text(xd.item(p_i), yd.item(p_i), f"{v_p:.2f}", fontsize=8, color=colores[idx_d], fontweight='bold')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Prueba en Unidades Inglesas (kip, ft) con consulta exacta a X = 15 ft
    est = Estructura2D(u_fuerza='kip', u_long='ft')
    est.agregar_nodo(1, 0, 0)
    est.agregar_nodo(2, 20, 0)
    est.agregar_apoyo(1, tipo='fijo')
    est.agregar_apoyo(2, tipo='movil')
    est.agregar_elemento(1, 1, 2, w1=-2.5)
    est.agregar_carga_puntual_barra(1, magnitud=-15, distancia=12)

    est.resolver()
    est.consultar_x_global(15.0)
    est.graficar_diagramas()
