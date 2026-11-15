#!/usr/bin/env python3
"""Gömböc radial de cinco estratos, sin ventana ni geometría visual.

El estrato inferior tiene densidad 2700 kg/m³ y los otros cuatro 135 kg/m³.
Los cortes internos reproducen pruebafinal8; los estratos extremos incluyen
los extremos del sólido que quedan fuera de la superficie muestreada.
La geometría de colisión sigue siendo la superficie radial analítica.

La superficie de colisión es r/a=(1+4 eps sin(theta) cos(phi-5 theta))**1/4.
Cada columna del terreno intersecta esta superficie mediante minimización y
bisección; no se usan triángulos para contacto. Las integrales de masa y el
contacto tienen precisión numérica finita.

Modelo exploratorio, NO SCM nativo: ley de carga de Bekker, descarga elástica
local y plasticidad irreversible sin flujo lateral. La anchura b de Bekker es
el diámetro nominal, no el ancho de una celda. Fricción de Coulomb cinética
regularizada en el plano tangente del cuerpo, sin memoria de corte ni
adherencia estática; NO es Janosi.
Base configurable, por defecto a 0.02 m bajo cada altura inicial: penalización elástica independiente,
sin restricción exacta ni garantía de los límites de hundimiento.
El terreno inicial es un heightmap sintético de amplitud configurable, no
una medición de Marte; cada columna se deforma bajo el cuerpo. No hay flujo
lateral ni contacto con caras verticales de las celdas. La presión vertical
se integra sobre área horizontal y se convierte a fuerza
normal a la superficie del cuerpo; su trabajo corresponde a la velocidad
vertical de intersección de cada columna. La rugosidad se representa mediante
alturas independientes por celda, no mediante normales interpoladas del suelo.

Para una columna de compresión d respecto a la superficie inicial y plástico
acumulado p, se usa presión elástica K_e max(d-p,0). Al cargar, el límite es
q=max(q_anterior,(k_c/b+k_phi) min(d,H)^n), con p nuevo=max(p,d-q/K_e),
limitado a la profundidad H de la base. k_c tiene unidades N/m^(n+1), k_phi N/m^(n+2);
q es Pa, K_e es Pa/m y el área de cuadratura es h². La base aporta
K_base max(d-H,0) Pa; esto permite penetrarla. La fuerza tangencial
regularizada se opone a la velocidad tangencial: su potencia es no positiva.
El amortiguamiento normal opcional actúa solo durante la compresión, con
coeficiente en Pa·s/m; por defecto es cero y no está calibrado.
El factor de resistencia opcional multiplica k_c y k_phi por igual; permite
estudiar una hipótesis de suelo más resistente, no representa una calibración.
El diámetro como anchura Bekker es una hipótesis simplificada, no una
medida de la huella ni necesariamente conservadora en todas las condiciones.
Para cuerpos grandes debe aumentarse explícitamente --rejilla para que el
heightmap que ocupa 4 diámetros no requiera memoria desmesurada. Eso cambia la
resolución espacial del terreno, no la geometría analítica ni su búsqueda.
Se contabilizan energía elástica, trabajo plástico aproximado q*delta_p y
trabajo de fricción. El balance es aproximado y no certifica estabilidad.
Se mide la profundidad del cuerpo respecto a la altura inicial local en los
centros de columnas; el soporte respecto a y=0 es un monitor independiente.
La cota del soporte supone que la búsqueda multisemilla encontró el mínimo
global. Estas mediciones no garantizan los máximos entre pasos ni entre
columnas; una advertencia no detiene ni corrige la simulación.
Con --limite-estricto se interrumpe con error si una medición o la cota
condicional supera 20 mm. Esto no es una restricción de contacto ni garantiza
que no exista penetración entre pasos o fuera de los puntos de cuadratura;
con relieve no se ofrece esta opción porque la cota usa el plano y=0.

La rigidez de descarga de 4e8 Pa/m procede de las pruebas anteriores y no
está calibrada para Marte. Una rigidez menor produce mayor recuperación
elástica a igual presión; no equivale a hacer el suelo más disipativo.
La malla de columnas y las búsquedas de raíces son aproximaciones; la base,
la compresión elástica, la regularización tangencial y el paso temporal deben
someterse a convergencia. No hay transporte lateral, cohesión, adherencia
estática, historial vectorial de corte ni calibración marciana.
No se modelan el flujo lateral, la cohesión completa ni la dependencia de la
velocidad de carga. El teorema del Gömböc sobre un plano no asegura equilibrio
único en una huella deformada. Este archivo no construye una malla visual.
La energía residual de orientación y la energía de maniobra se calculan
comparando alturas del COM sobre un plano horizontal hipotético; no incluyen
trabajo del terreno ni demuestran que esa energía sea recuperable.
Los umbrales de 0.877, 0.912 y 0.915 grados son estimaciones estáticas del
informe de auditoría para 10, 20 y 30 cm y su suelo de referencia; no se
recalculan para otros parámetros. La auditoría compara la resistencia de surco
Bekker b*(kc/b+kphi)*z**(n+1)/(n+1) con la fuerza motriz sobre plano.
No son constantes universales ni resultados de esta simulación.
Fuentes: https://api.projectchrono.org/classchrono_1_1vehicle_1_1_s_c_m_terrain.html
https://idosi.org/aerj/5(3)15/2.pdf
Los parámetros del suelo proceden de pruebafinal3.py, no están calibrados.
"""

import argparse
import csv
import ctypes
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation
# La búsqueda nativa opcional requiere libcontacto_radial9.so;
# GOMBOC_LIB_CONTACTO permite indicar su ubicación.


# Las condiciones comparacion30 se leen del CSV original.
# GOMBOC_CSV_PUNTOS indica su ruta; sin ella se busca junto a este archivo.
# GOMBOC_MODO_PUNTOS=fase1 genera 15 puntos sin necesidad del CSV.
_MODO_PUNTOS = os.environ.get("GOMBOC_MODO_PUNTOS", "comparacion30")
if _MODO_PUNTOS not in ("comparacion30", "fase1"):
    raise ValueError(f"GOMBOC_MODO_PUNTOS no reconocido: {_MODO_PUNTOS}")
N_PUNTOS = 30 if _MODO_PUNTOS == "comparacion30" else 15
CSV_PUNTOS_ORIGINALES = Path(os.environ.get(
    "GOMBOC_CSV_PUNTOS",
    Path(__file__).resolve().parent / "condiciones_comparacion_30_originales.csv",
))


def direccion_esferica(theta, phi):
    return np.array([
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    ])


def generar_puntos_fase1(coeficiente, scale, cog):
    """Crea las muestras Fibonacci y sus coordenadas sobre la ecuación radial."""
    count = 15
    theta_primero = math.acos(1.0 - 1.0 / count)
    primero = direccion_esferica(theta_primero, 0.0)
    objetivo = np.array([0.0, -math.cos(math.radians(15.0)),
                         math.sin(math.radians(15.0))])
    giro = Rotation.align_vectors(objetivo[None, :], primero[None, :])[0]
    puntos = []
    for i in range(count):
        theta = math.acos(1.0 - 2.0 * (i + 0.5) / count)
        phi_deg = (i * 137.508) % 360.0
        phi = math.radians(phi_deg)
        normal = giro.apply(direccion_esferica(theta, phi))
        theta = math.acos(np.clip(normal[2], -1.0, 1.0))
        phi = math.atan2(normal[1], normal[0])
        phi_deg = math.degrees(phi) % 360.0
        radio = scale * (
            1.0 + coeficiente * math.sin(theta)
            * math.cos(phi - 5.0 * theta)
        ) ** 0.25
        superficie = radio * normal
        puntos.append({
            "punto": i + 1,
            "theta_deg": math.degrees(theta),
            "phi_deg": phi_deg,
            "nx": normal[0],
            "ny": normal[1],
            "nz": normal[2],
            "r": radio,
            "x": superficie[0],
            "y": superficie[1],
            "z": superficie[2],
            "local_superficie": superficie - cog,
            "local_visual": superficie * 1.02 - cog,
        })
    return puntos


def generar_puntos(coeficiente, scale, cog):
    """Proporciona las direcciones originales de la fase seleccionada."""
    if _MODO_PUNTOS == "fase1":
        return generar_puntos_fase1(coeficiente, scale, cog)
    with CSV_PUNTOS_ORIGINALES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != N_PUNTOS:
        raise ValueError("El CSV original debe contener exactamente 30 puntos")
    puntos = []
    for index, row in enumerate(rows, 1):
        if (row["condition_id"] != f"P{index:02d}"
                or int(row["marker_index"]) != index):
            raise ValueError(f"Identificador u orden incorrecto para P{index:02d}")
        theta_deg, phi_deg = float(row["theta_deg"]), float(row["phi_deg"])
        normal = np.array([float(row[k]) for k in ("radial_x", "radial_y", "radial_z")])
        expected = direccion_esferica(math.radians(theta_deg), math.radians(phi_deg))
        if not np.allclose(normal, expected, rtol=0., atol=1e-12):
            raise ValueError(f"Dirección incompatible con los ángulos de P{index:02d}")
        radio = scale * (
            1.0 + coeficiente * math.sin(math.radians(theta_deg))
            * math.cos(math.radians(phi_deg) - 5.0 * math.radians(theta_deg))
        ) ** 0.25
        superficie = radio * normal
        puntos.append({
            "punto": index, "theta_deg": theta_deg, "phi_deg": phi_deg,
            "nx": normal[0], "ny": normal[1], "nz": normal[2],
            "r": radio, "x": superficie[0], "y": superficie[1], "z": superficie[2],
            "local_superficie": superficie - cog,
            "local_visual": superficie * 1.02 - cog,
        })
    return puntos


class NativeContactSearch:
    def __init__(self):
        path = Path(os.environ.get(
            "GOMBOC_LIB_CONTACTO",
            Path(__file__).with_name("libcontacto_radial9.so"),
        ))
        if not path.is_file():
            raise RuntimeError(
                f"No existe {path}. Compílalo con: "
                "gcc -O3 -shared -fPIC contacto_radial9.c -o libcontacto_radial9.so -lm. "
                "Indica su ruta con GOMBOC_LIB_CONTACTO.")
        self.library = ctypes.CDLL(str(path))
        doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        booleans = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        self.library.radial_minima.argtypes = [
            ctypes.c_size_t, doubles, doubles, doubles, ctypes.c_double,
            ctypes.c_double, ctypes.c_int, doubles, booleans,
        ]
        self.library.radial_minima.restype = None
        self.library.radial_lower_roots.argtypes = [
            ctypes.c_size_t, doubles, doubles, doubles, doubles,
            ctypes.c_double, ctypes.c_double, ctypes.c_int, doubles,
        ]
        self.library.radial_lower_roots.restype = None

    def minima(self, horizontal, vertical, half, scale, epsilon, iterations):
        horizontal = np.ascontiguousarray(horizontal, dtype=np.float64)
        vertical = np.ascontiguousarray(vertical, dtype=np.float64)
        half = np.ascontiguousarray(half, dtype=np.float64)
        mid = np.empty(len(half), dtype=np.float64)
        valid = np.empty(len(half), dtype=np.uint8)
        self.library.radial_minima(len(half), horizontal, vertical, half,
                                   scale, epsilon, iterations, mid, valid)
        return mid, valid.astype(bool)

    def lower_roots(self, horizontal, vertical, half, mid, scale, epsilon, iterations):
        horizontal = np.ascontiguousarray(horizontal, dtype=np.float64)
        vertical = np.ascontiguousarray(vertical, dtype=np.float64)
        half = np.ascontiguousarray(half, dtype=np.float64)
        mid = np.ascontiguousarray(mid, dtype=np.float64)
        roots = np.empty(len(half), dtype=np.float64)
        self.library.radial_lower_roots(len(half), horizontal, vertical, half,
                                        mid, scale, epsilon, iterations, roots)
        return roots

G = 3.71  # m/s²; gravedad marciana solicitada
EPS = 0.03
RHO = 2700.0
LAYER = 0.02  # profundidad predeterminada de la base física [m]
DEPTH_LIMIT = 0.02  # criterio de informe, independiente de la base [m]
SUPPORT_DIRECTION_TOL = 0.005  # diferencia máxima de direcciones para refrescar
GEOMETRIC_EQUILIBRIUM_TOL_DEG = 0.1  # diagnóstico estricto, no modifica la dinámica
KC, KPHI, EXPONENT = 5.721e5, 4.9153e6, 1.3
ELASTIC = 4.0e8  # Pa/m
BASE_K = 4.0e8  # Pa/m; base elástica penalizada, no contacto rígido exacto
PLANE_K = 2.0e5  # N/m; rigidez puntual de prueba13, sin amortiguador
MU = 0.6
SLIP_SPEED = 1.0e-3  # m/s, regularización de Coulomb; no fricción estática


def radial(theta, phi):
    return (1 + 4 * EPS * np.sin(theta) * np.cos(phi - 5 * theta)) ** 0.25


def success_checks(tilt_deg, linear_speed_m_s, angular_speed_rad_s):
    return (
        bool(tilt_deg < 7.5),
        bool(linear_speed_m_s < 1.0),
        bool(angular_speed_rad_s < 1.0),
    )


def should_stop_after_transient_pass(
        tilt_deg, angular_speed_rad_s, previous_angular_speed_rad_s,
        simulated_time_s):
    return (
        simulated_time_s >= 2.0
        and previous_angular_speed_rad_s is not None
        and tilt_deg < 7.5
        and angular_speed_rad_s < previous_angular_speed_rad_s
    )


def signed_tilt(up, reference):
    """Signed projection onto the initial vertical plane, plus full 3D tilt."""
    along = float(up @ reference)
    return (math.degrees(math.atan2(along, up[1])),
            math.degrees(math.acos(np.clip(up[1], -1., 1.))))


def crossed_vertical_near_equilibrium(previous, current, reference):
    """Detect a plane crossing only if its interpolated direction is near +Y."""
    a, b = float(previous @ reference), float(current @ reference)
    if not ((a > 0. and b <= 0.) or (a < 0. and b >= 0.)):
        return False
    crossing = previous + (current - previous) * (a / (a - b))
    norm = np.linalg.norm(crossing)
    return bool(norm > 0. and crossing[1] / norm > math.cos(math.radians(7.5)))


def directions(theta, phi):
    return np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                     np.cos(theta)), axis=-1)


def surface(theta, phi, scale):
    return scale * radial(theta, phi)[..., None] * directions(theta, phi)


def implicit_gradient(local, scale):
    """Gradiente local de |x|-r(x/|x|), no una normal de malla."""
    length = np.linalg.norm(local, axis=1)
    if np.any(length < 1e-14 * scale):
        raise RuntimeError("Gradiente radial indefinido en el origen")
    unit = local / length[:, None]
    theta = np.arccos(np.clip(unit[:, 2], -1, 1))
    phi = np.arctan2(unit[:, 1], unit[:, 0])
    st, ct = np.sin(theta), np.cos(theta)
    delta = phi - 5 * theta
    factor = 1 + 4 * EPS * st * np.cos(delta)
    rtheta = scale * EPS * factor**(-.75) * (
        ct * np.cos(delta) + 5 * st * np.sin(delta))
    # r_phi/sin(theta) tiene límite finito fuera de la singularidad polar.
    rphi_over_sin = -scale * EPS * factor**(-.75) * np.sin(delta)
    etheta = np.column_stack((ct*np.cos(phi), ct*np.sin(phi), -st))
    ephi = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros(len(phi))))
    return unit - (rtheta[:, None]*etheta + rphi_over_sin[:, None]*ephi) / length[:, None]


def geometry(diameter, ntheta, nphi, density=RHO):
    """Cuadratura de Gauss en cos(theta) y trapecio periódico en phi."""
    u, weights = np.polynomial.legendre.leggauss(ntheta)
    theta, phi = np.meshgrid(np.arccos(u), 2 * np.pi * np.arange(nphi) / nphi,
                             indexing="ij")
    unit = directions(theta, phi)
    radius = radial(theta, phi)
    # La escala coincide con el mayor diámetro de los ejes coordenados,
    # como en prueba13, pero refinando numéricamente sus extremos.
    points = (radius[..., None] * unit).reshape(-1, 3)
    widths = []
    for axis in range(3):
        extrema = []
        for sign in (-1, 1):
            guess = points[np.argmax(sign * points[:, axis])]
            x0 = [math.acos(np.clip(guess[2] / np.linalg.norm(guess), -1, 1)),
                  math.atan2(guess[1], guess[0])]
            result = minimize(lambda x: -sign * surface(x[0], x[1], 1)[axis],
                              x0, method="Nelder-Mead",
                              bounds=[(0., math.pi), (None, None)],
                              options={"xatol": 1e-11, "fatol": 1e-13})
            if not result.success:
                raise RuntimeError("No convergió la búsqueda de dimensiones")
            extrema.append(surface(*result.x, 1)[axis])
        widths.append(extrema[1] - extrema[0])
    scale = diameter / max(widths)
    r = scale * radius
    w = weights[:, None] * (2 * np.pi / nphi)
    volume = float(np.sum(w * r**3 / 3))
    first = np.einsum("ij,ijk->k", w * r**4 / 4, unit)
    cog = first / volume
    second = np.einsum("ij,ijk,ijl->kl", w * r**5 / 5, unit, unit)
    central = second - volume * np.outer(cog, cog)
    inertia = density * (np.trace(central) * np.eye(3) - central)
    return scale, np.array(widths) * scale, volume, cog, inertia


def support(direction, scale, cog):
    """Mínimo de dirección·superficie; semillas múltiples, no prueba global."""
    direction = np.asarray(direction)
    def objective(x):
        return float(direction @ (surface(x[0], x[1], scale) - cog))
    start = -direction / np.linalg.norm(direction)
    theta = math.acos(np.clip(start[2], -1, 1))
    phi = math.atan2(start[1], start[0])
    results = [minimize(objective, [theta, phi + shift], method="Nelder-Mead",
                        bounds=[(0., math.pi), (None, None)],
                        options={"xatol": 1e-10, "fatol": 1e-13})
               for shift in (0, math.pi / 2, -math.pi / 2)]
    successful = [result for result in results if result.success]
    if not successful:
        raise RuntimeError("No convergió la búsqueda de soporte radial")
    best = min(successful, key=lambda x: x.fun)
    candidate, _ = support_warm(direction, scale, cog, best.x, multistart=True)
    if direction @ candidate < best.fun:
        return candidate
    return surface(*best.x, scale) - cog


def support_warm(direction, scale, cog, start, multistart=False):
    """Soporte con gradiente analítico; multistart reduce mínimos locales."""
    def value_gradient(angles):
        th, ph = angles
        st, ct = math.sin(th), math.cos(th)
        delta = ph - 5 * th
        factor = 1 + 4 * EPS * st * math.cos(delta)
        radius = factor**.25
        rt = EPS * factor**(-.75) * (ct*math.cos(delta) + 5*st*math.sin(delta))
        rp = -EPS * factor**(-.75) * st*math.sin(delta)
        u = np.array([st*math.cos(ph), st*math.sin(ph), ct])
        ut = np.array([ct*math.cos(ph), ct*math.sin(ph), -st])
        up = np.array([-st*math.sin(ph), st*math.cos(ph), 0.])
        point = scale*radius*u-cog
        grad = scale * np.array([direction @ (rt*u + radius*ut),
                                 direction @ (rp*u + radius*up)])
        return float(direction @ point), grad
    starts = [start]
    if multistart:
        down = -direction / np.linalg.norm(direction)
        theta = math.acos(np.clip(down[2], -1, 1))
        phi = math.atan2(down[1], down[0])
        starts.extend([[theta, phi + shift] for shift in (0, math.pi / 2, -math.pi / 2)])
    results = [minimize(value_gradient, seed, jac=True, method="L-BFGS-B",
                        bounds=[(0., math.pi), (None, None)],
                        options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 100})
               for seed in starts]
    for candidate in results:
        if not candidate.success and np.linalg.norm(candidate.jac) > 1e-7*scale:
            raise RuntimeError("No convergió la búsqueda de soporte: " + candidate.message)
    result = min(results, key=lambda candidate: candidate.fun)
    return surface(*result.x, scale) - cog, result.x


def stable_points(scale, cog):
    """Extremos de distancia al COM para apoyos en plano horizontal.

    La búsqueda numérica global y su refinamiento no demuestran unicidad.
    No se supone que los extremos sean antipodales.
    """
    def objective(x, sign):
        point = surface(x[0], x[1], scale) - cog
        return sign * float(point @ point) / scale**2

    points = []
    for sign in (1, -1):
        sol = differential_evolution(
            lambda x: objective(x, sign), [(0, math.pi), (-math.pi, math.pi)],
            seed=7, tol=1e-11, polish=False)
        if not sol.success:
            raise RuntimeError("No convergió la búsqueda global del equilibrio")
        refined = minimize(
            objective, sol.x, args=(sign,), method="Nelder-Mead",
            bounds=[(0., math.pi), (None, None)],
            options={"xatol": 1e-12, "fatol": 1e-15, "maxiter": 500})
        if not refined.success:
            raise RuntimeError("No convergió el refinamiento del equilibrio")
        point = surface(*refined.x, scale) - cog
        gradient = implicit_gradient((point + cog)[None, :], scale)[0]
        residual = np.linalg.norm(np.cross(point, gradient)) / (
            np.linalg.norm(point) * np.linalg.norm(gradient))
        if residual > 1e-6:
            raise RuntimeError("El equilibrio calculado no satisface la condición normal")
        points.append(point)
    red, blue = points
    return red / np.linalg.norm(red), red, blue


def sampled_base_direction(vertices):
    """Reproduce la búsqueda direccional de la base de pruebafinal4."""
    count = 12000
    index = np.arange(count)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * math.pi * index / ((1.0 + math.sqrt(5.0)) / 2.0)
    unit = np.column_stack((radius * np.cos(phi), z, radius * np.sin(phi)))
    height = np.empty(count)
    block_size = max(1, int(4e6 / len(vertices)))
    for start in range(0, count, block_size):
        block = unit[start:start + block_size]
        height[start:start + len(block)] = (vertices @ block.T).max(axis=0)
    return unit[int(np.argmin(height))]


def layered_properties(scale, volume, ntheta, nphi):
    """Integra los cinco estratos históricos sin normalizar las densidades."""
    mu, weights = np.polynomial.legendre.leggauss(ntheta)
    theta, phi = np.meshgrid(np.arccos(mu), 2 * np.pi * np.arange(nphi) / nphi,
                             indexing="ij")
    unit = directions(theta, phi).reshape(-1, 3)
    radius = (scale * radial(theta, phi)).ravel()
    angular_weights = np.broadcast_to(
        weights[:, None] * (2 * np.pi / nphi), theta.shape).ravel()
    th, ph = np.meshgrid(np.linspace(0., math.pi, 122)[1:-1],
                         2 * math.pi * np.arange(240) / 240, indexing="ij")
    vertices = np.vstack(([0., 0., scale], surface(th, ph, scale).reshape(-1, 3),
                          [0., 0., -scale]))
    low = sampled_base_direction(vertices - np.average(vertices, axis=0))
    densities = np.array([2700., 135., 135., 135., 135.])
    for _ in range(12):
        axis_up = -low
        projections = vertices @ axis_up
        edges = np.linspace(float(projections.min()), float(projections.max()), 6)
        axial = unit @ axis_up
        volumes = np.zeros(5)
        first = np.zeros((5, 3))
        second = np.zeros((5, 3, 3))
        for layer in range(5):
            bottom, top = edges[layer:layer + 2]
            if layer == 0:
                bottom = -np.inf
            if layer == 4:
                top = np.inf
            r0 = np.zeros_like(radius)
            r1 = radius.copy()
            positive = axial > 1e-12
            negative = axial < -1e-12
            r0[positive] = np.maximum(0., bottom / axial[positive])
            r1[positive] = np.minimum(radius[positive], top / axial[positive])
            r0[negative] = np.maximum(0., top / axial[negative])
            r1[negative] = np.minimum(radius[negative], bottom / axial[negative])
            parallel = ~(positive | negative)
            if not bottom <= 0. <= top:
                r1[parallel] = 0.
            r0 = np.minimum(r0, radius)
            r1 = np.maximum(r1, r0)
            valid = r1 > r0
            w, d, lo, hi = (angular_weights[valid], unit[valid],
                            r0[valid], r1[valid])
            volumes[layer] = np.sum(w * (hi**3 - lo**3) / 3)
            first[layer] = np.einsum("i,ij->j", w * (hi**4 - lo**4) / 4, d)
            second[layer] = np.einsum(
                "i,ij,ik->jk", w * (hi**5 - lo**5) / 5, d, d)
        if not np.isclose(volumes.sum(), volume, rtol=1e-12, atol=1e-15):
            raise RuntimeError("Los estratos no cubren la cuadratura del volumen")
        mass = float(densities @ volumes)
        cog = np.einsum("i,ij->j", densities, first) / mass
        moment = np.einsum("i,ijk->jk", densities, second)
        central = moment - mass * np.outer(cog, cog)
        inertia = np.trace(central) * np.eye(3) - central
        new_low = sampled_base_direction(vertices - cog)
        if np.linalg.norm(new_low - low) < 1e-7:
            return mass, cog, inertia, densities, volumes, axis_up, edges
        low = new_low
    raise RuntimeError("No convergió el eje de la densidad estratificada")


def initial_rotation(low, degrees):
    down = np.array([0., -1., 0.])
    cross = np.cross(low, down)
    norm = np.linalg.norm(cross)
    angle = math.acos(np.clip(low @ down, -1, 1))
    if norm < 1e-12:
        align = Rotation.from_rotvec([angle, 0, 0])
    else:
        align = Rotation.from_rotvec(cross * angle / norm)
    return Rotation.from_rotvec([math.radians(degrees), 0, 0]) * align


def rotation_from_marker(point_id, scale, cog):
    """Reproduce la orientación de PUNTO_ABAJO para un punto Fibonacci."""
    if not 1 <= point_id <= N_PUNTOS:
        raise ValueError(f"El punto debe estar entre 1 y {N_PUNTOS}")
    points = generar_puntos(4 * EPS, scale, cog)
    radial_direction = np.array([
        points[point_id - 1]["nx"],
        points[point_id - 1]["ny"],
        points[point_id - 1]["nz"],
    ])
    return Rotation.align_vectors(
        np.array([[0., -1., 0.]]), radial_direction.reshape(1, 3))[0]


def load_chrono():
    chrono_bin = next((Path(p) for p in sys.path if p
                       and (Path(p) / "pychrono" / "__init__.py").is_file()), None)
    original_directory = os.getcwd()
    try:
        if chrono_bin and (chrono_bin.parent / "data" / "colormaps" /
                           "jet-table-float-0512.csv").is_file():
            os.chdir(chrono_bin)
        import pychrono as chrono
    finally:
        os.chdir(original_directory)
    data = Path(os.environ.get("CHRONO_DATA_DIR",
                               Path(chrono.__file__).resolve().parents[2] / "data"))
    if (data / "colormaps" / "jet-table-float-0512.csv").is_file():
        chrono.SetChronoDataPath(str(data) + os.sep)
    return chrono


def axes(body):
    q = body.GetRot()
    return np.array([[q.GetAxisX().x, q.GetAxisY().x, q.GetAxisZ().x],
                     [q.GetAxisX().y, q.GetAxisY().y, q.GetAxisZ().y],
                     [q.GetAxisX().z, q.GetAxisY().z, q.GetAxisZ().z]])


def vector(v):
    return np.array([v.x, v.y, v.z])


def contact_forces(gradient, vertical_force, velocity, mu, slip_speed):
    """Convierte carga vertical de columnas a fuerza normal y cortante."""
    norm = np.linalg.norm(gradient, axis=1)
    if np.any(norm <= 0.) or np.any(gradient[:, 1] >= -1e-10):
        raise RuntimeError("Normal inferior singular: refina la rejilla o la búsqueda radial")
    normal_unit = -gradient / norm[:, None]
    normal_magnitude = vertical_force / normal_unit[:, 1]
    normal_force = normal_magnitude[:, None] * normal_unit
    tangent_velocity = velocity - np.sum(
        velocity * normal_unit, axis=1)[:, None] * normal_unit
    tangent_speed = np.linalg.norm(tangent_velocity, axis=1)
    shear = -mu * normal_magnitude[:, None] * tangent_velocity / np.sqrt(
        tangent_speed**2 + slip_speed**2)[:, None]
    return normal_force, shear, tangent_velocity


class Ground:
    def __init__(self, diameter, spacing, iterations, rigid, mu=MU, slip_speed=SLIP_SPEED,
                 relief=0., wavelength=.12, normal_damping=0., resistance_factor=1.,
                 elastic=ELASTIC, base_depth=LAYER, relief_phase=0.,
                 kc=KC, kphi=KPHI, native_search=None):
        self.h = spacing
        self.native_search = native_search
        self.iterations = iterations
        self.rigid = rigid
        self.mu = mu
        self.slip_speed = slip_speed
        self.normal_damping = normal_damping
        self.elastic = elastic
        self.base_depth = base_depth
        self.extent = max(1., 4 * diameter)
        n = math.ceil(self.extent / spacing)
        self.axis = (np.arange(n) - (n - 1) / 2) * spacing
        x, z = np.meshgrid(self.axis, self.axis)
        self.initial_height = relief * np.sin(2 * np.pi * x / wavelength + relief_phase) * np.cos(
            2 * np.pi * z / wavelength)
        self.plastic = np.zeros((n, n))
        self.yield_pressure = np.zeros((n, n))
        self.kc = kc
        self.kphi = kphi
        self.factor = resistance_factor * (kc / diameter + kphi)
        self.plastic_work = 0.
        self.friction_work = 0.
        self.damping_work = 0.
        self.max_local_depth = 0.
        self.local_depth = 0.
        self.max_elastic_compression = 0.
        self.base_reached = np.zeros((n, n), dtype=bool)
        self.max_body = 0.
        self.max_body_bound = 0.
        self.warned = False
        self.warned_body = False
        self.warned_body_bound = False
        self.monitor_direction = np.zeros(3)
        self.monitor_support = 0.
        self.monitor_count = 0
        self.contacts = 0
        self.geometric_contacts = 0
        self.base_contacts = 0
        self.elastic_energy = 0.
        self.maximum_force = 0.
        self.maximum_torque = 0.
        self.maximum_pressure = 0.
        self.plane_angles = None
        self.last_force = np.zeros(3)
        self.last_torque = np.zeros(3)

    def apply(self, body, chrono, acc, scale, cog, dt):
        rot = axes(body)
        pos = vector(body.GetPos())
        self.last_force.fill(0.)
        self.last_torque.fill(0.)
        if self.rigid:
            direction = rot[1, :]
            if self.plane_angles is None:
                first = support(direction, scale, cog) + cog
                first /= np.linalg.norm(first)
                self.plane_angles = [math.acos(np.clip(first[2], -1, 1)),
                                     math.atan2(first[1], first[0])]
            point, self.plane_angles = support_warm(
                direction, scale, cog, self.plane_angles, multistart=True)
            arm = rot @ point
            depth = max(0., -pos[1] - arm[1])
            self.max_body = max(self.max_body, depth)
            self.contacts = int(depth > 0)
            self.geometric_contacts = self.contacts
            self.base_contacts = 0
            self.elastic_energy = .5 * PLANE_K * depth**2
            if depth:
                velocity = vector(body.GetLinVel()) + np.cross(
                    vector(body.GetAngVelParent()), arm)
                tangential = velocity.copy()
                tangential[1] = 0.
                speed = np.linalg.norm(tangential)
                normal = PLANE_K * depth
                shear = -self.mu * normal * tangential / math.sqrt(
                    speed**2 + self.slip_speed**2)
                force = shear + np.array([0., normal, 0.])
                self.friction_work += float(-shear @ tangential * dt)
                self.last_force = force
                self.last_torque = np.cross(arm, force)
                self.maximum_torque = max(
                    self.maximum_torque, float(np.linalg.norm(self.last_torque)))
                body.AccumulateForce(acc, chrono.ChVector3d(*force),
                                     chrono.ChVector3d(*(pos + arm)), False)
                self.maximum_force = max(self.maximum_force, float(np.linalg.norm(force)))
            return
        origin = pos - rot @ cog
        bound = scale * (1 + 4 * abs(EPS)) ** 0.25
        if (abs(origin[0]) + bound >= abs(self.axis[-1])
                or abs(origin[2]) + bound >= abs(self.axis[-1])):
            raise RuntimeError("El cuerpo alcanzó el borde de la rejilla del terreno")
        ix = np.flatnonzero(abs(self.axis - origin[0]) < bound)
        iz = np.flatnonzero(abs(self.axis - origin[2]) < bound)
        zz, xx = np.meshgrid(iz, ix, indexing="ij")
        x, z = self.axis[xx.ravel()], self.axis[zz.ravel()]
        xx, zz = xx.ravel(), zz.ravel()
        dx, dz = x - origin[0], z - origin[2]
        valid = dx*dx + dz*dz < bound*bound
        x, z, xx, zz, dx, dz = (v[valid] for v in (x, z, xx, zz, dx, dz))
        half = np.sqrt(bound*bound - dx*dx - dz*dz)
        valid = origin[1] - half < self.initial_height[zz, xx]
        x, z, xx, zz, dx, dz, half = (v[valid] for v in
                                       (x, z, xx, zz, dx, dz, half))
        if not len(x):
            self.contacts = self.geometric_contacts = self.base_contacts = 0
            self.local_depth = 0.
            self.elastic_energy = 0.
            return

        # g(y) = |local| - r(direction); negativa dentro del cuerpo. La
        # búsqueda del mínimo sobre la cuerda detecta intersecciones laterales.
        horizontal = np.column_stack((dx, np.zeros(len(x)), dz)) @ rot
        vertical = rot[1, :]
        def signed(y):
            local = horizontal + y[:, None] * vertical
            length = np.linalg.norm(local, axis=1)
            safe = np.maximum(length, 1e-30)
            theta = np.arccos(np.clip(local[:, 2] / safe, -1, 1))
            phi = np.arctan2(local[:, 1], local[:, 0])
            return length - scale * radial(theta, phi)

        if self.native_search is None:
            a, b = -half.copy(), half.copy()
            ratio = (math.sqrt(5) - 1) / 2
            c, d = b - ratio * (b-a), a + ratio * (b-a)
            fc, fd = signed(c), signed(d)
            for _ in range(self.iterations):
                left = fc < fd
                b, a = np.where(left, d, b), np.where(left, a, c)
                c, d = b - ratio * (b-a), a + ratio * (b-a)
                fc, fd = signed(c), signed(d)
            mid = (a+b)/2
            valid = signed(mid) < 0
        else:
            mid, valid = self.native_search.minima(
                horizontal, vertical, half, scale, EPS, self.iterations)
        x, z, xx, zz, half, mid, horizontal = (v[valid] for v in
                                               (x, z, xx, zz, half, mid, horizontal))
        if not len(x):
            self.contacts = self.geometric_contacts = self.base_contacts = 0
            self.local_depth = 0.
            self.elastic_energy = 0.
            return
        if self.native_search is None:
            lower, upper = -half, mid
            for _ in range(self.iterations + 4):
                trial = (lower + upper) / 2
                inside = signed(trial) < 0
                lower = np.where(inside, lower, trial)
                upper = np.where(inside, trial, upper)
            roots = (lower + upper)/2
        else:
            roots = self.native_search.lower_roots(
                horizontal, vertical, half, mid, scale, EPS, self.iterations)
        y = origin[1] + roots
        depth = self.initial_height[zz, xx] - y
        valid = depth > 0
        x, z, xx, zz, y, depth = (v[valid] for v in (x, z, xx, zz, y, depth))
        self.geometric_contacts = len(x)
        self.local_depth = float(depth.max()) if len(depth) else 0.
        self.max_local_depth = max(self.max_local_depth, self.local_depth)
        self.base_reached[zz, xx] |= depth >= self.base_depth
        self.base_contacts = int(np.count_nonzero(depth >= self.base_depth))
        if self.max_local_depth > DEPTH_LIMIT and not self.warned:
            print("AVISO: profundidad local del cuerpo > 20 mm; sin corrección de posición.",
                  flush=True)
            self.warned = True
        if not len(x):
            self.contacts = 0
            self.elastic_energy = 0.
            return
        valid = depth > self.plastic[zz, xx]
        x, z, xx, zz, y, depth = (v[valid] for v in (x, z, xx, zz, y, depth))
        self.contacts = len(x)
        if not len(x):
            self.elastic_energy = 0.
            return
        area = self.h**2
        old = self.plastic[zz, xx]
        yield_p = np.maximum(self.yield_pressure[zz, xx],
                             self.factor * np.minimum(depth, self.base_depth)**EXPONENT)
        new = np.minimum(self.base_depth, np.maximum(old, depth - yield_p / self.elastic))
        compression = np.maximum(depth - new, 0)
        pressure = self.elastic * compression
        self.plastic_work += float(np.sum(pressure * (new-old)) * area)
        self.plastic[zz, xx] = new
        self.yield_pressure[zz, xx] = np.maximum(yield_p, pressure)
        base_depth = np.maximum(depth-self.base_depth, 0)
        base_pressure = BASE_K * base_depth
        self.elastic_energy = float(np.sum(
            0.5 * self.elastic * compression**2
            + 0.5 * BASE_K * base_depth**2) * area)
        self.max_elastic_compression = max(self.max_elastic_compression, float(compression.max()))
        arms = np.column_stack((x, y, z)) - pos
        vel = vector(body.GetLinVel()) + np.cross(vector(body.GetAngVelParent()), arms)
        local = (np.column_stack((x, y, z)) - origin) @ rot
        gradient = implicit_gradient(local, scale) @ rot.T
        if np.any(gradient[:, 1] >= -1e-10):
            raise RuntimeError("Normal inferior singular: refina la rejilla o la búsqueda radial")
        compression_rate = np.maximum(
            -np.sum(gradient * vel, axis=1) / gradient[:, 1], 0.)
        damping_pressure = self.normal_damping * compression_rate
        self.maximum_pressure = max(
            self.maximum_pressure, float(np.max(pressure + base_pressure + damping_pressure)))
        vertical_force = (pressure + base_pressure + damping_pressure) * area
        self.damping_work += float(np.sum(damping_pressure * compression_rate) * area * dt)
        normal_force, shear, tangent_velocity = contact_forces(
            gradient, vertical_force, vel, self.mu, self.slip_speed)
        self.friction_work += float(-np.sum(shear * tangent_velocity) * dt)
        forces = normal_force + shear
        total, torque = forces.sum(axis=0), np.cross(arms, forces).sum(axis=0)
        self.last_force = total
        self.last_torque = torque
        self.maximum_torque = max(
            self.maximum_torque, float(np.linalg.norm(torque)))
        body.AccumulateForce(acc, chrono.ChVector3d(*total), body.GetPos(), False)
        body.AccumulateTorque(acc, chrono.ChVector3d(*torque), False)
        self.maximum_force = max(self.maximum_force, float(np.linalg.norm(total)))

    def measure_body_depth(self, body, scale, cog, angles, strict=False):
        """Estimación y cota temporal condicional del soporte, en cada paso."""
        rot = axes(body)
        direction = rot[1, :]
        check_branches = (self.monitor_count % 50 == 0
                          or np.linalg.norm(direction - self.monitor_direction)
                          > SUPPORT_DIRECTION_TOL)
        point, angles = support_warm(direction, scale, cog, angles,
                                     multistart=check_branches)
        height = float(direction @ point)
        if check_branches:
            self.monitor_direction = direction.copy()
            self.monitor_support = height
        self.monitor_count += 1
        radius_bound = scale * (1 + 4 * abs(EPS)) ** .25 + np.linalg.norm(cog)
        lower_height = self.monitor_support - radius_bound * np.linalg.norm(
            direction - self.monitor_direction)
        pos_y = vector(body.GetPos())[1]
        if strict and not check_branches and -pos_y - lower_height > DEPTH_LIMIT:
            point, angles = support_warm(direction, scale, cog, angles, multistart=True)
            height = float(direction @ point)
            self.monitor_direction = direction.copy()
            self.monitor_support = height
            lower_height = height
        depth = max(0., -pos_y - height)
        depth_bound = max(depth, -pos_y - lower_height, 0.)
        self.max_body = max(self.max_body, depth)
        self.max_body_bound = max(self.max_body_bound, depth_bound)
        if not self.rigid and depth > DEPTH_LIMIT and not self.warned_body:
            print("AVISO: hundimiento del cuerpo respecto a y=0 > 20 mm; "
                  "sin corrección de posición.", flush=True)
            self.warned_body = True
        elif not self.rigid and depth_bound > DEPTH_LIMIT and not self.warned_body_bound:
            print("AVISO: cota condicional de hundimiento del cuerpo > 20 mm; "
                  "la búsqueda numérica de soporte no garantiza el límite.", flush=True)
            self.warned_body_bound = True
        return depth, angles

    def check_depth_limit(self):
        if self.rigid:
            return
        if self.max_local_depth > DEPTH_LIMIT or self.plastic.max() > DEPTH_LIMIT:
            raise RuntimeError(
                "Límite estricto: profundidad local o plasticidad supera 20 mm "
                f"({self.max_local_depth * 1000:.4f}, {self.plastic.max() * 1000:.4f} mm).")
        if self.max_body_bound > DEPTH_LIMIT:
            raise RuntimeError(
                "Límite estricto: la cota condicional del hundimiento del cuerpo "
                f"supera 20 mm ({self.max_body_bound * 1000:.4f} mm); "
                "no se puede certificar el límite.")

def run(args, visual_factory=None):
    chrono = load_chrono()
    native_search = None
    if args.acelerar_contacto:
        native_search = NativeContactSearch()
    scale, widths, volume, geometric_cog, geometric_inertia = geometry(
        args.diametro, args.ntheta, args.nphi, args.densidad)
    if args.masa is not None or args.densidad != RHO:
        raise ValueError("El modelo estratificado requiere densidades 2700/135 kg/m³")
    mass, cog, inertia, layer_densities, layer_volumes, layer_axis, layer_edges = (
        layered_properties(scale, volume, args.ntheta, args.nphi))
    density = mass / volume
    if np.linalg.eigvalsh(inertia).min() <= 0:
        raise RuntimeError("Inercia no definida positiva")
    low, red, blue = stable_points(scale, cog)
    if float(layer_axis @ (red + cog)) > layer_edges[1]:
        raise RuntimeError("El equilibrio estable no pertenece al estrato inferior")
    if args.punto_abajo is None:
        requested_inclination = 15. if args.inclinacion is None else args.inclinacion
        rotation = initial_rotation(low, requested_inclination)
        initial_pose_source = "equilibrio estable inclinado alrededor de X mundial"
    else:
        requested_inclination = None
        rotation = rotation_from_marker(args.punto_abajo, scale, cog)
        initial_pose_source = "punto Fibonacci de pruebapuntosdatos.py alineado radialmente hacia abajo"
    matrix = rotation.as_matrix()
    initial_tilt = math.degrees(math.acos(np.clip(-(matrix @ low)[1], -1, 1)))
    point = support(matrix[1, :], scale, cog)
    horizontal_shift = matrix @ (cog - geometric_cog)
    pos = np.array([horizontal_shift[0], -float((matrix @ point)[1]),
                    horizontal_shift[2]])
    ground = Ground(args.diametro, args.rejilla, args.iteraciones, args.plano,
                    args.mu, args.velocidad_regularizacion,
                    args.relieve if not args.plano else 0., args.longitud_relieve,
                    args.amortiguamiento_normal, args.factor_resistencia,
                    args.rigidez_descarga, args.profundidad_base, args.fase_relieve,
                    args.kc, args.kphi, native_search)
    if not args.plano:
        # Cota superior sobre el footprint: se libera sin interpenetración.
        bound = scale * (1 + 4 * abs(EPS)) ** 0.25
        x, z = np.meshgrid(ground.axis, ground.axis)
        initial_origin = pos - matrix @ cog
        footprint = ((x - initial_origin[0])**2 + (z - initial_origin[2])**2
                     <= bound*bound)
        if not np.any(footprint):
            raise RuntimeError("La rejilla no contiene columnas bajo el cuerpo")
        pos[1] += float(ground.initial_height[footprint].max())
    if args.plano:
        # Igual que prueba13: compresión estática inicial del muelle puntual.
        pos[1] -= mass * G / PLANE_K
    initial_pos = pos.copy()
    stable_matrix = initial_rotation(low, 0).as_matrix()
    stable_angle = math.degrees(math.acos(np.clip(
        (stable_matrix @ blue)[1]/np.linalg.norm(blue), -1, 1)))
    quat = rotation.as_quat()
    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, -G, 0))
    system.SetSleepingAllowed(False)
    body = chrono.ChBody()
    body.SetMass(mass)
    body.SetInertiaXX(chrono.ChVector3d(*np.diag(inertia)))
    body.SetInertiaXY(chrono.ChVector3d(inertia[0, 1], inertia[0, 2], inertia[1, 2]))
    body.EnableCollision(False)
    body.SetRot(chrono.ChQuaterniond(quat[3], quat[0], quat[1], quat[2]))
    body.SetPos(chrono.ChVector3d(*pos))
    system.Add(body)
    acc = body.AddAccumulator()
    visual = (visual_factory(chrono, system, body, ground, scale, cog, blue, red,
                             args.diametro, args.dt)
              if visual_factory is not None else None)
    print("Geometría radial: epsilon=%.3f; diámetro nominal=%.6g m; dimensiones=%s m"
          % (EPS, args.diametro, widths))
    print("Densidad media=%.9g kg/m³; volumen=%.9g m³; masa=%.9g kg"
          % (density, volume, mass))
    print("MODELO DE DENSIDAD: cinco estratos; densidades [kg/m³] =", layer_densities)
    print("Eje de estratos =", layer_axis, "; extremos muestreados y cortes [m] =", layer_edges)
    print("Volúmenes de los cinco estratos [m³] =", layer_volumes)
    print("En equilibrio sobre plano: azul=%.4f° respecto a +Y" % stable_angle)
    print("COM local desde origen radial [m] =", cog)
    print("Inercia en COM, ejes locales [kg m²] =\n", inertia)
    print("inclinación estable efectiva=%.3f°; posición COM=%s m; "
          "v=(0,0,0) m/s; omega=(0,0,0) rad/s"
          % (initial_tilt, pos))
    if args.punto_abajo is None:
        print("Convención de inclinación: rotación de %.3f° alrededor del eje X mundial "
              "desde el equilibrio estable." % requested_inclination)
    else:
        print("Condición inicial P%02d: dirección radial del punto rojo %d alineada "
              "con -Y; la posición COM se calcula con el soporte analítico."
              % (args.punto_abajo, args.punto_abajo))
    print("Control plano: soporte puntual radial con muelle elástico, sin amortiguador"
          if args.plano else "Terreno elastoplástico Bekker simplificado")
    print("dt=%g s; rejilla=%g m; iteraciones=%d; cuadratura masa=%dx%d; g=%g m/s²"
          % (args.dt, args.rejilla, args.iteraciones, args.ntheta, args.nphi, G))
    print("Relieve sintético: amplitud=%g m; longitud=%g m; fase=%g rad; no calibrado"
          % (0. if args.plano else args.relieve, args.longitud_relieve, args.fase_relieve))
    print("Hundimiento: profundidad máxima en columnas respecto a su altura inicial; "
          "soporte respecto a y=0 es un monitor distinto.")
    if not args.plano and args.relieve:
        print("Liberación desde una cota superior sin interpenetración; posible caída inicial.")
    print("k_el=%g Pa/m; k_base=%g Pa/m; base=%g m; umbral=%g m; mu=%g; v_regularización=%g m/s"
          % (ground.elastic, BASE_K, ground.base_depth, DEPTH_LIMIT,
             args.mu, args.velocidad_regularizacion))
    print("Amortiguamiento normal del suelo=%g Pa·s/m (no calibrado; compresión únicamente)"
          % args.amortiguamiento_normal)
    if args.plano:
        print("k_plano=%g N/m; compresión inicial mg/k=%.6g m; "
              "plano penalizado, no perfectamente rígido" % (PLANE_K, mass*G/PLANE_K))
        if args.amortiguamiento_normal:
            print("AVISO: --amortiguamiento-normal solo actúa en el suelo deformable, "
                  "no en el control de plano.")
    print("Bekker: kc=%g, kphi=%g, n=%g, b=%g m; factor resistencia=%g; sin calibración"
          % (ground.kc, ground.kphi, EXPONENT, args.diametro, args.factor_resistencia))
    print("Reposo cinemático: |v|<0.01 m/s y |omega|<0.01 rad/s durante 1 s; "
          "asentamiento geométrico adicionalmente exige inclinación plana <= %.3f°."
          % GEOMETRIC_EQUILIBRIUM_TOL_DEG)
    print("Las fuerzas y momentos se evalúan numéricamente; un balance instantáneo "
          "no demuestra equilibrio.")
    print("AVISO: prototipo no validado; comprueba convergencia y balance "
          "energético antes de interpretar el movimiento.")
    initial_energy = mass * G * pos[1]
    stable_height = np.linalg.norm(red)
    maneuver_energy = mass * G * (-float(matrix[1, :] @ point) - stable_height)
    if args.plano:
        initial_energy += .5 * PLANE_K * (mass*G/PLANE_K)**2
    settled_since = None
    last_settled = False
    first_settled = None
    max_energy_error = 0.
    stop_on_transient_pass = args.detener_paso_transitorio
    transient_pass_detected = False
    previous_reported_angular_speed = None
    up_initial = -(matrix @ low)
    signed_reference = np.array([up_initial[0], 0., up_initial[2]])
    reference_length = np.linalg.norm(signed_reference)
    if reference_length < 1e-12:
        signed_reference = np.array([0., 0., 1.])
    else:
        signed_reference /= reference_length
    signed_angle, tilt = signed_tilt(up_initial, signed_reference)
    initial_signed_angle = signed_angle
    previous_up = up_initial
    success_reason = "angulo_menor_7_5" if tilt < 7.5 else None
    success_time = 0. if success_reason else None
    crossing_time = None
    print("Inclinación firmada: proyección sobre el plano vertical inicial; "
          "positivo hacia el lado inicial. En 3D no sustituye al ángulo total.")
    if success_reason:
        print("ÉXITO OPERATIVO inicial: inclinación total <7,5°; "
              "no demuestra asentamiento.")
    start = time.monotonic()
    steps = math.ceil(args.tiempo / args.dt)
    next_report = 1.
    now = 0.
    angle = math.degrees(math.acos(np.clip((matrix @ blue)[1]/np.linalg.norm(blue), -1, 1)))
    angular_speed = linear_speed = balance = 0.
    seen_contact = False
    was_contact = False
    was_geometric = False
    geometric_entries = geometric_losses = 0
    contact_loss_events = 0
    contact_start_events = 0
    was_airborne_rising = False
    airborne_rising_events = 0
    max_upward_speed_airborne = 0.
    point_direction = point + cog
    point_direction /= np.linalg.norm(point_direction)
    monitor_angles = [math.acos(np.clip(point_direction[2], -1, 1)),
                      math.atan2(point_direction[1], point_direction[0])]
    body_depth, monitor_angles = ground.measure_body_depth(
        body, scale, cog, monitor_angles, args.limite_estricto)
    if args.limite_estricto:
        ground.check_depth_limit()
    for step in range(0 if success_reason and not args.sin_parada_exito else steps):
        dt = min(args.dt, args.tiempo - step * args.dt)
        if visual is not None:
            visual.step(step)
        body.EmptyAccumulator(acc)
        ground.apply(body, chrono, acc, scale, cog, dt)
        if args.limite_estricto:
            ground.check_depth_limit()
        in_contact = ground.contacts > 0
        in_geometric = ground.geometric_contacts > 0
        if in_geometric and not was_geometric:
            geometric_entries += 1
        if was_geometric and not in_geometric:
            geometric_losses += 1
        was_geometric = in_geometric
        if in_contact and not was_contact:
            contact_start_events += 1
        if was_contact and not in_contact:
            contact_loss_events += 1
        was_contact = in_contact
        if ground.contacts:
            seen_contact = True
        airborne_rising = seen_contact and ground.contacts == 0 and body.GetLinVel().y > 0.
        if airborne_rising and not was_airborne_rising:
            airborne_rising_events += 1
        if airborne_rising:
            max_upward_speed_airborne = max(max_upward_speed_airborne, body.GetLinVel().y)
        was_airborne_rising = airborne_rising
        system.DoStepDynamics(dt)
        now = system.GetChTime()
        v = vector(body.GetLinVel())
        omega = vector(body.GetAngVelParent())
        if not (np.all(np.isfinite(v)) and np.all(np.isfinite(omega))
                and np.all(np.isfinite(vector(body.GetPos())))):
            raise RuntimeError("Estado no finito; la integración es inestable")
        linear_speed, angular_speed = np.linalg.norm(v), np.linalg.norm(omega)
        if linear_speed < .01 and angular_speed < .01:
            if settled_since is None:
                settled_since = now
        else:
            settled_since = None
        last_settled = settled_since is not None and now - settled_since >= 1.
        if last_settled and first_settled is None:
            first_settled = now
        rot = axes(body)
        up = -(rot @ low)
        signed_angle, tilt = signed_tilt(up, signed_reference)
        crossing = (crossing_time is None and
                    crossed_vertical_near_equilibrium(previous_up, up, signed_reference))
        if crossing:
            crossing_time = now
            print("CRUCE cercano de la vertical detectado en t=%.6f s." % now, flush=True)
        if success_reason is None:
            if crossing:
                success_reason, success_time = "cruce_vertical_cercano", now
            elif tilt < 7.5:
                success_reason, success_time = "angulo_menor_7_5", now
            if success_reason is not None:
                print("ÉXITO OPERATIVO t=%.6f s: %s; no demuestra asentamiento."
                      % (now, success_reason), flush=True)
        previous_up = up
        stop_for_success = success_reason is not None and not args.sin_parada_exito
        report_due = now + 1e-9 >= next_report or step == steps-1 or stop_for_success
        measurement_due = (
            args.limite_estricto
            or (step + 1) % args.body_depth_interval_steps == 0
            or report_due
        )
        if measurement_due:
            body_depth, monitor_angles = ground.measure_body_depth(
                body, scale, cog, monitor_angles, args.limite_estricto)
            if args.limite_estricto:
                ground.check_depth_limit()
        if report_due:
            blue_world = rot @ blue
            angle = math.degrees(math.acos(np.clip(blue_world[1]/np.linalg.norm(blue), -1, 1)))
            local_omega = rot.T @ omega
            kinetic = .5*mass*linear_speed**2 + .5*local_omega @ inertia @ local_omega
            energy = mass*G*body.GetPos().y + kinetic + ground.elastic_energy
            balance = (energy + ground.plastic_work + ground.friction_work
                       + ground.damping_work - initial_energy)
            max_energy_error = max(max_energy_error, abs(balance))
            print("t=%7.3f azul=%7.3f° (estable %.3f°) |ω|=%9.5f rad/s "
                  "|v|=%9.5f m/s profundidad_local=%.3f mm plástico_max=%.3f mm "
                  "columnas_nivel_original=%d cargadas=%d base=%d balance=%+.5g J "
                  "inclinación_residual=%.3f° firmada=%.3f°"
                  % (now, angle, stable_angle, angular_speed, linear_speed,
                     1000*ground.local_depth, 1000*ground.plastic.max(),
                     ground.geometric_contacts, ground.contacts, ground.base_contacts,
                     balance, tilt, signed_angle), flush=True)
            if stop_for_success:
                break
            if (stop_on_transient_pass and should_stop_after_transient_pass(
                    tilt, angular_speed, previous_reported_angular_speed, now)):
                transient_pass_detected = True
                print("PARADA TEMPRANA: ángulo <7,5° y |ω| menor que en "
                      "el reporte del segundo anterior. Esto identifica un "
                      "paso transitorio, no asentamiento ni equilibrio.",
                      flush=True)
                break
            previous_reported_angular_speed = angular_speed
            next_report += 1
    # Evalúa también la deformación en el último estado integrado.
    body.EmptyAccumulator(acc)
    ground.apply(body, chrono, acc, scale, cog, 0.)
    if args.limite_estricto:
        ground.check_depth_limit()
    rot = axes(body)
    blue_world = rot @ blue
    angle = math.degrees(math.acos(np.clip(
        blue_world[1] / np.linalg.norm(blue), -1, 1)))
    tilt = math.degrees(math.acos(np.clip(-(rot @ low)[1], -1, 1)))
    signed_angle, _ = signed_tilt(-(rot @ low), signed_reference)
    local_omega = rot.T @ vector(body.GetAngVelParent())
    kinetic = .5*mass*linear_speed**2 + .5*local_omega @ inertia @ local_omega
    energy = mass*G*body.GetPos().y + kinetic + ground.elastic_energy
    balance = (energy + ground.plastic_work + ground.friction_work
               + ground.damping_work - initial_energy)
    max_energy_error = max(max_energy_error, abs(balance))
    final_support = support(rot[1, :], scale, cog)
    residual_potential = mass * G * (-float(rot[1, :] @ final_support) - stable_height)
    near_equilibrium = tilt <= GEOMETRIC_EQUILIBRIUM_TOL_DEG
    fully_settled = last_settled and near_equilibrium
    final_pos = vector(body.GetPos())
    potential = mass * G * final_pos[1]
    dissipated = ground.plastic_work + ground.friction_work + ground.damping_work
    orientation_improved = tilt < initial_tilt - 1.
    final_geometric_contacts = ground.geometric_contacts
    final_loaded_contacts = ground.contacts
    if final_geometric_contacts == 0:
        final_state = "sin contacto geométrico al final; posible vuelo/separación"
    elif linear_speed >= .01 or angular_speed >= .01:
        final_state = ("movimiento con cambio angular hacia equilibrio; "
                       "autoorientación no confirmada" if orientation_improved
                       else "movimiento residual sin autoorientación neta")
    elif final_loaded_contacts == 0:
        final_state = ("reposo cinemático con huella geométrica descargada"
                       if not near_equilibrium else "casi equilibrio; huella descargada")
    elif orientation_improved:
        final_state = ("reposo cinemático tras cambio angular; atrapamiento no demostrado"
                       if not near_equilibrium else "reposo cerca del equilibrio geométrico")
    else:
        final_state = "reposo cinemático sin aproximación al equilibrio"
    print("FINAL t=%.4f s azul=%.4f° inclinación_plano=%.4f° firmada=%.4f° |ω|=%.6g rad/s "
          "|v|=%.6g m/s E_cinética=%.9g J reposo_cinemático=%s asentamiento_geométrico=%s"
          % (now, angle, tilt, signed_angle, angular_speed, linear_speed, kinetic,
             "sí" if last_settled else "no", "sí" if fully_settled else "no"))
    print("Estado final: %s" % final_state)
    if last_settled and not near_equilibrium:
        print("Reposo lejos del equilibrio geométrico: posible equilibrio condicionado "
              "por la huella; el atrapamiento no queda demostrado solo por las velocidades.")
    print("Máximos: profundidad local cuerpo=%.4f mm; plástico=%.4f mm; "
          "compresión elástica=%.4f mm; columnas distintas que alcanzaron la base=%d; "
          "fuerza=%.5g N; presión vertical=%.5g Pa" %
          (1000*ground.max_local_depth, 1000*ground.plastic.max(),
           1000*ground.max_elastic_compression, np.count_nonzero(ground.base_reached),
           ground.maximum_force, ground.maximum_pressure))
    print("Soporte respecto a y=0: máximo=%.4f mm; cota condicional=%.4f mm"
          % (1000*ground.max_body, 1000*ground.max_body_bound))
    if np.any(ground.base_reached):
        print("AVISO: columnas alcanzaron la base; el hundimiento puede estar condicionado "
              "por ella y no es una predicción de suelo profundo.")
    angle_pass, linear_speed_pass, angular_speed_pass = success_checks(
        tilt, linear_speed, angular_speed)
    print("Criterios de éxito: ángulo final <7,5° %s; "
          "velocidad lineal final <1 m/s %s; velocidad angular final "
          "<1 rad/s %s; resultado operativo=%s (motivo=%s). "
          "Velocidades y hundimiento son informativos, no excluyentes; "
          "el éxito no demuestra asentamiento."
          % ("cumple" if angle_pass else "NO cumple",
             "cumple" if linear_speed_pass else "NO cumple",
             "cumple" if angular_speed_pass else "NO cumple",
             "ÉXITO" if success_reason else "NO ÉXITO",
             success_reason or "no alcanzado"))
    print("Primer cumplimiento del criterio de reposo cinemático: %s"
          % ("no detectado" if first_settled is None else "%.4f s" % first_settled))
    print("Energía: balance=%+.6g J, máximo absoluto muestreado=%.6g J; "
          "disipación fricción=%.6g J, amortiguamiento=%.6g J, trabajo plástico=%.6g J; "
          "energía elástica=%.6g J; E0=%.6g J"
          % (balance, max_energy_error, ground.friction_work,
             ground.damping_work, ground.plastic_work, ground.elastic_energy, initial_energy))
    print("Energía final: potencial gravitatoria (datum y=0)=%.9g J; "
          "cinética=%.9g J; disipada contabilizada (plástico+fricción+amortiguamiento)=%.9g J."
          % (potential, kinetic, dissipated))
    print("Energía potencial de orientación sobre plano: residual=%+.9g J; "
          "energía inicial de maniobra=%+.9g J (no incluye trabajo del terreno)."
          % (residual_potential, maneuver_energy))
    if maneuver_energy > 1e-12:
        print("Balance absoluto máximo / energía de maniobra = %.6g%%"
              % (100 * max_energy_error / maneuver_energy))
        if max_energy_error > .01 * maneuver_energy:
            print("AVISO: balance superior al 1% de la energía de maniobra; resultado no validado.")
    else:
        print("Balance normalizado no disponible: energía de maniobra nula o negativa.")
    audit_thresholds = {0.1: .877, 0.2: .912, 0.3: .915}
    reference = next((v for d, v in audit_thresholds.items()
                      if abs(args.diametro - d) < 1e-9), None)
    if reference is not None and not args.plano:
        ratio = tilt / reference
        comparison = "por debajo" if ratio < .75 else "por encima" if ratio > 1.25 else "cerca"
        print("Umbral de rodadura de la auditoría para suelo de referencia: %.3f°; "
              "ángulo/umbral=%.3f (%s, margen descriptivo 25%%). Estimación estática "
              "externa, no calculada para esta huella ni validada para otros parámetros."
              % (reference, ratio, comparison))
    else:
        print("Umbral de rodadura de la auditoría: no disponible para este tamaño/control.")
    print("Episodios ascendentes sin columnas cargadas tras carga=%d; "
          "velocidad vertical máxima=%.6g m/s (detección discreta, no prueba de ausencia de rebotes)."
          % (airborne_rising_events, max_upward_speed_airborne))
    print("Transiciones de contacto: entradas=%d; pérdidas totales=%d "
          "(columnas cargadas)."
          % (contact_start_events, contact_loss_events))
    print("Transiciones bajo nivel original: entradas=%d; pérdidas=%d. "
          "Estar bajo el nivel original no demuestra contacto con el suelo deformado; "
          "ausencia de carga tampoco demuestra separación. No se certifica ausencia de rebotes."
          % (geometric_entries, geometric_losses))
    result = {
        "density_model": "layered",
        "density_profile": {
            "type": "five_layers",
            "densities_kg_m3": layer_densities.tolist(),
            "volumes_m3": layer_volumes.tolist(),
            "axis_up_local": layer_axis.tolist(),
            "sampled_edges_m": layer_edges.tolist(),
            "outer_layers_extend_to_surface": True,
        },
        "diameter_m": args.diametro,
        "initial_condition_id": (None if args.punto_abajo is None
                                 else f"P{args.punto_abajo:02d}"),
        "initial_marker_index": args.punto_abajo,
        "initial_pose_source": initial_pose_source,
        "requested_inclination_deg": requested_inclination,
        "initial_tilt_deg": initial_tilt,
        "initial_signed_tilt_deg": initial_signed_angle,
        "signed_tilt_reference_world": signed_reference.tolist(),
        "initial_position_com_m": initial_pos.tolist(),
        "initial_rotation_xyzw": rotation.as_quat().tolist(),
        "initial_linear_velocity_m_s": [0.0, 0.0, 0.0],
        "initial_angular_velocity_rad_s": [0.0, 0.0, 0.0],
        "final_blue_angle_deg": angle,
        "final_residual_tilt_deg": tilt,
        "final_signed_tilt_deg": signed_angle,
        "success_reason": success_reason,
        "success_time_s": success_time,
        "vertical_crossing_time_s": crossing_time,
        "success_stop_enabled": not args.sin_parada_exito,
        "contact_search_backend": "native" if native_search else "numpy",
        "angle_limit_deg": 7.5,
        "angle_below_limit": angle_pass,
        "linear_speed_limit_m_s": 1.0,
        "linear_speed_below_limit": linear_speed_pass,
        "angular_speed_limit_rad_s": 1.0,
        "angular_speed_below_limit": angular_speed_pass,
        "success_criteria_met": success_reason is not None,
        "outcome_classification": "success" if success_reason else "not_successful",
        "kinematic_settled_at_end": bool(last_settled),
        "geometric_settled_at_end": bool(fully_settled),
        "geometric_equilibrium_tolerance_deg": GEOMETRIC_EQUILIBRIUM_TOL_DEG,
        "linear_speed_m_s": linear_speed,
        "angular_speed_rad_s": angular_speed,
        "kinetic_energy_j": kinetic,
        "potential_energy_j_datum_y0": potential,
        "dissipated_energy_j_accounted": dissipated,
        "energy_balance_j": balance,
        "maximum_energy_error_j": max_energy_error,
        "maximum_energy_error_pct_maneuver": (
            100 * max_energy_error / maneuver_energy
            if maneuver_energy > 1e-12 else None),
        "first_kinematic_settlement_s": first_settled,
        "max_body_local_depth_m": ground.max_local_depth,
        "max_body_depth_from_initial_plane_m": ground.max_body,
        "max_body_depth_bound_m": ground.max_body_bound,
        "max_soil_plastic_deformation_m": float(ground.plastic.max()),
        "max_elastic_compression_m": ground.max_elastic_compression,
        "base_reached_column_count": int(np.count_nonzero(ground.base_reached)),
        "rebound_events_detected": airborne_rising_events,
        "loaded_contact_losses": contact_loss_events,
        "geometric_contact_losses": geometric_losses,
        "final_geometric_contact_columns": final_geometric_contacts,
        "final_loaded_contact_columns": final_loaded_contacts,
        "final_state": final_state,
        "duration_s": now,
        "mass_kg": mass,
        "density_kg_m3": density,
        "volume_m3": volume,
        "gravity_m_s2": G,
        "epsilon": EPS,
        "bekker_exponent": EXPONENT,
        "base_stiffness_pa_m": BASE_K,
        "base_depth_m": ground.base_depth,
        "relief_amplitude_m": 0.0 if args.plano else args.relieve,
        "relief_wavelength_m": args.longitud_relieve,
        "relief_phase_rad": args.fase_relieve,
        "mass_quadrature_ntheta": args.ntheta,
        "mass_quadrature_nphi": args.nphi,
        "iterations_per_contact_search": args.iteraciones,
        "body_depth_interval_steps": args.body_depth_interval_steps,
        "transient_pass_stop_enabled": stop_on_transient_pass,
        "transient_pass_detected": transient_pass_detected,
        "com_local_m": cog.tolist(),
        "inertia_local_kg_m2": inertia.tolist(),
        "final_contact_force_n": ground.last_force.tolist(),
        "final_contact_moment_nm": ground.last_torque.tolist(),
        "maximum_contact_force_n": ground.maximum_force,
        "maximum_contact_moment_nm": ground.maximum_torque,
        "maximum_vertical_pressure_pa": ground.maximum_pressure,
        "friction_coefficient": args.mu,
        "elastic_unloading_pa_m": ground.elastic,
        "bekker_kc": ground.kc,
        "bekker_kphi": ground.kphi,
        "bekker_resistance_factor": args.factor_resistencia,
        "normal_damping_pa_s_m": args.amortiguamiento_normal,
        "grid_spacing_m": args.rejilla,
        "time_step_s": args.dt,
    }
    print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")),
          flush=True)
    print("El balance debe aproximarse a cero; se evalúa con fuerzas y energía "
          "elástica del paso anterior, por lo que incluye error temporal.")
    print("Tiempo real %.2f s. Contacto, soporte y energías evaluados numéricamente; "
          "la cota supone soporte global correcto en cada refresco; "
          "máximos entre pasos no garantizados." % (time.monotonic()-start))
    if visual is not None:
        visual.finish()


def main(visual_factory=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plano", action="store_true", help="control elástico sin suelo plástico")
    parser.add_argument("--limite-estricto", action="store_true",
                        help="abortar si la medición del suelo o la cota del cuerpo supera 20 mm")
    parser.add_argument("--diametro", type=float, default=1., help="mayor dimensión local [m]")
    mass_group = parser.add_mutually_exclusive_group()
    mass_group.add_argument("--densidad", type=float, default=RHO,
                            help="densidad del estrato inferior: 2700 kg/m³")
    mass_group.add_argument("--masa", type=float,
                            help="no compatible con las densidades absolutas de los estratos")
    initial_group = parser.add_mutually_exclusive_group()
    initial_group.add_argument(
        "--inclinacion", type=float, default=None,
        help="grados desde el equilibrio estable (por defecto 15°)")
    initial_group.add_argument(
        "--punto-abajo", type=int, choices=range(1, N_PUNTOS + 1),
        help=f"ID 1..{N_PUNTOS} de punto rojo; reproduce la orientación PUNTO_ABAJO")
    parser.add_argument("--tiempo", type=float, default=5., help="duración simulada [s]")
    parser.add_argument(
        "--sin-parada-exito", action="store_true",
        help=("continuar tras el éxito operativo, salvo otra parada explícita "
              "como --detener-paso-transitorio"),
    )
    parser.add_argument(
        "--acelerar-contacto", action="store_true",
        help="usar la búsqueda radial nativa compilada de contacto_radial9.c",
    )
    parser.add_argument(
        "--detener-paso-transitorio", action="store_true",
        help=("detener en un reporte por segundo a partir de 2 s si inclinación "
              "<7,5° y |omega| disminuye respecto al segundo anterior; no "
              "certifica equilibrio"),
    )
    parser.add_argument("--dt", type=float, default=.0001, help="paso [s]")
    parser.add_argument("--rejilla", type=float, default=.0375, help="paso del terreno [m]")
    parser.add_argument(
        "--intervalo-medicion-cuerpo-pasos", dest="body_depth_interval_steps",
        type=int, default=10,
        help=("medir soporte/hundimiento cada N pasos; solo afecta al muestreo "
              "diagnóstico, no a fuerzas; --limite-estricto comprueba cada paso"),
    )
    parser.add_argument(
        "--iteraciones", type=int, default=22,
        help="máximo de iteraciones de oro (y +4 para bisección); predeterminado 22",
    )
    parser.add_argument("--ntheta", type=int, default=80, help="cuadratura de masa en theta")
    parser.add_argument("--nphi", type=int, default=160, help="cuadratura de masa en phi")
    parser.add_argument("--mu", type=float, default=MU, help="fricción de Coulomb cinética")
    parser.add_argument("--velocidad-regularizacion", type=float, default=SLIP_SPEED,
                        help="regularización de la fricción [m/s]")
    parser.add_argument("--relieve", type=float, default=0.,
                        help="amplitud sintética inicial [m]; 0 para suelo plano")
    parser.add_argument("--fase-relieve", type=float, default=0., help="fase del relieve [rad]")
    parser.add_argument("--longitud-relieve", type=float, default=.12,
                        help="longitud de onda del relieve sintético [m]")
    parser.add_argument("--amortiguamiento-normal", type=float, default=0.,
                        help="amortiguamiento de compresión del suelo [Pa s/m]; no calibrado")
    parser.add_argument("--factor-resistencia", type=float, default=1.,
                        help="multiplicador exploratorio de kc y kphi; no calibrado")
    parser.add_argument("--kc", type=float, default=KC, help="coeficiente Bekker kc")
    parser.add_argument("--kphi", type=float, default=KPHI, help="coeficiente Bekker kphi")
    parser.add_argument("--rigidez-descarga", type=float, default=ELASTIC,
                        help="rigidez de descarga elástica [Pa/m]; no calibrada")
    parser.add_argument("--profundidad-base", type=float, default=LAYER,
                        help="profundidad física de la base penalizada [m], independiente del umbral")
    args = parser.parse_args()
    if args.plano and args.limite_estricto:
        parser.error("--limite-estricto solo se aplica al terreno deformable, "
                     "no al control --plano.")
    if args.relieve and args.limite_estricto:
        parser.error("--limite-estricto con relieve no dispone de una cota local fiable; "
                     "utiliza --relieve 0 o mide sin interrupción estricta.")
    if (not all(math.isfinite(x) and x > 0 for x in
                (args.diametro, args.tiempo, args.dt, args.rejilla,
                 args.velocidad_regularizacion, args.longitud_relieve))
            or not math.isfinite(args.rigidez_descarga) or args.rigidez_descarga <= 0
            or not math.isfinite(args.profundidad_base) or args.profundidad_base <= 0
            or not math.isfinite(args.fase_relieve)
            or not math.isfinite(args.densidad) or args.densidad <= 0
            or (args.masa is not None
                and (not math.isfinite(args.masa) or args.masa <= 0))
            or not math.isfinite(args.kc) or args.kc < 0
            or not math.isfinite(args.kphi) or args.kphi < 0
            or not math.isfinite(args.factor_resistencia) or args.factor_resistencia <= 0
            or (args.inclinacion is not None and not math.isfinite(args.inclinacion))
            or not math.isfinite(args.mu) or args.mu < 0
            or not math.isfinite(args.relieve) or args.relieve < 0
            or not math.isfinite(args.amortiguamiento_normal) or args.amortiguamiento_normal < 0
            or args.iteraciones < 4 or args.ntheta < 8 or args.nphi < 16
            or args.body_depth_interval_steps < 1):
        parser.error("Diámetro, tiempo, dt, rejilla, rigidez y profundidad de base "
                     "deben ser positivos; fase e inclinación finitas; parámetros de "
                     "contacto válidos; iteraciones>=4, ntheta>=8 y nphi>=16.")
    run(args, visual_factory=visual_factory)


if __name__ == "__main__":
    main()
