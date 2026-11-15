#!/usr/bin/env python3
"""Representa y registra los puntos de apoyo sobre la superficie del Gömböc."""

# Para comparacion30, GOMBOC_CSV_PUNTOS indica la ruta del CSV original.
# Sin esa variable se busca condiciones_comparacion_30_originales.csv
# junto a este archivo; fase1 genera sus 15 puntos sin ese recurso.


import csv
import math
import os
from pathlib import Path
import struct
import zlib

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


# Parámetros principales, iguales a los del script de referencia.
DIAMETRO = 0.20
MASA_GOMBOC_KG = 4.0
EPSILON = 0.03
RELACION_DENSIDAD_BASE = 75.0
NUM_ESTRATOS = 5

# Configuración de los puntos, las etiquetas y su orientación opcional.
_MODO_PUNTOS = os.environ.get("GOMBOC_MODO_PUNTOS", "comparacion30")
if _MODO_PUNTOS not in ("comparacion30", "fase1"):
    raise ValueError(f"GOMBOC_MODO_PUNTOS no reconocido: {_MODO_PUNTOS}")
N_PUNTOS = 30 if _MODO_PUNTOS == "comparacion30" else 15
CSV_PUNTOS_ORIGINALES = Path(os.environ.get(
    "GOMBOC_CSV_PUNTOS",
    Path(__file__).resolve().parent / "condiciones_comparacion_30_originales.csv",
))
TAMANO_NUMERO = 0.016
COLOR_NUMERO = (1.0, 0.0, 0.0)
MOSTRAR_MARCA_PUNTO = True
RADIO_MARCA = 0.005 * DIAMETRO
COLOR_EMISIVO_NUMERO = (0.85, 0.0, 0.0)
COLOR_EMISIVO_GOMBOC = (0.035, 0.08, 0.14)
RADIO_MARCADOR_EQUILIBRIO = 0.012 * DIAMETRO
PUNTO_ABAJO = None

_GLIFOS = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("110", "001", "111", "100", "111"),
    "3": ("110", "001", "111", "001", "110"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "110", "001", "110"),
    "6": ("011", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "110"),
    "A": ("010", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("011", "100", "100", "100", "011"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("011", "100", "101", "101", "011"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "010"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("010", "101", "101", "101", "010"),
    "P": ("110", "101", "110", "100", "100"),
    "Q": ("010", "101", "101", "111", "011"),
    "R": ("110", "101", "110", "101", "101"),
    "S": ("011", "100", "010", "001", "110"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "Y": ("101", "101", "010", "010", "010"),
    "+": ("000", "010", "111", "010", "000"),
    "(": ("010", "100", "100", "100", "010"),
    ")": ("010", "001", "001", "001", "010"),
    " ": ("000", "000", "000", "000", "000"),
}


def crear_geometria(diametro, masa_objetivo, epsilon):
    """Reproduce la malla, escala y propiedades geométricas de referencia."""
    n_theta = 121
    n_phi = 240
    theta = np.linspace(0.0, math.pi, n_theta + 1)
    phi = 2.0 * math.pi * np.arange(n_phi) / n_phi
    th = theta[1:-1, None]
    ph = phi[None, :]
    coeficiente = 4.0 * epsilon
    radicando = (1.0 + coeficiente * np.sin(th)
                 * np.cos(ph - 5.0 * th))
    radio = np.power(radicando, 0.25)
    anillos = np.stack((
        radio * np.sin(th) * np.cos(ph),
        radio * np.sin(th) * np.sin(ph),
        radio * np.cos(th),
    ), axis=2).reshape(-1, 3)
    vertices = np.vstack((
        np.array([[0.0, 0.0, 1.0]]),
        anillos,
        np.array([[0.0, 0.0, -1.0]]),
    ))

    ip = np.arange(n_phi)
    ip_sig = (ip + 1) % n_phi
    ring = 1 + np.arange(n_theta - 1)[:, None] * n_phi + ip[None, :]
    faces = [np.column_stack((
        np.zeros(n_phi, dtype=np.int64),
        ring[0, ip_sig],
        ring[0, ip],
    ))]
    for j in range(n_theta - 2):
        upper = ring[j]
        lower = ring[j + 1]
        faces.extend((
            np.column_stack((upper, upper[ip_sig], lower[ip_sig])),
            np.column_stack((upper, lower[ip_sig], lower)),
        ))
    south = len(vertices) - 1
    faces.append(np.column_stack((
        np.full(n_phi, south, dtype=np.int64),
        ring[-1, ip],
        ring[-1, ip_sig],
    )))
    faces = np.vstack(faces)

    scale = diametro / (vertices.max(0) - vertices.min(0)).max()
    vertices *= scale
    a, b, c = (vertices[faces[:, 0]], vertices[faces[:, 1]],
               vertices[faces[:, 2]])
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    if volume < 0:
        faces[:, [1, 2]] = faces[:, [2, 1]]
        a, b, c = (vertices[faces[:, 0]], vertices[faces[:, 1]],
                   vertices[faces[:, 2]])
        volume = -volume
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("La ecuación generó un volumen no positivo")

    tetra_volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    cog = (tetra_volumes[:, None] * ((a + b + c) / 4.0)).sum(0) / volume
    face_sum = a + b + c
    weights = tetra_volumes / 20.0
    second_moment = (
        np.einsum("i,ij,ik->jk", weights, a, a)
        + np.einsum("i,ij,ik->jk", weights, b, b)
        + np.einsum("i,ij,ik->jk", weights, c, c)
        + np.einsum("i,ij,ik->jk", weights, face_sum, face_sum)
    )
    densidad = masa_objetivo / volume
    inertia = densidad * (
        np.trace(second_moment) * np.eye(3) - second_moment
        - volume * ((cog @ cog) * np.eye(3) - np.outer(cog, cog))
    )
    local_vertices = vertices - cog
    if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
        raise ValueError("La matriz de inercia no es definida positiva")
    return (coeficiente, scale, cog, local_vertices, faces, masa_objetivo,
            densidad, inertia)


def direccion_esferica(theta, phi):
    return np.array([
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    ])


def crear_puntos_caracteristicos(coeficiente, scale, cog, local_vertices,
                                 initial_low=None):
    """Busca el eje estable usando la ecuación radial exacta."""
    directions_count = 8000 if initial_low is None else 256
    i = np.arange(directions_count)
    z = 1.0 - 2.0 * (i + 0.5) / directions_count
    xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * math.pi * i / ((1.0 + math.sqrt(5.0)) / 2.0)
    directions = np.column_stack((xy * np.cos(phi), z, xy * np.sin(phi)))
    heights = np.empty(directions_count)
    block = max(1, int(48e6 / (8 * len(local_vertices))))
    for start in range(0, directions_count, block):
        heights[start:start + block] = (
            local_vertices @ directions[start:start + block].T
        ).max(axis=0)

    def superficie(theta, phi):
        radial = (1.0 + coeficiente * math.sin(theta)
                  * math.cos(phi - 5.0 * theta)) ** 0.25
        return scale * radial * direccion_esferica(theta, phi) - cog

    def soporte(direction, start):
        def value_gradient(angles):
            theta, phi = angles
            st, ct = math.sin(theta), math.cos(theta)
            delta = phi - 5.0 * theta
            sd, cd = math.sin(delta), math.cos(delta)
            f = 1.0 + coeficiente * st * cd
            radius = f ** 0.25
            factor = 0.25 * f ** -0.75
            r_theta = factor * coeficiente * (ct * cd + 5.0 * st * sd)
            r_phi = -factor * coeficiente * st * sd
            unit = np.array([st * math.cos(phi), st * math.sin(phi), ct])
            unit_theta = np.array([
                ct * math.cos(phi), ct * math.sin(phi), -st])
            unit_phi = np.array([-st * math.sin(phi), st * math.cos(phi), 0])
            point = scale * radius * unit - cog
            d_theta = scale * (r_theta * unit + radius * unit_theta)
            d_phi = scale * (r_phi * unit + radius * unit_phi)
            return float(direction @ point), np.array([
                direction @ d_theta, direction @ d_phi])

        result = minimize(
            value_gradient, start, method="L-BFGS-B", jac=True,
            bounds=((1e-9, math.pi - 1e-9), (None, None)),
            options={"ftol": 1e-16, "gtol": 1e-15, "maxiter": 80},
        )
        if not result.success and np.linalg.norm(result.jac) > 1e-6:
            raise RuntimeError("Falló la búsqueda del soporte: %s"
                               % result.message)
        return result.x, superficie(*result.x)

    def stable_height(angles):
        theta, phi = angles
        direction = direccion_esferica(theta, phi)
        _, point = soporte(-direction, np.array([theta, phi]))
        d_theta = np.array([
            math.cos(theta) * math.cos(phi),
            math.cos(theta) * math.sin(phi),
            -math.sin(theta),
        ])
        d_phi = np.array([
            -math.sin(theta) * math.sin(phi),
            math.sin(theta) * math.cos(phi),
            0.0,
        ])
        return float(direction @ point), np.array([
            point @ d_theta, point @ d_phi])

    if initial_low is None:
        low = directions[int(np.argmin(heights))]
        start = np.array([
            math.acos(float(np.clip(low[2], -1, 1))),
            math.atan2(low[1], low[0]),
        ])
    else:
        low = np.asarray(initial_low, dtype=float)
        start = np.array([
            math.acos(float(np.clip(low[2], -1, 1))),
            math.atan2(low[1], low[0]),
        ])
    stable = minimize(
        stable_height, start, method="L-BFGS-B", jac=True,
        bounds=((1e-9, math.pi - 1e-9), (None, None)),
        options={"ftol": 1e-14, "gtol": 1e-11, "maxiter": 60},
    )
    if not stable.success and np.linalg.norm(stable.jac) > 1e-7:
        raise RuntimeError("Falló la búsqueda del equilibrio: %s"
                           % stable.message)
    theta, phi = stable.x
    low = direccion_esferica(theta, phi)
    angles = np.array([
        math.acos(float(np.clip(low[2], -1, 1))),
        math.atan2(low[1], low[0]),
    ])
    _, support_point = soporte(-low, angles)
    opposite_angles = np.array([math.pi - angles[0], angles[1] + math.pi])
    _, blue_point = soporte(low, opposite_angles)
    return low, support_point, blue_point, float(stable.fun), heights, soporte


def integrar_estratos(vertices, scale, coeficiente, axis_up, total_mass):
    """Integra las cinco densidades con el mismo método del script base."""
    projections = vertices @ axis_up
    bottom = float(projections.min())
    top = float(projections.max())
    edges = np.linspace(bottom, top, NUM_ESTRATOS + 1)

    mu, mu_weights = np.polynomial.legendre.leggauss(200)
    phi = 2.0 * math.pi * np.arange(400) / 400
    theta = np.arccos(mu)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    directions = np.column_stack((
        (sin_theta[:, None] * np.cos(phi[None, :])).ravel(),
        (sin_theta[:, None] * np.sin(phi[None, :])).ravel(),
        np.broadcast_to(mu[:, None], (len(mu), len(phi))).ravel(),
    ))
    radial_factor = 1.0 + coeficiente * np.sin(tt) * np.cos(pp - 5.0 * tt)
    radii = (scale * np.power(radial_factor, 0.25)).ravel()
    angular_weights = np.broadcast_to(
        mu_weights[:, None] * (2.0 * math.pi / len(phi)),
        (len(mu), len(phi)),
    ).ravel()
    axial_directions = directions @ axis_up

    layer_volumes = np.zeros(NUM_ESTRATOS)
    layer_first_moments = np.zeros((NUM_ESTRATOS, 3))
    layer_second_moments = np.zeros((NUM_ESTRATOS, 3, 3))
    for layer in range(NUM_ESTRATOS):
        lower, upper = edges[layer:layer + 2]
        r_min = np.zeros_like(radii)
        r_max = radii.copy()
        positive = axial_directions > 1e-12
        negative = axial_directions < -1e-12
        r_min[positive] = np.maximum(
            0.0, lower / axial_directions[positive])
        r_max[positive] = np.minimum(
            radii[positive], upper / axial_directions[positive])
        r_min[negative] = np.maximum(
            0.0, upper / axial_directions[negative])
        r_max[negative] = np.minimum(
            radii[negative], lower / axial_directions[negative])
        parallel = ~(positive | negative)
        if not lower <= 0.0 <= upper:
            r_max[parallel] = 0.0
        r_min = np.minimum(r_min, radii)
        r_max = np.maximum(r_max, r_min)
        valid = r_max > r_min
        r0, r1 = r_min[valid], r_max[valid]
        weights = angular_weights[valid]
        dirs = directions[valid]
        volume_integral = weights * (r1**3 - r0**3) / 3.0
        first_integral = weights * (r1**4 - r0**4) / 4.0
        second_integral = weights * (r1**5 - r0**5) / 5.0
        layer_volumes[layer] = volume_integral.sum()
        layer_first_moments[layer] = np.einsum(
            "i,ij->j", first_integral, dirs)
        layer_second_moments[layer] = np.einsum(
            "i,ij,ik->jk", second_integral, dirs, dirs)

    if np.any(layer_volumes <= 0.0):
        raise ValueError("La integración produjo un estrato sin volumen")
    density_top = total_mass / (
        layer_volumes.sum() + (RELACION_DENSIDAD_BASE - 1.0)
        * layer_volumes[0])
    layer_densities = np.full(NUM_ESTRATOS, density_top)
    layer_densities[0] *= RELACION_DENSIDAD_BASE
    layer_masses = layer_densities * layer_volumes
    if not math.isclose(float(layer_masses.sum()), total_mass, rel_tol=1e-10):
        raise RuntimeError("La masa estratificada no coincide con la masa objetivo")

    first_moment = np.einsum(
        "i,ij->j", layer_densities, layer_first_moments)
    cog = first_moment / total_mass
    second_moment = np.einsum(
        "i,ijk->jk", layer_densities, layer_second_moments)
    inertia = (
        np.trace(second_moment) * np.eye(3) - second_moment
        - total_mass * (
            (cog @ cog) * np.eye(3) - np.outer(cog, cog))
    )
    if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
        raise ValueError("La inercia estratificada no es definida positiva")
    layers = [
        {
            "layer": index + 1,
            "height_min": float(edges[index]),
            "height_max": float(edges[index + 1]),
            "volume": float(layer_volumes[index]),
            "density": float(layer_densities[index]),
            "mass": float(layer_masses[index]),
        }
        for index in range(NUM_ESTRATOS)
    ]
    return cog, inertia, total_mass / float(layer_volumes.sum()), layers


def construir_gomboc():
    """Construye la misma malla y obtiene centro de masa y densidades."""
    (coeficiente, scale, geometric_cog, local_vertices, faces, _mass,
     _density, _inertia) = crear_geometria(
         DIAMETRO, MASA_GOMBOC_KG, EPSILON)
    vertices = local_vertices + geometric_cog
    low, *_ = crear_puntos_caracteristicos(
        coeficiente, scale, geometric_cog, local_vertices)
    for _ in range(12):
        cog, inertia, density, layers = integrar_estratos(
            vertices, scale, coeficiente, -low, MASA_GOMBOC_KG)
        current_vertices = vertices - cog
        new_low, *_ = crear_puntos_caracteristicos(
            coeficiente, scale, cog, current_vertices, initial_low=low)
        if np.linalg.norm(new_low - low) < 1e-7:
            low = new_low
            break
        low = new_low
    else:
        raise RuntimeError("No convergió la orientación estable de los estratos")

    cog, inertia, density, layers = integrar_estratos(
        vertices, scale, coeficiente, -low, MASA_GOMBOC_KG)
    local_vertices = vertices - cog
    low, punto_estable, punto_opuesto, *_ = crear_puntos_caracteristicos(
        coeficiente, scale, cog, local_vertices, initial_low=low)
    return (coeficiente, scale, cog, local_vertices, faces, inertia, density,
            layers, punto_estable, punto_opuesto, -low)


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


def añadir_marcadores(chrono, body, puntos):
    """Añade números como malla, marcas negras y ejes al cuerpo."""
    malla_numeros = chrono.ChTriangleMeshConnected()
    for punto in puntos:
        normal = np.array([punto["nx"], punto["ny"], punto["nz"]])
        referencia = np.array([0.0, 0.0, 1.0])
        if abs(float(normal @ referencia)) > 0.95:
            referencia = np.array([0.0, 1.0, 0.0])
        eje_u = np.cross(referencia, normal)
        eje_u /= np.linalg.norm(eje_u)
        eje_v = np.cross(normal, eje_u)
        centro = punto["local_visual"]
        celda = TAMANO_NUMERO / 5.0
        caracteres = str(punto["punto"])
        ancho_celdas = len(caracteres) * 3 + len(caracteres) - 1
        desplazamiento_x = -0.5 * ancho_celdas * celda
        desplazamiento_y = -2.5 * celda

        for indice, caracter in enumerate(caracteres):
            patron = _GLIFOS[caracter]
            origen_x = desplazamiento_x + indice * 4 * celda
            for fila, bits in enumerate(patron):
                for columna, bit in enumerate(bits):
                    if bit != "1":
                        continue
                    x0 = origen_x + columna * celda
                    x1 = x0 + celda * 0.88
                    y1 = desplazamiento_y + (2.5 - fila) * celda
                    y0 = y1 - celda * 0.88
                    vertices = [
                        centro + eje_u * x0 + eje_v * y0,
                        centro + eje_u * x1 + eje_v * y0,
                        centro + eje_u * x1 + eje_v * y1,
                        centro + eje_u * x0 + eje_v * y1,
                    ]
                    malla_numeros.AddTriangle(
                        chrono.ChVector3d(*vertices[0]),
                        chrono.ChVector3d(*vertices[1]),
                        chrono.ChVector3d(*vertices[2]))
                    malla_numeros.AddTriangle(
                        chrono.ChVector3d(*vertices[0]),
                        chrono.ChVector3d(*vertices[2]),
                        chrono.ChVector3d(*vertices[3]))

        if MOSTRAR_MARCA_PUNTO:
            marca = chrono.ChVisualShapeSphere(RADIO_MARCA)
            marca.SetColor(chrono.ChColor(0.03, 0.03, 0.03))
            posicion = punto["local_superficie"]
            body.AddVisualShape(
                marca, chrono.ChFramed(chrono.ChVector3d(*posicion)))

    forma_numeros = chrono.ChVisualShapeTriangleMesh()
    forma_numeros.SetMesh(malla_numeros)
    forma_numeros.SetColor(chrono.ChColor(*COLOR_NUMERO))
    forma_numeros.GetMaterial(0).SetEmissiveColor(
        chrono.ChColor(*COLOR_EMISIVO_NUMERO))
    body.AddVisualShape(forma_numeros)

    longitud = 0.25 * DIAMETRO
    radio_eje = 0.003 * DIAMETRO
    ejes = (
        (np.array([1.0, 0.0, 0.0]), chrono.ChColor(1.0, 0.0, 0.0)),
        (np.array([0.0, 1.0, 0.0]), chrono.ChColor(0.0, 1.0, 0.0)),
        (np.array([0.0, 0.0, 1.0]), chrono.ChColor(0.0, 0.0, 1.0)),
    )
    eje_local = np.array([0.0, 1.0, 0.0])
    for direccion, color in ejes:
        segmento = chrono.ChVisualShapeCylinder(radio_eje, longitud)
        segmento.SetColor(color)
        rotacion = Rotation.align_vectors(
            direccion.reshape(1, 3), eje_local.reshape(1, 3))[0]
        qx, qy, qz, qw = rotacion.as_quat()
        transformacion = chrono.ChFramed(
            chrono.ChVector3d(*(0.5 * longitud * direccion)),
            chrono.ChQuaterniond(qw, qx, qy, qz))
        body.AddVisualShape(segmento, transformacion)


class Lienzo:
    """Lienzo RGB mínimo para crear el mapa PNG sin paquetes adicionales."""

    def __init__(self, ancho, alto, color=(255, 255, 255)):
        self.ancho = ancho
        self.alto = alto
        self.pixeles = bytearray(color * (ancho * alto))

    def punto(self, x, y, color):
        if 0 <= x < self.ancho and 0 <= y < self.alto:
            inicio = 3 * (y * self.ancho + x)
            self.pixeles[inicio:inicio + 3] = bytes(color)

    def rectangulo(self, x0, y0, x1, y1, color, relleno=True):
        if relleno:
            x0, x1 = sorted((max(0, x0), min(self.ancho - 1, x1)))
            y0, y1 = sorted((max(0, y0), min(self.alto - 1, y1)))
            fila = bytes(color) * max(0, x1 - x0 + 1)
            for y in range(y0, y1 + 1):
                inicio = 3 * (y * self.ancho + x0)
                self.pixeles[inicio:inicio + len(fila)] = fila
        else:
            self.linea(x0, y0, x1, y0, color)
            self.linea(x1, y0, x1, y1, color)
            self.linea(x1, y1, x0, y1, color)
            self.linea(x0, y1, x0, y0, color)

    def linea(self, x0, y0, x1, y1, color):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx - dy
        while True:
            self.punto(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            doble = 2 * error
            if doble > -dy:
                error -= dy
                x0 += sx
            if doble < dx:
                error += dx
                y0 += sy

    def circulo(self, cx, cy, radio, color):
        for y in range(-radio, radio + 1):
            ancho = int(math.sqrt(max(0, radio * radio - y * y)))
            for x in range(-ancho, ancho + 1):
                self.punto(cx + x, cy + y, color)

    def poligono(self, vertices, color):
        """Rellena un polígono convexo mediante barrido horizontal."""
        y_min = max(0, int(math.floor(min(p[1] for p in vertices))))
        y_max = min(self.alto - 1, int(math.ceil(max(p[1] for p in vertices))))
        for y in range(y_min, y_max + 1):
            intersecciones = []
            for inicio, fin in zip(vertices, vertices[1:] + vertices[:1]):
                x0, y0 = inicio
                x1, y1 = fin
                if (y0 <= y < y1) or (y1 <= y < y0):
                    intersecciones.append(
                        x0 + (y - y0) * (x1 - x0) / (y1 - y0))
            if len(intersecciones) >= 2:
                x0 = max(0, int(math.floor(min(intersecciones))))
                x1 = min(self.ancho - 1, int(math.ceil(max(intersecciones))))
                self.rectangulo(x0, y, x1, y, color)

    def texto(self, x, y, texto, color, escala=5):
        cursor = x
        for caracter in texto.upper():
            glifo = _GLIFOS.get(caracter, _GLIFOS[" "])
            for fila, bits in enumerate(glifo):
                for columna, bit in enumerate(bits):
                    if bit == "1":
                        self.rectangulo(
                            cursor + columna * escala,
                            y + fila * escala,
                            cursor + (columna + 1) * escala - 1,
                            y + (fila + 1) * escala - 1,
                            color)
            cursor += 4 * escala

    def guardar_png(self, ruta):
        def fragmento(tipo, datos):
            contenido = tipo + datos
            return (struct.pack(">I", len(datos)) + contenido
                    + struct.pack(">I", zlib.crc32(contenido) & 0xffffffff))

        filas = bytearray()
        ancho_fila = self.ancho * 3
        for y in range(self.alto):
            filas.append(0)
            inicio = y * ancho_fila
            filas.extend(self.pixeles[inicio:inicio + ancho_fila])
        cabecera = struct.pack(
            ">IIBBBBB", self.ancho, self.alto, 8, 2, 0, 0, 0)
        with ruta.open("wb") as archivo:
            archivo.write(b"\x89PNG\r\n\x1a\n")
            archivo.write(fragmento(b"IHDR", cabecera))
            archivo.write(fragmento(b"IDAT", zlib.compress(filas, 9)))
            archivo.write(fragmento(b"IEND", b""))


def crear_mapa_puntos(vertices, puntos, rotacion_body, punto_estable,
                      punto_opuesto, direccion_estable):
    """Dibuja vistas ortográficas frontal, trasera, superior e inferior."""
    ancho, alto = 1800, 1320
    lienzo = Lienzo(ancho, alto, (250, 250, 250))
    margen_x, margen_y = 35, 40
    panel_ancho, panel_alto = 850, 605
    vistas = (
        ("FRENTE (+Y)", np.array([0.0, 1.0, 0.0]),
         np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        ("TRASERA (-Y)", np.array([0.0, -1.0, 0.0]),
         np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        ("SUPERIOR (+Z)", np.array([0.0, 0.0, 1.0]),
         np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        ("INFERIOR (-Z)", np.array([0.0, 0.0, -1.0]),
         np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    )
    vertices_rotados = rotacion_body.apply(vertices)
    puntos_rotados = [
        (rotacion_body.apply(punto["local_superficie"]),
         rotacion_body.apply(punto["local_visual"]),
         rotacion_body.apply(np.array([punto["nx"], punto["ny"], punto["nz"]])))
        for punto in puntos
    ]
    estable_rotado = rotacion_body.apply(punto_estable)
    opuesto_rotado = rotacion_body.apply(punto_opuesto)
    direccion_rotada = rotacion_body.apply(direccion_estable)
    azul = (91, 145, 198)
    contorno = (35, 65, 95)
    rojo = tuple(round(255 * min(1.0, max(0.0, componente)))
                 for componente in COLOR_NUMERO)

    for indice, (titulo, mirada, eje_x, eje_y) in enumerate(vistas):
        col, fila = indice % 2, indice // 2
        px = margen_x + col * (panel_ancho + 30)
        py = margen_y + fila * (panel_alto + 30)
        lienzo.rectangulo(
            px, py, px + panel_ancho, py + panel_alto,
            (255, 255, 255))
        lienzo.rectangulo(
            px, py, px + panel_ancho, py + panel_alto,
            (190, 195, 202), relleno=False)
        lienzo.texto(px + 22, py + 18, titulo, (30, 35, 42), escala=5)

        proyectados = np.column_stack((
            vertices_rotados @ eje_x, vertices_rotados @ eje_y))
        contorno_2d = proyectados[ConvexHull(proyectados).vertices]
        escala = min(
            (panel_ancho - 150) / np.ptp(contorno_2d[:, 0]),
            (panel_alto - 140) / np.ptp(contorno_2d[:, 1]))
        centro_x = px + panel_ancho / 2
        centro_y = py + panel_alto / 2 + 22

        def a_pixel(coordenadas):
            return [
                (int(round(centro_x + coordenada[0] * escala)),
                 int(round(centro_y - coordenada[1] * escala)))
                for coordenada in coordenadas
            ]

        poligono = a_pixel(contorno_2d)
        lienzo.poligono(poligono, azul)
        for inicio, fin in zip(poligono, poligono[1:] + poligono[:1]):
            lienzo.linea(*inicio, *fin, contorno)

        for punto, (superficie, etiqueta, normal) in enumerate(puntos_rotados):
            if normal @ mirada <= 0.0:
                continue
            posicion = np.array([superficie @ eje_x, superficie @ eje_y])
            posicion_etiqueta = np.array([etiqueta @ eje_x, etiqueta @ eje_y])
            marca_x, marca_y = a_pixel([posicion])[0]
            numero_x, numero_y = a_pixel([posicion_etiqueta])[0]
            lienzo.circulo(marca_x, marca_y, 4, (15, 15, 15))
            lienzo.texto(numero_x + 5, numero_y - 9, str(punto + 1),
                         rojo, escala=5)

        for posicion, normal, color in (
                (estable_rotado, direccion_rotada, (230, 20, 25)),
                (opuesto_rotado, -direccion_rotada, (20, 70, 235))):
            if normal @ mirada > 0.0:
                px_marcador, py_marcador = a_pixel([[
                    posicion @ eje_x, posicion @ eje_y]])[0]
                lienzo.circulo(px_marcador, py_marcador, 11, (255, 255, 255))
                lienzo.circulo(px_marcador, py_marcador, 8, color)

    lienzo.circulo(76, alto - 24, 9, (230, 20, 25))
    lienzo.texto(94, alto - 36, "ESTABLE", (35, 35, 40), escala=4)
    lienzo.circulo(310, alto - 24, 9, (20, 70, 235))
    lienzo.texto(328, alto - 36, "OPUESTO", (35, 35, 40), escala=4)

    ruta = Path(__file__).with_name("mapa_puntos.png")
    lienzo.guardar_png(ruta)
    print("Mapa de cuatro vistas guardado en: %s" % ruta)


def guardar_puntos(puntos):
    """Imprime y guarda las coordenadas solicitadas."""
    columnas = (
        "punto", "theta_deg", "phi_deg", "nx", "ny", "nz", "r",
        "x", "y", "z",
    )
    ruta = Path(__file__).with_name("puntos.csv")
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        for punto in puntos:
            escritor.writerow({columna: punto[columna] for columna in columnas})

    print("punto, theta_deg, phi_deg, nx, ny, nz, r, x, y, z")
    for punto in puntos:
        print("%d, %.6f, %.6f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f"
              % tuple(punto[columna] for columna in columnas))
    print("CSV guardado en: %s" % ruta)


def main():
    """Prepara el cuerpo fijo, el mapa y la ventana Irrlicht."""
    (coeficiente, scale, cog, vertices, faces, inertia, density, layers,
     punto_estable, punto_opuesto, direccion_estable) = construir_gomboc()
    puntos = generar_puntos(coeficiente, scale, cog)
    guardar_puntos(puntos)
    if PUNTO_ABAJO is not None:
        if (not isinstance(PUNTO_ABAJO, int)
                or not 1 <= PUNTO_ABAJO <= N_PUNTOS):
            raise ValueError(
                "PUNTO_ABAJO debe ser None o un entero entre 1 y N_PUNTOS")
        normal = np.array([
            puntos[PUNTO_ABAJO - 1]["nx"],
            puntos[PUNTO_ABAJO - 1]["ny"],
            puntos[PUNTO_ABAJO - 1]["nz"],
        ])
        rotacion_body = Rotation.align_vectors(
            np.array([[0.0, 0.0, -1.0]]), normal.reshape(1, 3))[0]
    else:
        rotacion_body = Rotation.identity()
    crear_mapa_puntos(
        vertices, puntos, rotacion_body, punto_estable, punto_opuesto,
        direccion_estable)

    import pychrono as chrono
    import pychrono.irrlicht as chronoirr

    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, 0))
    body = chrono.ChBody()
    body.SetName(" ")
    body.SetMass(MASA_GOMBOC_KG)
    body.SetInertiaXX(chrono.ChVector3d(
        inertia[0, 0], inertia[1, 1], inertia[2, 2]))
    body.SetInertiaXY(chrono.ChVector3d(
        inertia[0, 1], inertia[0, 2], inertia[1, 2]))
    body.SetFixed(True)
    body.EnableCollision(False)

    body_mesh = chrono.ChTriangleMeshConnected()
    for face in faces:
        body_mesh.AddTriangle(
            chrono.ChVector3d(*vertices[face[0]]),
            chrono.ChVector3d(*vertices[face[1]]),
            chrono.ChVector3d(*vertices[face[2]]))
    body_shape = chrono.ChVisualShapeTriangleMesh()
    body_shape.SetMesh(body_mesh)
    body_shape.SetColor(chrono.ChColor(0.20, 0.46, 0.82))
    body_shape.GetMaterial(0).SetAmbientColor(chrono.ChColor(0.48, 0.48, 0.48))
    body_shape.GetMaterial(0).SetEmissiveColor(
        chrono.ChColor(*COLOR_EMISIVO_GOMBOC))
    body.AddVisualShape(body_shape)
    for punto, color in (
            (punto_estable, chrono.ChColor(0.95, 0.04, 0.03)),
            (punto_opuesto, chrono.ChColor(0.04, 0.12, 0.98))):
        marcador = chrono.ChVisualShapeSphere(RADIO_MARCADOR_EQUILIBRIO)
        marcador.SetColor(color)
        marcador.GetMaterial(0).SetEmissiveColor(color)
        body.AddVisualShape(
            marcador, chrono.ChFramed(chrono.ChVector3d(*punto)))
    quaternion = rotacion_body.as_quat()
    qx, qy, qz, qw = quaternion
    body.SetRot(chrono.ChQuaterniond(qw, qx, qy, qz))
    system.Add(body)
    añadir_marcadores(chrono, body, puntos)

    visual = chronoirr.ChVisualSystemIrrlicht()
    visual.AttachSystem(system)
    visual.SetWindowTitle("Gömböc estático — números Fibonacci")
    visual.SetWindowSize(1280, 800)
    visual.Initialize()
    # Ilumina desde todos los lados para que la base no dependa de la luz cenital.
    for posicion in (
            (0.0, 0.0, 0.28), (0.0, 0.0, -0.28),
            (0.28, 0.0, 0.0), (-0.28, 0.0, 0.0),
            (0.0, 0.28, 0.0), (0.0, -0.28, 0.0)):
        visual.AddLight(
            chrono.ChVector3d(*posicion), 1.0,
            chrono.ChColor(0.85, 0.85, 0.85))
    visual.AddCamera(
        chrono.ChVector3d(0.65, 0.50, -0.85),
        chrono.ChVector3d(0.0, 0.0, 0.0))

    density_base = layers[0]["density"]
    density_top = layers[1]["density"]
    print("Masa total: %.6f kg" % MASA_GOMBOC_KG)
    print("Densidad superior: %.9g kg/m³" % density_top)
    print("Densidad de la base: %.9g kg/m³" % density_base)
    print("Diámetro máximo: %.6f m" % DIAMETRO)
    print("Etiquetas 3D solidarias al cuerpo, orientadas hacia fuera.")
    print("Control de cámara Irrlicht: arrastra con el ratón para orbitar.")
    while visual.Run():
        visual.BeginScene()
        visual.Render()
        visual.EndScene()


if __name__ == "__main__":
    main()
