#!/usr/bin/env python3
"""Visualización de la malla radial del Gömböc, sin marcadores ni suelo."""

import pychrono as chrono
import pychrono.irrlicht as chronoirr

import math
import os

import numpy as np
from scipy.optimize import minimize

DIAMETRO = 0.20
MASA_GOMBOC_KG = 4.0
EPSILON = 0.03
RELACION_DENSIDAD_BASE = 75.0
NUM_ESTRATOS = 5

_MODO_PUNTOS = os.environ.get("GOMBOC_MODO_PUNTOS", "comparacion30")
if _MODO_PUNTOS not in ("comparacion30", "fase1"):
    raise ValueError(f"GOMBOC_MODO_PUNTOS no reconocido: {_MODO_PUNTOS}")


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


def main():
    """Muestra la malla original como un cuerpo fijo sin colisiones."""
    _, _, _, vertices, caras, _, _, _, _, _, _ = construir_gomboc()

    sistema = chrono.ChSystemNSC()
    sistema.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, 0))
    cuerpo = chrono.ChBody()
    cuerpo.SetFixed(True)
    cuerpo.EnableCollision(False)

    malla = chrono.ChTriangleMeshConnected()
    for cara in caras:
        malla.AddTriangle(
            chrono.ChVector3d(*vertices[cara[0]]),
            chrono.ChVector3d(*vertices[cara[1]]),
            chrono.ChVector3d(*vertices[cara[2]]))
    forma = chrono.ChVisualShapeTriangleMesh()
    forma.SetMesh(malla)
    forma.SetColor(chrono.ChColor(0.20, 0.46, 0.82))
    cuerpo.AddVisualShape(forma)
    sistema.Add(cuerpo)

    visual = chronoirr.ChVisualSystemIrrlicht()
    visual.AttachSystem(sistema)
    visual.SetWindowTitle("Gomboc - malla visual")
    visual.SetWindowSize(1280, 800)
    visual.Initialize()
    visual.AddTypicalLights()
    visual.AddCamera(
        chrono.ChVector3d(0.65, 0.50, -0.85),
        chrono.ChVector3d(0.0, 0.0, 0.0))

    print("Ratón: arrastrar para orbitar y rueda para acercar o alejar.")
    while visual.Run():
        visual.BeginScene()
        visual.Render()
        visual.EndScene()


if __name__ == "__main__":
    main()
