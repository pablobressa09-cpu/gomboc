"""Simula el enderezamiento del Gömböc sobre un plano sólido."""

import os
import math
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize

chrono_bin = next(
    (Path(entry) for entry in sys.path
     if entry and (Path(entry) / "pychrono" / "__init__.py").is_file()),
    None,
)
if chrono_bin and (chrono_bin.parent / "data" / "colormaps"
                   / "jet-table-float-0512.csv").is_file():
    os.chdir(chrono_bin)

import pychrono as chrono

chrono_data_dir = Path(os.environ.get(
    "CHRONO_DATA_DIR", Path(chrono.__file__).resolve().parents[2] / "data"))
if (chrono_data_dir / "colormaps" / "jet-table-float-0512.csv").is_file():
    chrono.SetChronoDataPath(str(chrono_data_dir) + os.sep)

import pychrono.irrlicht as chronoirr

TAMANO = 0.20
DENSIDAD = 1200.0
GRAVEDAD = 3.721
ZETA = 0.07
EPSILON = 0.03
INCLINACION_INICIAL = 60.0
N_THETA = 121
N_PHI = 240
PUNTOS_ALTURA = 8000
DT = 8.0e-4
PASOS_POR_FOTOGRAMA = 60
ACTUALIZAR_CONTACTO_CADA = 2

RIGIDEZ_NORMAL = 2.0e5
FRICCION = 0.6
AMORTIGUAMIENTO_TANGENCIAL = 150.0
ALTURA_SUELO = 0.0

chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(2.0e-4)
chrono.ChCollisionModel.SetDefaultSuggestedMargin(1.0e-4)
chrono.ChCollisionSystemBullet.SetContactBreakingThreshold(1.0e-4)

# Coordenadas esféricas: theta desde +Z y phi desde +X hacia +Y.
theta = np.linspace(0.0, math.pi, N_THETA + 1)
phi = 2.0 * math.pi * np.arange(N_PHI) / N_PHI
theta_rings = theta[1:-1, None]
phi_grid = phi[None, :]
coeficiente = 4.0 * EPSILON
if abs(EPSILON) > 0.032:
    raise ValueError(
        "El valor de epsilon excede el limite de convexidad usado por esta "
        "simulacion: |epsilon| debe ser <= 0.032 (valor actual %.3f)."
        % EPSILON
    )
radicando = (1.0 + coeficiente * np.sin(theta_rings)
             * np.cos(phi_grid - 5.0 * theta_rings))
radio = np.power(radicando, 0.25)
anillo_x = radio * np.sin(theta_rings) * np.cos(phi_grid)
anillo_y = radio * np.sin(theta_rings) * np.sin(phi_grid)
anillo_z = radio * np.cos(theta_rings)
anillos = np.stack((anillo_x, anillo_y, anillo_z), axis=2).reshape(-1, 3)
V = np.vstack((np.array([[0.0, 0.0, 1.0]]), anillos,
               np.array([[0.0, 0.0, -1.0]])))

indices_phi = np.arange(N_PHI)
indices_phi_sig = (indices_phi + 1) % N_PHI
anillo_idx = (1 + np.arange(N_THETA - 1)[:, None] * N_PHI
              + indices_phi[None, :])
caras = [
    np.column_stack((
        np.zeros(N_PHI, dtype=np.int64),
        anillo_idx[0, indices_phi_sig],
        anillo_idx[0, indices_phi],
    ))
]
for j in range(N_THETA - 2):
    superior = anillo_idx[j]
    inferior = anillo_idx[j + 1]
    caras.extend((
        np.column_stack((superior, superior[indices_phi_sig],
                         inferior[indices_phi_sig])),
        np.column_stack((superior, inferior[indices_phi_sig], inferior)),
    ))
indice_sur = len(V) - 1
caras.append(np.column_stack((
    np.full(N_PHI, indice_sur, dtype=np.int64),
    anillo_idx[-1, indices_phi],
    anillo_idx[-1, indices_phi_sig],
)))
F = np.vstack(caras)

escala = TAMANO / (V.max(0) - V.min(0)).max()
V *= escala

a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
volumen_firmado = np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0
if volumen_firmado < 0.0:
    F[:, [1, 2]] = F[:, [2, 1]]
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    volumen_firmado = -volumen_firmado
if not np.isfinite(volumen_firmado) or volumen_firmado <= 0.0:
    raise ValueError("La malla generada no encierra un volumen positivo")

vt = np.einsum('ij,ij->i', a, np.cross(b, c)) / 6.0
com = (vt[:, None] * ((a + b + c) / 4.0)).sum(0) / volumen_firmado
sm = a + b + c
w = vt / 20.0
C = (np.einsum('i,ij,ik->jk', w, a, a) + np.einsum('i,ij,ik->jk', w, b, b)
     + np.einsum('i,ij,ik->jk', w, c, c) + np.einsum('i,ij,ik->jk', w, sm, sm))
MASA = DENSIDAD * volumen_firmado
AMORTIGUAMIENTO_NORMAL = 2.0 * math.sqrt(RIGIDEZ_NORMAL * MASA)
INER = DENSIDAD * (np.trace(C) * np.eye(3) - C
                   - volumen_firmado * ((com @ com) * np.eye(3)
                                        - np.outer(com, com)))
Vc = V - com

i = np.arange(PUNTOS_ALTURA)
z = 1.0 - 2.0 * (i + 0.5) / PUNTOS_ALTURA
rr = np.sqrt(np.maximum(0.0, 1.0 - z * z))
th = 2.0 * np.pi * i / ((1.0 + 5.0 ** 0.5) / 2.0)
DIR = np.stack((rr * np.cos(th), z, rr * np.sin(th)), axis=1)
alt = np.empty(PUNTOS_ALTURA)
bl = max(1, int(48e6 / (8 * len(Vc))))
for k in range(0, PUNTOS_ALTURA, bl):
    alt[k:k + bl] = (Vc @ DIR[k:k + bl].T).max(axis=0)
u_bajo = DIR[int(np.argmin(alt))]
u_alto = DIR[int(np.argmax(alt))]
R_ROD = float(alt.mean())
I_CONT = float(np.trace(INER)) / 3.0 + MASA * R_ROD ** 2
PALANCA = (alt.max() - alt.min()) / (math.pi / 2.0)
OMEGA = math.sqrt(MASA * GRAVEDAD * PALANCA / I_CONT)
TAU = 1.0 / (2.0 * ZETA * OMEGA)

sistema = chrono.ChSystemNSC()
sistema.SetGravitationalAcceleration(chrono.ChVector3d(0, -GRAVEDAD, 0))
sistema.SetSleepingAllowed(False)

material = chrono.ChContactMaterialNSC()
material.SetFriction(0.6)
material.SetRestitution(0.0)

suelo = chrono.ChBodyEasyBox(2.0, 0.2, 2.0, 2000, True, False, material)
suelo.SetPos(chrono.ChVector3d(0, -0.1, 0))
suelo.SetFixed(True)
suelo.GetVisualShape(0).SetColor(chrono.ChColor(0.50, 0.36, 0.26))
sistema.Add(suelo)

malla_vis = chrono.ChTriangleMeshConnected()
for tri in F:
    p0, p1, p2 = Vc[tri[0]], Vc[tri[1]], Vc[tri[2]]
    malla_vis.AddTriangle(chrono.ChVector3d(*p0), chrono.ChVector3d(*p1),
                          chrono.ChVector3d(*p2))

gomboc = chrono.ChBodyAuxRef()
gomboc.SetMass(MASA)
gomboc.SetInertiaXX(chrono.ChVector3d(INER[0, 0], INER[1, 1], INER[2, 2]))
gomboc.SetInertiaXY(chrono.ChVector3d(INER[0, 1], INER[0, 2], INER[1, 2]))
gomboc.EnableCollision(False)

forma = chrono.ChVisualShapeTriangleMesh()
forma.SetMesh(malla_vis)
forma.SetColor(chrono.ChColor(0.20, 0.45, 0.85))
gomboc.AddVisualShape(forma)

marca = chrono.ChVisualShapeSphere(0.02 * TAMANO)
marca.SetColor(chrono.ChColor(0.9, 0.1, 0.1))
marca_azul = chrono.ChVisualShapeSphere(0.02 * TAMANO)
marca_azul.SetColor(chrono.ChColor(0.1, 0.3, 0.95))

def punto_superficie(theta_val, phi_val):
    delta = phi_val - 5.0 * theta_val
    radicando_val = 1.0 + coeficiente * math.sin(theta_val) * math.cos(delta)
    radio_val = radicando_val ** 0.25
    return escala * radio_val * np.array([
        math.sin(theta_val) * math.cos(phi_val),
        math.sin(theta_val) * math.sin(phi_val),
        math.cos(theta_val),
    ]) - com


def encontrar_soporte(direccion_local, inicio):
    def valor_y_gradiente(x):
        theta_val, phi_val = x
        sin_theta = math.sin(theta_val)
        cos_theta = math.cos(theta_val)
        sin_delta = math.sin(phi_val - 5.0 * theta_val)
        cos_delta = math.cos(phi_val - 5.0 * theta_val)
        radicando_val = 1.0 + coeficiente * sin_theta * cos_delta
        radio_val = radicando_val ** 0.25
        radio_factor = 0.25 * radicando_val ** -0.75
        radio_theta = radio_factor * coeficiente * (
            cos_theta * cos_delta + 5.0 * sin_theta * sin_delta)
        radio_phi = -radio_factor * coeficiente * sin_theta * sin_delta
        unidad = np.array([
            sin_theta * math.cos(phi_val),
            sin_theta * math.sin(phi_val),
            cos_theta,
        ])
        unidad_theta = np.array([
            cos_theta * math.cos(phi_val),
            cos_theta * math.sin(phi_val),
            -sin_theta,
        ])
        unidad_phi = np.array([
            -sin_theta * math.sin(phi_val),
            sin_theta * math.cos(phi_val),
            0.0,
        ])
        punto = escala * radio_val * unidad - com
        derivada_theta = escala * (radio_theta * unidad + radio_val * unidad_theta)
        derivada_phi = escala * (radio_phi * unidad + radio_val * unidad_phi)
        return float(direccion_local @ punto), np.array([
            direccion_local @ derivada_theta,
            direccion_local @ derivada_phi,
        ])

    solucion = minimize(
        valor_y_gradiente,
        inicio,
        method="L-BFGS-B",
        jac=True,
        bounds=((1e-9, math.pi - 1e-9), (None, None)),
        options={"ftol": 1e-16, "gtol": 1e-15, "maxiter": 80},
    )
    if not solucion.success and np.linalg.norm(solucion.jac) > 1e-6:
        raise RuntimeError("No convergio la busqueda del punto de contacto: %s"
                           % solucion.message)
    return solucion.x, punto_superficie(*solucion.x)


def direccion_esferica(theta_val, phi_val):
    return np.array([
        math.sin(theta_val) * math.cos(phi_val),
        math.sin(theta_val) * math.sin(phi_val),
        math.cos(theta_val),
    ])


def altura_y_gradiente_esfericos(angulos):
    theta_val, phi_val = angulos
    direccion = direccion_esferica(theta_val, phi_val)
    inicio_superficie = np.array([theta_val, phi_val])
    _, punto = encontrar_soporte(-direccion, inicio_superficie)
    derivada_theta = np.array([
        math.cos(theta_val) * math.cos(phi_val),
        math.cos(theta_val) * math.sin(phi_val),
        -math.sin(theta_val),
    ])
    derivada_phi = np.array([
        -math.sin(theta_val) * math.sin(phi_val),
        math.sin(theta_val) * math.cos(phi_val),
        0.0,
    ])
    return float(direccion @ punto), np.array([
        punto @ derivada_theta,
        punto @ derivada_phi,
    ])


inicio_estable = np.array([
    math.acos(float(np.clip(u_bajo[2], -1.0, 1.0))),
    math.atan2(u_bajo[1], u_bajo[0]),
])
solucion_estable = minimize(
    altura_y_gradiente_esfericos,
    inicio_estable,
    method="L-BFGS-B",
    jac=True,
    bounds=((1e-9, math.pi - 1e-9), (None, None)),
    options={"ftol": 1e-14, "gtol": 1e-11, "maxiter": 60},
)
if not solucion_estable.success and np.linalg.norm(solucion_estable.jac) > 1e-7:
    raise RuntimeError("No convergio la busqueda del equilibrio estable: %s"
                       % solucion_estable.message)

theta_estable, phi_estable = solucion_estable.x
u_bajo = direccion_esferica(theta_estable, phi_estable)
altura_equilibrio = float(solucion_estable.fun)
angulos_contacto = np.array([
    math.acos(float(np.clip(u_bajo[2], -1.0, 1.0))),
    math.atan2(u_bajo[1], u_bajo[0]),
])
_, punto_equilibrio_local = encontrar_soporte(-u_bajo, angulos_contacto)
angulos_extremo_opuesto = np.array([
    math.pi - angulos_contacto[0],
    angulos_contacto[1] + math.pi,
])
_, punto_extremo_opuesto_local = encontrar_soporte(
    u_bajo, angulos_extremo_opuesto)

rot = chrono.ChQuaterniond()
c0 = float(np.clip(u_bajo @ np.array([0.0, -1.0, 0.0]), -1.0, 1.0))
perp = np.cross(u_bajo, np.array([0.0, -1.0, 0.0]))
if np.linalg.norm(perp) < 1e-9:
    rot.SetFromAngleAxis(0.0 if c0 > 0 else math.pi,
                         chrono.ChVector3d(1, 0, 0))
else:
    perp /= np.linalg.norm(perp)
    rot.SetFromAngleAxis(math.acos(c0), chrono.ChVector3d(*perp))

inclinacion_inicial = chrono.ChQuaterniond()
inclinacion_inicial.SetFromAngleAxis(
    math.radians(INCLINACION_INICIAL), chrono.ChVector3d(1, 0, 0))
rot = inclinacion_inicial * rot

direccion_contacto_inicial = np.array([
    rot.GetAxisX().y, rot.GetAxisY().y, rot.GetAxisZ().y,
])
direccion_superficie_inicial = -direccion_contacto_inicial
angulos_contacto = np.array([
    math.acos(float(np.clip(direccion_superficie_inicial[2], -1.0, 1.0))),
    math.atan2(direccion_superficie_inicial[1],
               direccion_superficie_inicial[0]),
])
_, soporte_local = encontrar_soporte(direccion_contacto_inicial,
                                     angulos_contacto)

gomboc.AddVisualShape(
    marca, chrono.ChFramed(chrono.ChVector3d(*punto_equilibrio_local), chrono.QUNIT))
gomboc.AddVisualShape(
    marca_azul,
    chrono.ChFramed(chrono.ChVector3d(*punto_extremo_opuesto_local), chrono.QUNIT))

# La altura inicial corresponde al soporte de la orientación inclinada.
penetracion_equilibrio = MASA * GRAVEDAD / RIGIDEZ_NORMAL
altura_com_inicial = -float(direccion_contacto_inicial @ soporte_local)
altura_com_inicial -= penetracion_equilibrio
gomboc.SetFrameRefToAbs(chrono.ChFramed(
    chrono.ChVector3d(0, altura_com_inicial, 0), rot))
sistema.Add(gomboc)

print("superficie           : r = (1 + 4 epsilon sin(theta) cos(phi - 5 theta))^(1/4)")
print("epsilon              : %.3f (4*epsilon = %.3f)" % (EPSILON, coeficiente))
print("malla visual         : %d vertices, %d triangulos" % (len(V), len(F)))
print("contacto             : soporte continuo de la ecuacion (optimizacion numerica)")
print("masa                 : %.3f kg" % MASA)
print("altura del COM       : %.2f mm en equilibrio, %.2f mm en el peor caso"
      % (1000 * altura_equilibrio, 1000 * alt.max()))
print("energia a liberar    : %.4f J"
      % (MASA * GRAVEDAD * (alt.max() - altura_equilibrio)))
print("periodo de balanceo  : %.2f s   (amortiguamiento tau = %.2f s, zeta = %.2f)"
      % (2 * math.pi / OMEGA, TAU, ZETA))
print("inclinacion inicial  : %.1f grados respecto al equilibrio estable"
      % INCLINACION_INICIAL)
print("punto rojo           : punto de apoyo estable de la ecuacion continua")
print("punto azul           : extremo opuesto; se mide su angulo con +Y mundial")

acc = gomboc.AddAccumulator()
c_ang = float(np.trace(INER)) / 3.0 / TAU
c_lin = MASA / TAU
soporte_local = soporte_local.copy()


def actualizar_contacto():
    global angulos_contacto, soporte_local
    q = gomboc.GetRot()
    direccion_local = np.array([q.GetAxisX().y, q.GetAxisY().y, q.GetAxisZ().y])
    angulos_contacto, soporte_local = encontrar_soporte(direccion_local,
                                                         angulos_contacto)


def aplicar_contacto():
    q = gomboc.GetRot()
    ejes = np.column_stack((
        np.array([q.GetAxisX().x, q.GetAxisX().y, q.GetAxisX().z]),
        np.array([q.GetAxisY().x, q.GetAxisY().y, q.GetAxisY().z]),
        np.array([q.GetAxisZ().x, q.GetAxisZ().y, q.GetAxisZ().z]),
    ))
    posicion_com = np.array([gomboc.GetPos().x, gomboc.GetPos().y, gomboc.GetPos().z])
    brazo = ejes @ soporte_local
    punto_mundo = posicion_com + brazo
    penetracion = ALTURA_SUELO - punto_mundo[1]
    if penetracion <= 0.0:
        return

    velocidad = np.array([gomboc.GetLinVel().x, gomboc.GetLinVel().y,
                          gomboc.GetLinVel().z])
    omega = np.array([gomboc.GetAngVelParent().x, gomboc.GetAngVelParent().y,
                      gomboc.GetAngVelParent().z])
    velocidad_contacto = velocidad + np.cross(omega, brazo)
    fuerza_normal = max(
        0.0, RIGIDEZ_NORMAL * penetracion
        - AMORTIGUAMIENTO_NORMAL * min(velocidad_contacto[1], 0.0))
    fuerza = np.array([0.0, fuerza_normal, 0.0])
    velocidad_tangencial = velocidad_contacto.copy()
    velocidad_tangencial[1] = 0.0
    rapidez_tangencial = np.linalg.norm(velocidad_tangencial)
    if rapidez_tangencial > 1e-12 and fuerza_normal > 0.0:
        modulo_friccion = min(FRICCION * fuerza_normal,
                              AMORTIGUAMIENTO_TANGENCIAL * rapidez_tangencial)
        fuerza -= modulo_friccion * velocidad_tangencial / rapidez_tangencial
    gomboc.AccumulateForce(acc, chrono.ChVector3d(*fuerza),
                           chrono.ChVector3d(*punto_mundo), False)


def angulo_punto_azul_con_eje_y():
    q = gomboc.GetRot()
    extremo_mundo = (
        np.array([q.GetAxisX().x, q.GetAxisX().y, q.GetAxisX().z])
        * punto_extremo_opuesto_local[0]
        + np.array([q.GetAxisY().x, q.GetAxisY().y, q.GetAxisY().z])
        * punto_extremo_opuesto_local[1]
        + np.array([q.GetAxisZ().x, q.GetAxisZ().y, q.GetAxisZ().z])
        * punto_extremo_opuesto_local[2]
    )
    modulo = np.linalg.norm(extremo_mundo)
    if modulo <= 1e-12:
        raise RuntimeError("No se puede medir el angulo: el punto azul coincide con el COM")
    coseno = float(np.clip(extremo_mundo[1] / modulo, -1.0, 1.0))
    return math.degrees(math.acos(coseno))


vis = chronoirr.ChVisualSystemIrrlicht()
vis.AttachSystem(sistema)
vis.SetWindowTitle("Gomboc - ecuacion radial")
vis.SetWindowSize(1280, 800)
vis.Initialize()
vis.AddTypicalLights()
vis.AddCamera(chrono.ChVector3d(0.20, 0.25, -0.35),
              chrono.ChVector3d(0, 0.10, 0))
# La cámara Irrlicht debe registrarse para recibir eventos del ratón.
vis.AddUserEventReceiver(vis.GetActiveCamera())
print("Controles de camara: arrastrar con boton izquierdo para orbitar,")
print("boton derecho para desplazar y rueda del raton para zoom.")
print("Cierre: boton X de la ventana.")

t_sig = 1.0
try:
    while vis.Run():
        vis.BeginScene()
        vis.Render()
        vis.EndScene()
        for paso in range(PASOS_POR_FOTOGRAMA):
            if paso % ACTUALIZAR_CONTACTO_CADA == 0:
                actualizar_contacto()
            gomboc.EmptyAccumulator(acc)
            gomboc.AccumulateTorque(
                acc, gomboc.GetAngVelParent() * (-c_ang), False)
            aplicar_contacto()
            gomboc.AccumulateForce(acc, gomboc.GetLinVel() * (-c_lin),
                                   gomboc.GetPos(), False)
            sistema.DoStepDynamics(DT)
        if sistema.GetChTime() >= t_sig:
            print("t = %5.2f s   altura COM = %6.2f mm  (equilibrio %.2f mm)"
                  "   |omega| = %.4f rad/s   punto azul con +Y = %.2f grados"
                  % (sistema.GetChTime(), 1000 * gomboc.GetPos().y,
                     1000 * altura_equilibrio,
                     gomboc.GetAngVelParent().Length(),
                     angulo_punto_azul_con_eje_y()))
            t_sig += 1.0
finally:
    vis.Quit()
