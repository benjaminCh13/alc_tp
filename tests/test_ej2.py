"""
Tests unitarios Ej 2 — obtenerQ3d y resolverDLT3D (issue #4).

Ejecutar desde la raíz del repo: pytest tests/test_ej2.py -v
np.linalg está permitido acá: los tests NO van al entregado y sirven de
referencia contra la implementación propia de alc.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import alc

ATOL = 1e-6
SEED = 2026


@pytest.fixture
def rng():
    yield np.random.default_rng(SEED)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _proyectar(M, objPts):
    """Proyecta puntos (n, 3) con la cámara M (3, 4) -> (n, 2)."""
    n = len(objPts)
    imgPts = np.zeros((n, 2))
    for i in range(n):
        P = np.append(objPts[i], 1.0)          # (U, V, W, 1) en cm
        p = np.dot(M, P)
        imgPts[i] = p[:2] / p[2]
    return imgPts


def _camara_sintetica():
    """
    Cámara plausible: K razonable, rotación real (ortogonal por
    construcción) y todos los puntos delante (w > 0 garantizado).
    """
    K = np.array([[800., 0., 320.],
                  [0., 800., 240.],
                  [0., 0., 1.]])
    c, s = np.cos(0.3), np.sin(0.3)
    R = np.array([[c, 0., -s],
                  [0., 1., 0.],
                  [s, 0., c]])                 # ortogonal
    t = np.array([10., -20., 100.])
    return np.hstack([K @ R, t.reshape(3, 1)]), K, R, t


def _puntos_general(rng, n=8):
    """n puntos 3D en posición general (no coplanares ni colineales)."""
    pts = rng.uniform(-5., 5., (n, 3))
    pts[:, 0] -= pts[:, 0].mean()              # centrados (Z>0 preservado)
    pts[:, 2] += 5.
    return pts


def _misma_m_a_escala(M1, M2, atol=1e-4):
    """True si M1 ~ M2 salvo factor real (comparación por norma, robusta
    a cualquier pivote cero)."""
    N1 = M1 / np.linalg.norm(M1)
    N2 = M2 / np.linalg.norm(M2)
    # salvo signo: la cámara es una dirección en R^12 a menos de escala
    return np.allclose(N1, N2, atol=atol) or np.allclose(N1, -N2, atol=atol)


# ======================================================================
# obtenerQ3d
# ======================================================================


# ======================================================================
class TestObtenerQ3d:

    def test_shape(self):
        rng = np.random.default_rng(SEED)
        img = rng.uniform(0., 500., (9, 2))
        obj = rng.uniform(0., 30., (9, 3))
        Q = alc.obtenerQ3d(img, obj)
        assert Q.shape == (18, 12)

    def test_filas_por_punto(self):
        """Estructura exacta de las 2 filas por punto (ec. 6)."""
        img = np.array([[10., 20.], [30., 40.]])
        obj = np.array([[1., 2., 3.], [4., 5., 6.]])
        Q = alc.obtenerQ3d(img, obj)
        assert np.allclose(Q[0], [1, 2, 3, 1, 0, 0, 0, 0, -10., -20., -30., -10.])
        assert np.allclose(Q[1], [0, 0, 0, 0, 1, 2, 3, 1, -20., -40., -60., -20.])

    def test_expansion_homogenea(self):
        """El 1 homogéneo entra tal cual y el bloque 3D no se reescala."""
        rng = np.random.default_rng(SEED)
        obj = np.array([[1e4, -2e3, 3e2], [0.1, 0.2, 0.3]])
        img = rng.uniform(100., 700., (2, 2))
        Q = alc.obtenerQ3d(img, obj)
        assert Q[0, 3] == 1.0 and Q[1, 7] == 1.0     # el 1 en su columna
        assert np.allclose(Q[0, :3], obj[0])          # bloque 3D intacto

    def test_mismatch_puntos(self):
        img = np.zeros((5, 2))
        obj = np.zeros((4, 3))
        with pytest.raises(AssertionError):
            alc.obtenerQ3d(img, obj)


# ======================================================================
# obtenerQ3d con cámara sintética (el test madre)
# ======================================================================
class TestObtenerQ3dConCamara:

    def test_orden_de_columnas(self):
        """
        Q·m0 = 0 con m0 = M apilada row-major: es el test que sincroniza
        el orden de columnas de obtenerQ3d con el reshape de resolverDLT3D.
        """
        rng = np.random.default_rng(SEED)
        M, *_ = _camara_sintetica()
        obj = _puntos_general(rng)
        img = _proyectar(M, obj)                   # sin ruido: datos exactos
        Q = alc.obtenerQ3d(img, obj)
        assert np.abs(np.dot(Q, M.reshape(-1))).max() < 1e-6


# ======================================================================
# resolverDLT3D
# ======================================================================
class TestResolverDLT3D:

    def _Q_exacto(self, rng, n=8):
        """Q libre de ruido: se proyecta con una cámara sintética conocida."""
        M, *_ = _camara_sintetica()
        obj = _puntos_general(rng, n)
        img = _proyectar(M, obj)
        return alc.obtenerQ3d(img, obj), M, obj, img

    def test_devuelve_3x4(self, rng):
        Q, *_ = self._Q_exacto(rng)
        assert alc.resolverDLT3D(Q).shape == (3, 4)

    def test_recupera_camara_exacta(self, rng):
        """Datos sin ruido: M a menos de escala respecto de la sintética."""
        Q, M, *_ = self._Q_exacto(rng)
        M_rec = alc.resolverDLT3D(Q)
        # primer coeficiente no nulo fijo: elimina la ambigüedad de escala
        assert np.allclose(M / M[0, 0], M_rec / M_rec[0, 0], atol=1e-4)

    def test_nmin_seis_puntos(self, rng):
        """n = 6 (el mínimo del Ej 1a): el núcleo exacto dim 1 igual recuper."""
        Q, M, *_ = self._Q_exacto(rng, n=6)
        M_rec = alc.resolverDLT3D(Q)
        assert np.allclose(M / M[0, 0], M_rec / M_rec[0, 0], atol=1e-4)

    def test_rechaza_cinco_puntos(self, rng):
        """n = 5 < 6: teorema del rango del Ej 1a — el rango de Q no puede
        llegar a 11, el guard debe rechazar la entrada."""
        Q, *_ = self._Q_exacto(rng, n=5)
        with pytest.raises(AssertionError):
            alc.resolverDLT3D(Q)

    def test_respeta_norma_uno(self, rng):
        """Ej 1b: la solución min ||Qm||² sujeto a ||m|| = 1 (la norma la
        impone V, ortogonal, sin normalizar a mano)."""
        Q, *_ = self._Q_exacto(rng)
        m = alc.resolverDLT3D(Q).reshape(-1)
        assert abs(np.linalg.norm(m) - 1.0) < 1e-6

    def test_signo_canonico(self, rng):
        """Convención de salida determinista: m34 >= 0."""
        Q, *_ = self._Q_exacto(rng)
        assert alc.resolverDLT3D(Q)[2, 3] >= 0

    def test_shape_asserts(self, rng):
        with pytest.raises(AssertionError):
            alc.resolverDLT3D(np.zeros((24, 9)))   # columnas != 12
        Q, *_ = self._Q_exacto(rng)
        with pytest.raises(AssertionError):
            alc.resolverDLT3D(Q[:11])              # < 12 filas (n < 6)


# ======================================================================
# integrado: pipeline completo con ruido
# ======================================================================
class TestDLTIntegrado:

    def test_robustez_a_ruido(self):
        """
        Ruido de detección de ~0.5 px (imgPts enteros a .5, lo mismo que
        da findCorners en calib/3d): el error de reproyección con la M
        recuperada queda acotado a O(px).
        """
        rng = np.random.default_rng(SEED)
        sigma_px = 0.5
        M, *_ = _camara_sintetica()
        obj = _puntos_general(rng)
        img = _proyectar(M, obj) + rng.normal(0., sigma_px, (len(obj), 2))
        Q = alc.obtenerQ3d(img, obj)
        M_rec = alc.resolverDLT3D(Q)

        err = np.linalg.norm(_proyectar(M_rec, obj) - img, axis=1)
        assert err.mean() < 5.0
        assert err.max() < 15.0

    def test_referencia_np(self):
        """
        Sanity de la SVD propia de alc.py: el autovector derecho del menor
        valor singular coincide con el de np.linalg (alineando el signo,
        libre en ambas).
        """
        rng = np.random.default_rng(SEED)
        Q, *_ = TestResolverDLT3D()._Q_exacto(rng)
        _, _, Vt_prop = alc.svd(Q)
        _, _, Vt_np = np.linalg.svd(Q)
        m_p = Vt_prop[11] * np.sign(np.dot(Vt_prop[11], Vt_np[11]))
        assert np.allclose(np.abs(m_p), np.abs(Vt_np[11]), atol=1e-6)


# ======================================================================
# degenerados (documenta el Ej 1a en código)
# ======================================================================
class TestDegenerados:

    def test_coplanar_no_recupera(self):
        """Todos los puntos en un plano: rango(Q) < 11 — la cámara como
        solución NO es única, la reconstrucción no matchea."""
        rng = np.random.default_rng(SEED)
        M, *_ = _camara_sintetica()
        obj = np.array([[2.0 * i - 4., 3.0 - i, 0.] for i in range(6)])
        Q = alc.obtenerQ3d(_proyectar(M, obj), obj)
        M_rec = alc.resolverDLT3D(Q)
        assert not _misma_m_a_escala(M, M_rec)


# ======================================================================
# devolverMatricesCalibracion3d (issue #5) — descomposición M -> K, R, t
# ======================================================================

def _camara_para_descomponer(rng):
    """
    Cámara de referencia para la descomposición, caso general del Ej 2:
    K con skew γ≠0 y focales distintos por eje, R rotación exacta,
    t con t3 > 0, y M con escala arbitraria (simula la salida de
    resolverDLT3D): M = μ·K[R|t], con M[2,3] >= 0 impuesto.
    IMPORTANTE: la columna 4 es K·t (ec. 4 del TP), no t a secas.
    """
    K = np.array([[1000., 5., 320.],       # γ = 5 ≠ 0, αu ≠ αv
                  [0.,   900., 240.],
                  [0.,    0.,  1.]])
    # rotación exacta: QR de una matriz fija con semilla (np.linalg es
    # libre acá: los tests NO van al entregado)
    R = np.linalg.qr(rng.standard_normal((3, 3)) + np.eye(3))[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1.0                    # asegura det(R) = +1
    t = np.array([10., -20., 400.])        # t3 > 0: puntos delante
    M = K @ np.hstack([R, t.reshape(3, 1)])
    M *= 3.7e-7                            # μ arbitrario (simula el DLT)
    if M[2, 3] < 0:
        M *= -1.0                          # convención m34 >= 0
    return M, K, R, t


def _camara_sintetica_skew(rng, gamma):
    """Cámara con skew forzado (para el test del signo de γ)."""
    K = np.array([[1000., gamma, 320.],
                  [0.,   900.,  240.],
                  [0.,    0.,   1.]])
    R = np.linalg.qr(rng.standard_normal((3, 3)) + np.eye(3))[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1.0
    t = np.array([10., -20., 400.])
    M = K @ np.hstack([R, t.reshape(3, 1)]) * 3.7e-7
    if M[2, 3] < 0:
        M *= -1.0
    return M, K, R, t


class TestDevolverMatricesCalibracion3d:

    def test_orden_y_formas(self, rng):
        """Contrato del enunciado (pág. 4): devuelve R, t y K en ese orden,
        con K33 = 1 por construcción."""
        M, *_ = _camara_para_descomponer(rng)
        R, t, K = alc.devolverMatricesCalibracion3d(M)
        assert R.shape == (3, 3) and t.shape == (3,) and K.shape == (3, 3)
        assert K[2, 2] == 1.0

    def test_recuperacion_exacta(self, rng):
        """Datos sin ruido: K, R y t exactos salvo punto flotante
        (con la escala μ arbitraria del DLT incluida)."""
        M, K_ref, R_ref, t_ref = _camara_para_descomponer(rng)
        R, t, K = alc.devolverMatricesCalibracion3d(M)
        assert np.allclose(K, K_ref, atol=1e-8)
        assert np.allclose(R, R_ref, atol=1e-8)
        assert np.allclose(t, t_ref, atol=1e-8)

    def test_ortonormalidad_R(self, rng):
        """R es rotación: RᵀR = I y det = +1 (lo pide el enunciado)."""
        M, *_ = _camara_para_descomponer(rng)
        R, *_ = alc.devolverMatricesCalibracion3d(M)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_consistencia_con_M(self, rng):
        """QA cruzado de todo el despeje: μ·K[R|t] reproduce la entrada."""
        M, *_ = _camara_para_descomponer(rng)
        R, t, K = alc.devolverMatricesCalibracion3d(M)
        mu = np.linalg.norm(M[2, :3])
        assert np.allclose(mu * K @ np.hstack([R, t.reshape(3, 1)]),
                           M, atol=1e-12 * max(1.0, np.abs(M).max()))

    def test_signo_de_gamma(self, rng):
        """γ sale con SU signo: el 'signo automático' de a1'·a2p.
        (Normas no distinguen signo; el producto interno sí.)"""
        M, *_ = _camara_sintetica_skew(rng, -5.0)      # γ negativo
        R, t, K = alc.devolverMatricesCalibracion3d(M)
        assert K[0, 1] < 0
        R, t, K = alc.devolverMatricesCalibracion3d(
            _camara_sintetica_skew(rng, 5.0)[0])       # γ positivo
        assert K[0, 1] > 0

    def test_invariante_a_escala(self, rng):
        """M con dos escalas distintas → misma (K, R, t): el μ es
        invisible para la descomposición tras ρ = 1/‖a3‖."""
        M1, *_ = _camara_para_descomponer(rng)
        M2 = M1 * 1e5
        if M2[2, 3] < 0:
            M2 *= -1.0
        R1, t1, K1 = alc.devolverMatricesCalibracion3d(M1)
        R2, t2, K2 = alc.devolverMatricesCalibracion3d(M2)
        assert np.allclose(K1, K2, atol=1e-5)
        assert np.allclose(R1, R2, atol=1e-10)
        assert np.allclose(t1, t2, atol=1e-5)

    def test_rechaza_m34_negativo(self, rng):
        """Guard de la convención de signos de resolverDLT3D: se exige
        m34 >= 0 (μ > 0, cámara 'hacia adelante')."""
        M, *_ = _camara_para_descomponer(rng)
        with pytest.raises(AssertionError):
            alc.devolverMatricesCalibracion3d(-M)

    def test_gamma_0_exacto(self, rng):
        """gamma_0 sobre cámara con γ=0 real: despeje exacto."""
        M, K_ref, R_ref, t_ref = _camara_sintetica_skew(rng, 0.0)
        R, t, K = alc.devolverMatricesCalibracion3d(M, gamma_0=True)
        assert K[0, 1] == 0.0
        assert np.allclose(K, K_ref, atol=1e-8)
        assert np.allclose(R, R_ref, atol=1e-8)
        assert np.allclose(t, t_ref, atol=1e-8)

    def test_gamma_0_sobre_gamma_real(self, rng):
        """γ=5 real con gamma_0 impuesto: K12 = 0 y αu absorbe el γ²
        (αu = √(αu² + γ²)); R queda sana (fila 3 intacta, r1 desde r2×r3).
        Documenta que la restricción falsa imponeed un residuo (en K)."""
        M, *_ = _camara_sintetica_skew(rng, 5.0)
        R, t, K = alc.devolverMatricesCalibracion3d(M, gamma_0=True)
        assert K[0, 1] == 0.0
        assert np.isclose(K[0, 0], np.hypot(1000., 5.), atol=1e-6)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_au_av_exacto(self, rng):
        """au_av sobre cámara que de verdad cumple αu=αv (γ=5): no-op →
        despeje exacto (αu_base == αv_base al punto flotante)."""
        K_ref = np.array([[1000., 5., 320.], [0., 1000., 240.], [0., 0., 1.]])
        R_ref = np.linalg.qr(rng.standard_normal((3, 3)) + np.eye(3))[0]
        if np.linalg.det(R_ref) < 0:
            R_ref[:, 0] *= -1.0
        t_ref = np.array([10., -20., 400.])
        M = K_ref @ np.hstack([R_ref, t_ref.reshape(3, 1)]) * 3.7e-7
        if M[2, 3] < 0:
            M *= -1.0
        R, t, K = alc.devolverMatricesCalibracion3d(M, au_av=True)
        assert np.allclose(K, K_ref, atol=1e-8)
        assert np.allclose(R, R_ref, atol=1e-8)
        assert np.allclose(t, t_ref, atol=1e-8)

    def test_ambas_flags_exacto(self, rng):
        """γ=0 y αu=αv en cámara que cumple ambos → exacto."""
        K_ref = np.array([[1000., 0., 320.], [0., 1000., 240.], [0., 0., 1.]])
        R_ref = np.linalg.qr(rng.standard_normal((3, 3)) + np.eye(3))[0]
        if np.linalg.det(R_ref) < 0:
            R_ref[:, 0] *= -1.0
        t_ref = np.array([10., -20., 400.])
        M = K_ref @ np.hstack([R_ref, t_ref.reshape(3, 1)]) * 3.7e-7
        if M[2, 3] < 0:
            M *= -1.0
        R, t, K = alc.devolverMatricesCalibracion3d(M, au_av=True, gamma_0=True)
        assert np.allclose(K, K_ref, atol=1e-8)
        assert np.allclose(R, R_ref, atol=1e-8)
        assert np.allclose(t, t_ref, atol=1e-8)

    def test_flags_default_regresion(self, rng):
        """Defaults (False, False): el caso general da lo mismo que el
        despeje base (regresión del código previo al inciso d)."""
        M, K_ref, R_ref, t_ref = _camara_para_descomponer(rng)
        R, t, K = alc.devolverMatricesCalibracion3d(M)
        assert np.allclose(K, K_ref, atol=1e-8)
        assert np.allclose(R, R_ref, atol=1e-8)
        assert np.allclose(t, t_ref, atol=1e-8)
