import numpy as np

# ======================================================================
# LABO — ALC: operaciones y álgebra base (issue #1)
# ======================================================================


def multiplicar(A,B):
    """
    Producto matricial A·B (reemplaza a @ / np.dot).

    Parámetros
    ----------
    A : ndarray (m, n)
    B : ndarray (n, p), o vector (n,)

    Devuelve
    --------
    ndarray (m, p), o vector (m,) si B es un vector.
    """
    if len(B.shape) == 1:
        return multiplicar(A, B.reshape(B.shape[0], 1))[:,0]
    assert A.shape[1] == B.shape[0], "Dimensiones incompatibles para multiplicar"
    res = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            res[i,j] = sum(A[i,:] * B[:,j])
    return res


def transponer(A):
    """
    Transpuesta de A.

    Parámetros
    ----------
    A : ndarray (m, n)

    Devuelve
    --------
    ndarray (n, m) con resultado[j, i] = A[i, j].
    """
    trasMatrix = np.zeros(A.shape[::-1], dtype=A.dtype)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            trasMatrix[j][i] = A[i][j]
    return trasMatrix


def restri(A,b,inferior=True):
    """
    Resuelve A·x = b con A triangular, por sustitución hacia adelante
    (inferior) o hacia atrás (superior).

    Parámetros
    ----------
    A : ndarray (n, n) — triangular.
    b : ndarray (n,)
    inferior : bool — True si A es triangular inferior, False si es superior.

    Devuelve
    --------
    x : ndarray (n,), o None si A o b son None o A tiene un cero en la diagonal.
    """

    if A is None or b is None:
        return None
    for i in range(A.shape[0]):
        if abs(A[i][i]) < 1e-15:
            return None

    x=np.zeros(b.shape, dtype=float)
    if inferior:
        for i in range(A.shape[0]):
            _sum = 0
            for j in range(i):
                _sum+= A[i][j] * x[j]
            x[i] = (b[i]-_sum)/A[i][i]
    else:
        for i in range(A.shape[0]-1, -1, -1):
            _sum = 0
            for j in range(A.shape[1]-1, i, -1):
                _sum+= A[i][j] * x[j]
            x[i] = (b[i]-_sum)/A[i][i]

    return x


def resolver_triangular_inferior(A, b):
    """
    Resuelve A·x = b con A triangular inferior (sustitución hacia adelante).

    Parámetros
    ----------
    A : ndarray (n, n) — triangular inferior.
    b : ndarray (n,)

    Devuelve
    --------
    x : ndarray (n,), o None si A tiene un cero en la diagonal.
    """
    return restri(A, b, True)


def resolver_triangular_superior(A, b):
    """
    Resuelve A·x = b con A triangular superior (sustitución hacia atrás).

    Parámetros
    ----------
    A : ndarray (n, n) — triangular superior.
    b : ndarray (n,)

    Devuelve
    --------
    x : ndarray (n,), o None si A tiene un cero en la diagonal.
    """
    return restri(A, b, False)


def norma(X,p=2):
    """
    Norma vectorial p de X.

    Parámetros
    ----------
    X : ndarray (n,)
    p : número >= 1, o 'inf' / float('inf')

    Devuelve
    --------
    float — ‖X‖_p
    """
    if p == 'inf' or p == float('inf'):
        return max([abs(xi) for xi in X])
    return (sum([abs(xi)**p for xi in X]))**(1/p)


# ======================================================================
# LABO — ALC: factorizaciones matriciales (issue #2)
# ======================================================================


# ---------------------------- Auxiliares ------------------------------

def esCuadrada(aMatrix):
    """
    Indica si una matriz es cuadrada.

    Parámetros
    ----------
    aMatrix : ndarray (m, n)

    Devuelve
    --------
    bool — True si m == n.
    """
    return aMatrix.shape[0] == aMatrix.shape[1]


def sonIgualesTol(x,y,aTol):
    """
    Compara dos escalares con tolerancia absoluta.

    Parámetros
    ----------
    x, y : float
    aTol : float — tolerancia absoluta.

    Devuelve
    --------
    bool — True si |x - y| <= aTol.
    """
    return abs(float(x)-float(y)) <= aTol


def esSimetrica(aMatrix, atol=1e-7):
    """
    Indica si una matriz es simétrica (A = Aᵀ) con tolerancia.

    Parámetros
    ----------
    aMatrix : ndarray (m, n)
    atol : float — tolerancia absoluta por elemento.

    Devuelve
    --------
    bool — True si es cuadrada y |A[i, j] - A[j, i]| <= atol para todo i, j.
    """
    if not esCuadrada(aMatrix): return False
    aMatrixTras = transponer(aMatrix)
    for i in range(aMatrix.shape[0]):
        for j in range(aMatrix.shape[1]):
            if not sonIgualesTol(aMatrix[i][j],aMatrixTras[i][j],atol): return False
    return True


# ------------------------------ LU / inversa --------------------------

def calculaLU(A):
    """
    Factorización LU sin pivoteo (eliminación gaussiana): A = L·U.

    Parámetros
    ----------
    A : ndarray (n, n)

    Devuelve
    --------
    L : ndarray (n, n) — triangular inferior con unos en la diagonal.
    U : ndarray (n, n) — triangular superior.
    cant_op : int — cantidad de operaciones realizadas.
    Si A no es cuadrada o aparece un pivote nulo, devuelve (None, None, cant_op).
    """
    cant_op = 0
    if A is None:
        return None, None, cant_op
    m = A.shape[0]
    n = A.shape[1]
    Ac = np.array(A, dtype=float)

    if m != n:
        return None, None, cant_op

    for k in range(m-1):
        akk = Ac[k, k]
        if abs(akk) < 1e-15:
            return None, None, cant_op
        for i in range(k+1, m):
            lik = Ac[i, k] / akk
            Ac[i, k] = lik
            Ac[i, k+1:] = Ac[i, k+1:] - lik * Ac[k, k+1:]
            cant_op += 1 + 2 * (n - k - 1) + 2

    L = np.eye(m, dtype=Ac.dtype)
    U = np.zeros((m, n), dtype=Ac.dtype)
    for i in range(m):
        for j in range(n):
            if i > j:
                L[i][j] = Ac[i][j]
            else:
                U[i][j] = Ac[i][j]

    return L, U, cant_op


def inversa(A):
    """
    Inversa de A, resolviendo A·x_j = e_j para cada columna con LU.

    No usa pivoteo: si aparece un pivote nulo devuelve None, aunque A sea
    inversible (ej: [[0, 1], [1, 0]]).

    Parámetros
    ----------
    A : ndarray (n, n) — inversible.

    Devuelve
    --------
    ndarray (n, n) — A⁻¹, o None si la LU sin pivoteo falla (pivote nulo).
    """
    if A is None:
        return None
    I = np.eye(A.shape[0])
    inv = np.eye(A.shape[0])
    L,U,o = calculaLU(A)
    if L is None:
        return None
    for j in range(A.shape[1]):
        #colInv = resAxEqb(A,inv[:,j]) esta bueno pero recalculamos mil veces LU
        y = restri(L,I[:,j])
        x = restri(U,y,False)
        if x is None:
            return None
        inv[:,j] = x
    return inv


# ------------------------------ Cholesky ------------------------------

def calculaLDV(A):
    """
    Factorización LDV a partir de la LU: A = L·D·V.

    Parámetros
    ----------
    A : ndarray (n, n)

    Devuelve
    --------
    L : ndarray (n, n) — triangular inferior con unos en la diagonal.
    D : ndarray (n, n) — diagonal (los pivotes de U).
    V : ndarray (n, n) — triangular superior con unos en la diagonal.
    (None, None, None) si la LU falla.
    """
    if A is None:
        return None, None, None
    L,U,o = calculaLU(A)
    if(L is None):
        return None, None, None
    n = A.shape[0]
    m = A.shape[1]
    D = np.eye(n)
    V = np.eye(n)
    for i in range(n):
        D[i,i] = U[i,i]
        for j in range(i+1, m):
            V[i,j] = U[i,j]/D[i,i]
    return L,D,V


def esSDP(A,atol=1e-5):
    """
    Indica si A es simétrica definida positiva (simétrica y con
    todos los pivotes de la LDV positivos).

    Parámetros
    ----------
    A : ndarray (n, n)
    atol : float — tolerancia para la simetría y para considerar positivo un pivote.

    Devuelve
    --------
    bool
    """
    if A is None: return False
    if not esSimetrica(A,atol): return False
    L,D,V = calculaLDV(A)
    if D is None: return False
    for i in range(D.shape[0]):
        if D[i,i]<=atol:
            return False
    return True


def cholesky(A, atol=1e-10):
    """
    Factorización de Cholesky: A = L·Lᵀ, con L = L_LDV·√D.

    Parámetros
    ----------
    A : ndarray (n, n) — simétrica definida positiva.
    atol : float — tolerancia que se le pasa a esSDP.

    Devuelve
    --------
    ndarray (n, n) — L triangular inferior, o None si A no es SDP.
    """
    if A is None: return None
    if not esSDP(A, atol): return None
    L, D, V = calculaLDV(A)
    n = A.shape[0]
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            R[i][j] = L[i][j] * np.sqrt(D[j][j])
    return R


# --------------------------------- QR ---------------------------------

def QR_con_GS(A, tol = 1e-12, retorna_nops=False):
    """
    Factorización QR por Gram-Schmidt clásico. Descarta las columnas
    linealmente dependientes (norma <= tol).

    Parámetros
    ----------
    A : ndarray (n, n) — cuadrada.
    tol : float — umbral para descartar columnas dependientes.
    retorna_nops : bool — si True, también devuelve la cantidad de operaciones.

    Devuelve
    --------
    Q : ndarray (n, r) — columnas ortonormales (r = columnas independientes).
    R : ndarray (r, n) — triangular superior.
    (nops : int, solo si retorna_nops)
    None si A no es cuadrada.
    """
    if not esCuadrada(A): return None

    Q = np.zeros(A.shape, dtype = float)
    R = np.zeros(A.shape, dtype = float)
    n = A.shape[0]
    nops = 0
    r = 0

    for j in range(A.shape[0]):
        qj = A[:,j]

        for k in range(r):
            R[k,j] = sum(Q[:,k] * qj)
            nops += n + (n-1) #producto interno
            qj = qj - R[k,j]*Q[:,k]
            nops += n + n #n productos, n restas

        nqj = norma(qj, 2)
        nops += n + (n-1) + 1 #norma otra vez
        if nqj > tol:
            Q[:,r] = qj/nqj
            nops +=n #n divisiones
            R[r,j] = nqj
            r+=1

    Q = Q[:, :r]
    R = R[:r, :]

    if retorna_nops:
        return Q,R,nops
    return Q,R


def QR_con_HH(A, tol=1e-12, extras=False):
    """
    Factorización QR por reflexiones de Householder: A = Q·R.

    Parámetros
    ----------
    A : ndarray (m, n) con m >= n.
    tol : float — umbral para saltear una reflexión cuando u ≈ 0.
    extras : bool — si True, también devuelve las R y Q intermedias de cada paso.

    Devuelve
    --------
    Q : ndarray (m, m) — ortogonal.
    R : ndarray (m, n) — triangular superior.
    (dict con 'R_matrices' y 'Q_matrices', solo si extras)
    None si m < n.
    """

    m = A.shape[0]
    n = A.shape[1]
    if m<n: return None
    if extras:
        R_list = []
        Q_list = []

    R = np.array(A, dtype=float)
    Q = np.eye(m)

    for k in range(n):
        # x = columna k de R, desde la fila k hasta el final
        x = R[k:m,k]

        # alpha = -sign(x_0)·‖x‖₂ (el signo evita restar números parecidos)
        a = 1 if x[0] >= 0 else - 1
        a = -a * norma(x,2)

        # u = x - alpha·e_1 (e_1 de largo m-k: solo cambia u[0])
        e1 = np.zeros(m-k)
        e1[0] = 1
        u = x - a*e1

        # Si u ≈ 0, la columna ya está alineada con e_1 y no hace falta reflejar
        if norma(u,2) > tol:

            # u normalizado y como columna (m-k)×1, para armar el producto externo u·uᵀ
            u = u/norma(u,2)
            u = u.reshape(-1,1)

            # Reflexión H_k = I - 2·u·uᵀ, de tamaño (m-k)×(m-k)
            Hk = np.eye(m-k) - 2*multiplicar(u,transponer(u))

            # Se embebe en una identidad m×m, en el bloque inferior derecho desde (k, k)
            H_k = np.eye(m)
            H_k[k:,k:] = Hk

            # R ← H_k·R  y  Q ← Q·H_kᵀ
            R = multiplicar(H_k,R)
            Q = multiplicar(Q,transponer(H_k))

            # (si extras) guardar una copia de R y de Q de este paso
            if extras:
                R_list.append(np.array(R))
                Q_list.append(np.array(Q))

    if extras: return Q,R, {'R_matrices': R_list, 'Q_matrices': Q_list}
    return Q,R


def qr(A,metodo='RH' , tol=1e-12):
    """
    Factorización QR: A = Q·R.

    Parámetros
    ----------
    A : ndarray (m, n)
    metodo : 'RH' (Householder, admite m >= n) o 'GS' (Gram-Schmidt, solo cuadradas).
    tol : float — tolerancia del método elegido.

    Devuelve
    --------
    (Q, R) según el método, o None si el método no existe o A no es válida para él.
    """
    if metodo == 'RH':
        return QR_con_HH(A, tol)
    elif metodo == 'GS':
        return QR_con_GS(A, tol)
    return None


# ---------------------------- Autovalores -----------------------------

def metpot2k(A,tol=1e-15,K=1000):
    """
    Método de la potencia (aplicando A dos veces por iteración): aproxima
    el autovector y el autovalor de mayor módulo.

    Parámetros
    ----------
    A : ndarray (n, n)
    tol : float — corta cuando |v_nuevo · v_viejo - 1| <= tol.
    K : int — cantidad máxima de iteraciones.

    Devuelve
    --------
    v : ndarray (n, 1) — autovector dominante normalizado.
    lamb : float — autovalor dominante (vᵀ·A·v).
    k : int — iteraciones realizadas.
    """

    # Vector inicial fijo (1, 2, ..., n): no puede ser ortogonal al autovector
    # dominante salvo casos muy particulares, y hace el resultado reproducible
    v = np.arange(1, A.shape[0]+1, dtype=float).reshape(A.shape[0], 1)
    v_ = v
    for i in range(2):
        v_= multiplicar(A,v_)
        if norma(v_, 2) > 0:
            v_ = v_/norma(v_,2)
        else:
            v_ = np.zeros((v_.shape[0],1))

    e = multiplicar(transponer(v_), v)[0,0]
    k = 0
    while abs(e-1) > tol and k < K:
        v = v_

        for i in range(2):
            v_= multiplicar(A,v_)
            if norma(v_, 2) > 0:
                v_ = v_/norma(v_,2)
            else:
                v_ = np.zeros((v_.shape[0],1))

        e = multiplicar(transponer(v_),v)[0,0]
        k += 1

    lamb = multiplicar(transponer(v_), multiplicar(A,v_))[0,0]
    #err = e - 1

    return v_, lamb, k #, err


def autovalores(A, tol=1e-15, K=1000):
    """
    Diagonaliza una matriz simétrica: A = S·D·Sᵀ. En cada paso calcula el
    autovalor dominante con metpot2k, deflaciona con una reflexión de
    Householder y repite sobre la submatriz.

    Parámetros
    ----------
    A : ndarray (n, n) — simétrica.
    tol : float — tolerancia de metpot2k.
    K : int — iteraciones máximas de metpot2k.

    Devuelve
    --------
    S : ndarray (n, n) — ortogonal, con los autovectores como columnas.
    D : ndarray (n, n) — diagonal, con los autovalores de mayor a menor módulo.
    None si A no es simétrica.
    """
    if not esSimetrica(A): return None

    # Caso 1×1: ya es diagonal, su autovalor es el único elemento y su autovector es [1]
    if A.shape[0] == 1:
        return np.eye(1), np.array(A, dtype=float)

    v1, lamb1, ktrash = metpot2k(A,tol,K)

    I = np.eye(A.shape[0])
    e1=np.zeros((A.shape[0], 1))
    e1[0,0] = 1
    res1 = e1-v1
    Hv1 = I - 2*(multiplicar(res1, transponer(res1))/multiplicar(transponer(res1),res1))

    if A.shape[0] == 2:
        S = Hv1
        D = multiplicar(Hv1, multiplicar(A, transponer(Hv1)))
        D = (D + transponer(D)) / 2 #simetrizar (el redondeo la deja apenas asimetrica)

    else:
        B = multiplicar(Hv1, multiplicar(A, transponer(Hv1)))
        B = (B + transponer(B)) / 2 #simetrizar (el redondeo la deja apenas asimetrica)
        A_ = B[1:,1:]
        S_, D_ = autovalores(A_,tol,K)

        D = np.zeros((D_.shape[0]+1, D_.shape[1]+1))
        D[0,0] = lamb1
        D[1:, 1:] = D_

        S = np.eye(S_.shape[0]+1)
        S[1:, 1:] = S_
        S = multiplicar(Hv1, S)

    return S, D


# ------------------------- Pendientes del #2 --------------------------

def svd(A, tol=1e-7):
    """
    Descomposición en valores singulares: A = U·Σ·Vᵀ, calculada a partir de
    la diagonalización de AᵀA (AᵀA = V·Σ²·Vᵀ).

    Se devuelve V completa y U reducida: solo se calculan los u_i de los
    σ_i no nulos, porque u_i = A·v_i / σ_i. Para el DLT alcanza, ya que la
    solución es Vt[-1], el v del menor valor singular.

    Parámetros
    ----------
    A : ndarray (m, n)
    tol : float — un σ_i con |σ_i - 0| <= tol se considera cero (no se
          calcula su u_i). No puede ser mucho menor que 1e-8: al pasar por
          AᵀA, un σ que en teoría es 0 sale como √(error de redondeo) ≈ 1e-8·σ_1.

    Devuelve
    --------
    U : ndarray (m, r) — vectores singulares izquierdos (columnas) de los
        r valores singulares no nulos.
    S : ndarray (n,) — valores singulares de mayor a menor (incluye los nulos;
        si m < n, al menos n - m son cero).
    Vt : ndarray (n, n) — vectores singulares derechos (filas); Vt[-1] es el
         del menor valor singular (solución de mín ‖Ax‖ con ‖x‖ = 1).
    """
    n = A.shape[1]

    # Paso 1: armar AᵀA (n×n, simétrica y semidefinida positiva)
    AtA = multiplicar(transponer(A), A)

    # Paso 2: diagonalizar AᵀA = S·D·Sᵀ
    #   columnas de S = autovectores v_i, diagonal de D = autovalores λ_i = σ_i²
    #   (autovalores ya los devuelve ordenados de mayor a menor)
    S, D = autovalores(AtA)

    # Paso 3: valores singulares σ_i = √λ_i
    #   (λ_i >= 0 en teoría; un negativo chiquito es error de redondeo → 0)
    sigmas = np.zeros(n)
    for i in range(n):
        lamb = D[i, i]
        if lamb < 0:
            lamb = 0
        sigmas[i] = np.sqrt(lamb)

    # Paso 4: vectores singulares izquierdos u_i = A·v_i / σ_i,
    #   solo para los σ_i no nulos (no se puede dividir por cero)
    columnasU = []
    for i in range(n):
        if not sonIgualesTol(sigmas[i], 0, tol):
            u_i = multiplicar(A, S[:, i]) / sigmas[i]
            columnasU.append(u_i)
    U = np.array(columnasU).T

    # Paso 5: devolver en el formato de np.linalg.svd (V traspuesta)
    Vt = transponer(S)
    return U, sigmas, Vt


def producto_cruz(a, b):
    """
    Producto vectorial a × b (reemplaza a np.cross).

    Parámetros
    ----------
    a, b : ndarray (3,)

    Devuelve
    --------
    ndarray (3,) — vector perpendicular a a y a b.
    """
    assert a.shape[0] == 3 and b.shape[0] == 3, "El producto cruz está definido solo en R^3"
    res = np.zeros(3)
    res[0] = a[1]*b[2] - a[2]*b[1]
    res[1] = a[2]*b[0] - a[0]*b[2]
    res[2] = a[0]*b[1] - a[1]*b[0]
    return res

# ======================================================================
# Ejercicio 2 — Calibración de cámara 3-D (DLT)
# ======================================================================
def _ortonormalizarR(r1, r2, r3):
    """
    Devuelve una rotación exacta (RRᵀ = I, det = +1) a partir de tres
    filas r1, r2, r3 (convención filas = ejes de cámara en mundo) que
    vienen de la descomposición de M y pueden salir con ruido.

    Procedimiento (Gram-Schmidt corto, de abajo hacia arriba):
      1. r3 se toma como referencia (normalizado).
      2. r2 se reortogonaliza contra r3 y se normaliza.
      3. r1 se RECOMPUTA como r2×r3 → garantiza det = +1 exacto.

    Parámetros
    ----------
    r1, r2, r3 : ndarrays (3,) — filas de la rotación estimada.

    Devuelve
    --------
    R : ndarray (3, 3) — rotación ortonormal (filas r1, r2, r3).
    """
    # 1. normalizar r3
    r3 = r3 / norma(r3)

    # 2. quitar la componente de r3 de r2 y normalizar
    r2 = r2 - np.sum(r2 * r3) * r3
    r2 = r2 / norma(r2)

    # 3. r1 = r2×r3: unitaria por construcción y de la mano correcta
    #    (sistema de filas derecho: r3 = r1×r2 ⟺ r1 = r2×r3)
    r1 = producto_cruz(r2, r3)

    R = np.zeros((3, 3))
    R[0, :] = r1
    R[1, :] = r2
    R[2, :] = r3
    return R


def obtenerQ3d(imgPts, objPts):
    """
    Arma la matriz Q del sistema DLT homogéneo (ec. 6 de Tsai 1987): Q m = 0.

    Parámetros
    ----------
    imgPts : ndarray (n, 2) — mediciones en esta vista, filas (u, v) en píxeles.
    objPts : ndarray (n, 3) — correspondencias (U, V, W) en unidades del
             patrón (cm en calib/3d), mismo orden que imgPts.

    Devuelve
    --------
    Q : ndarray (2n, 12) — sistema Q m = 0, con m = (m11..m34) los
        coeficientes de M por filas. Dos filas por punto i:
          fila impar : [U, V, W, 1,  0,  0,  0,  0,  -u·U, -u·V, -u·W, -u]
          fila par   : [0, 0, 0, 0,  U,     V,     W,     1,    -v·U, -v·V, -v·W, -v]

    Supone: los n puntos comparten una única pose (misma cámara fija
    respecto del patrón dentro de la vista). Con n < 6 el sistema queda
    subdeterminado (rango esperado 11, Ej 1-a) — lo valida resolverDLT3D.
    """
    n_obj = objPts.shape[0]
    n_img = imgPts.shape[0]
    assert n_obj == n_img, "Cantidad de puntos 3D y 2D debe ser la misma"

    Q = np.zeros((2*n_img, 12))

    for i in range(n_img):
        u, v = imgPts[i]
        U, V, W = objPts[i]

        Q[2*i]     = [U, V, W, 1, 0, 0, 0, 0, -u*U, -u*V, -u*W, -u]
        Q[2*i + 1] = [0, 0, 0, 0, U, V, W, 1, -v*U, -v*V, -v*W, -v]
    
    return Q

def resolverDLT3D(Q):
    """
    Resuelve el DLT: mínimos cuadrados homogéneo → matriz de cámara M.

    Implementa la ec. 8 (Ej 1-b): m = arg min ‖Qm‖² sujeto a ‖m‖ = 1,
    equivalente al autovector del menor autovalor de QᵀQ. Se resuelve
    con SVD: m = v12, última fila de Vt (valor singular mínimo).

    Parámetros
    ----------
    Q : ndarray (2n, 12) — sistema de obtenerQ3d; requiere n ≥ 6 puntos
        no coplanarios (Q de rango 11, Ej 1-a), se valida con asserts.

    Devuelve
    --------
    M : ndarray (3, 4) — cámara μ·K[R|t], reshape de m por filas.
        Determinada salvo escala; se fija el signo con m34 ≥ 0
        ("mira adelante de la cámara"; en calib/3d equivale a w > 0)
        para que la descomposición de devolverMatricesCalibracion3d
        sea única.
    """

    assert Q.shape[1] == 12, "Q debe tener 12 columnas (12 incognitas del DLT)"
    assert Q.shape[0] >= 12, "Q debe tener al menos 12 filas (n >= 6 puntos, Ej 1a)"

    _,_,Vt = svd(Q)
    v12 = Vt[11] # v12 es un array de R12 -> autovector de menor sigma
    m = v12 # [m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34]

    M = m.reshape(3,4) # reshape reordena row-major, llena fila 1, fila 2, fila 3, con elementos de izquierda a derecha
    # M =
    # [m11, m12, m13, m14] = [r11, r12, r13, t1] 
    # [m21, m22, m23, m24] = [r21, r22, r23, t2]
    # [m31, m32, m33, m34] = [r31, r32, r33, t3]

    if M[2,3] < 0: M *= -1 # Opcional: se pide para determinar el signo y facilitar la ejecución de Tests

    return M


def devolverMatricesCalibracion3d(M, au_av=False, gamma_0=False):
    """
    Descompone la matriz de cámara M = μ·K[R|t] (salida de resolverDLT3D)
    en intrínsecos K (K33 = 1), rotación R (ortonormal, filas = ejes de
    cámara en mundo) y traslación t (en cm, unidades del patrón calib/3d).

    Parámetros
    ----------
    M : ndarray (3, 4) — matriz de cámara de resolverDLT3D (requiere m34 ≥ 0).
    au_av (bool, opcional) — impone αu = αv conservando αv, el despeje
        directo de la fila 2 (sin término de skew); r1 y t1 se
        re-derivan con el αu nuevo. Default: False.
    gamma_0 (bool, opcional) — impone γ = 0 antes del despeje de αu.
        Default: False.

    Devuelve
    --------
    R : ndarray (3, 3) — rotación ortonormal (RRᵀ = I, det = +1).
    t : ndarray (3,)   — traslación en cm.
    K : ndarray (3, 3) — intrínsecos con K33 = 1.
    """
    
    assert M.shape == (3, 4), "M debe ser (3, 4) (salida de resolverDLT3D)"
    assert M[2, 3] >= 0, "m34 < 0: viola la convención de resolverDLT3D"

    a1 = M[0, :3] # μ·(αu·r1 + γ·r2 + u0·r3)
    a2 = M[1, :3] # μ·(αv·r2 + v0·r3)
    a3 = M[2, :3] # μ·r3
    b  = M[:, 3]  # μ·(αu·t1 + γ·t2 + u0·t3, αv·t2 + v0·t3, t3)

    # --- escala: ρ = 1/μ (μ = ‖a3‖ > 0 por la convención m34 ≥ 0) -----
    # a3 = M_est[2,:3] = μ · r3ᵀ   →   ‖a3‖ = μ · ‖r3‖ = μ · 1 = μ
    rho = 1 / norma(a3)

    # Despejamos centro óptico (u0, v0), alpha_u, alpha_v, gamma, R
    
    # Recordando: 
    # M = μ·K[R|t]
    # ri,j son ortonormales: r1·r3 = 0,    r2·r3 = 0,    r3·r3 = 1 
    # 1/μ² = ρ²

    # --- centro óptico -------------------------------------------------
    # Del producto interno: a1·a3 - tenemos que
    # a1·a3 = μ²·(αu·r1 + γ·r2 + u0·r3)·r3
    #       = μ²·[ αu·(r1·r3)  +  γ·(r2·r3)  +  u0·(r3·r3) ] = μ²·u0
    u0 = rho**2 * np.sum(a1 * a3) # u0 = ρ²·(a1·a3)
    
    # a2·a3 - Similar
    v0 = rho**2 * np.sum(a2 * a3) # v0 = ρ²·(a2·a3)

    # --- filas sin el aporte del centro  --------
    a1p = a1 - u0 * a3 # (a1' = μ(αu r1 + γ r2))
    a2p = a2 - v0 * a3 # (a2' = μ(αv r2))

    # ‖a2p‖ = ‖μ·αv·r2‖ = μ·αv·‖r2‖ = μ·αv·1 = μ·αv
    a2pNorma = norma(a2p)
    assert a2pNorma > 0, "a2' degenerado: M no corresponde a una cámara válida"

    # Despejo alpha_v y gamma
    alpha_v = rho * a2pNorma # αv = (1/μ)·μ·αv = ρ·μ·αv = ρ·‖a2'‖
    gamma = rho * np.sum(a1p * a2p) / a2pNorma # γ = ρ·(a1'·a2')/‖a2'‖
    if gamma_0: gamma = 0.0

    # Despejo alpha_u
    # ‖a1p‖² = ‖μ(αu·r1 + γ·r2)‖²
    #        = μ²·‖αu·r1 + γ·r2‖²
    #        = μ²·[ αu²·‖r1‖² + γ²·‖r2‖² + 2·αu·γ·(r1·r2) ]
    #        = μ²·[ αu² + γ² + 0 ] 
    #        = μ²·(αu² + γ²)
    # αu² = ‖a1p‖²/μ² − γ² = ρ²·‖a1p‖² − γ²; max(0,·) absorbe redondeos negativos de punto flotante
    if au_av:
        alpha_u = alpha_v
    else:
        alpha_u = np.sqrt(np.max((0.0, rho**2 * np.sum(a1p * a1p) - gamma**2)))
    
    # --- K --Ecuación (4) del TP -----------------------------------------
    K = np.zeros((3, 3))
    K[0, 0] = alpha_u
    K[0, 1] = gamma
    K[0, 2] = u0
    K[1, 1] = alpha_v
    K[1, 2] = v0
    K[2, 2] = 1.0

    # --- R (filas = ejes de cámara en el mundo) ----------------------
    r2 = a2p / a2pNorma
    r3 = rho * a3
    r1 = (rho * a1p - gamma * r2) / alpha_u
    R = _ortonormalizarR(r1, r2, r3) # garantiza RRᵀ = I y det(R) = +1

    # --- t (traslación en cm) ----------------------------------------
    t3 = rho * b[2]
    t2 = (rho * b[1] - v0 * t3) / alpha_v
    t1 = (rho * b[0] - u0 * t3 - gamma * t2) / alpha_u
    t = np.array((t1, t2, t3))

    return R,t,K
