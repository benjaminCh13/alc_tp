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


def norma(X,p):
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
# Ejercicio 6 — Aterrizaje suave: POSIT coplanar
#
# Convenciones:
#   - Pose: X_cam = R·X_obj + t. Las filas de R son los ejes i, j, k de la
#     cámara expresados en el sistema del objeto; k es el eje óptico.
#   - Píxeles (u, v): origen arriba a la izquierda, v hacia abajo.
#   - K tiene fx != fy. POSIT trabaja con una única focal f = fx sobre
#     coordenadas "virtuales" centradas: x = fx·x_n, y = fx·y_n, donde
#     (x_n, y_n) son las coordenadas normalizadas K⁻¹·(u, v, 1). Así la
#     cámara virtual es diag(fx, fx) y no se confunde fy con fx.
#   - Los puntos del objeto son coplanares con W constante; la normal
#     unitaria al plano en el sistema del objeto es n = (0, 0, 1).
# ======================================================================


def calculoLambdaMu(I0, J0):
    """
    Resuelve λ y μ para I = I0 + λ·n, J = J0 + μ·n (POSIT coplanar), con
    I0, J0 ⟂ n, de modo que ‖I‖ = ‖J‖ e I·J = 0:

        λ·μ     = -I0·J0
        λ² - μ² = ‖J0‖² - ‖I0‖²

    Equivale a (λ + iμ)² = C con C = (‖J0‖² - ‖I0‖²) - 2i·(I0·J0); las dos
    soluciones son las raíces cuadradas opuestas de C.

    Parámetros
    ----------
    I0, J0 : ndarray (3,) — soluciones de mínimos cuadrados, ortogonales a n.

    Devuelve
    --------
    ((λ1, μ1), (λ2, μ2)) : tupla de dos parejas de floats, con
        λ1 >= 0 y (λ2, μ2) = (-λ1, -μ1). Si C ≈ 0 (caso degenerado:
        I0 y J0 ya ortogonales y de igual norma) ambas parejas son (0, 0).
    """
    # Parte real e imaginaria de C = (λ + iμ)²
    parte_real = np.sum(J0 * J0) - np.sum(I0 * I0)
    parte_imag = -2.0 * np.sum(I0 * J0)
    modulo = np.sqrt(parte_real**2 + parte_imag**2)

    # Raíz cuadrada compleja sin trigonometría:
    #   λ = √((|C| + Re C)/2),  μ = signo(Im C)·√((|C| - Re C)/2)
    # Los max(·, 0) evitan raíces de negativos por redondeo.
    lambda_1 = np.sqrt(max((modulo + parte_real) / 2.0, 0.0))
    mu_1 = np.sqrt(max((modulo - parte_real) / 2.0, 0.0))
    if parte_imag < 0:
        mu_1 = -mu_1

    return (float(lambda_1), float(mu_1)), (float(-lambda_1), float(-mu_1))


def calcularSalidaPOSIT(Is, Js, fx, m0, M0):
    """
    Recupera la pose (R, t) a partir de los vectores I = (f/Z0)·i y
    J = (f/Z0)·j de una pose candidata de POSIT.

    Parámetros
    ----------
    Is, Js : ndarray (3,) — vectores I y J de la candidata (ya corregidos
             con λ·n y μ·n).
    fx     : float — focal común de la cámara virtual, en píxeles.
    m0     : ndarray (2,) — imagen del punto de referencia M0 en
             coordenadas virtuales centradas (x0, y0).
    M0     : ndarray (3,) — punto de referencia en el sistema del objeto.

    Devuelve
    --------
    i, j, k : ndarray (3,) — ejes unitarios de la cámara en el sistema del
              objeto (k = i × j es el eje óptico).
    Z0      : float — profundidad de M0 en el sistema de la cámara.
    R       : ndarray (3, 3) — rotación con filas i, j, k (RᵀR = I, det R = +1).
    t       : ndarray (3,)   — traslación tal que X_cam = R·X_obj + t.
    """
    norma_I = np.sqrt(np.sum(Is * Is))
    norma_J = np.sqrt(np.sum(Js * Js))
    assert norma_I > 0 and norma_J > 0, "I o J nulos: pose degenerada"

    # Ejes de la cámara. Con λ, μ de calculoLambdaMu, I ⟂ J y ‖I‖ = ‖J‖,
    # así que i, j son ortonormales y k = i × j completa la rotación.
    eje_i = Is / norma_I
    eje_j = Js / norma_J
    eje_k = producto_cruz(eje_i, eje_j)
    R = np.vstack([eje_i, eje_j, eje_k])

    # Profundidad de M0: ‖I‖ = f/Z0
    Z0 = fx / norma_I

    # M0 en la cámara es (x0·Z0/f, y0·Z0/f, Z0); de ahí t = M0_cam - R·M0
    M0_camara = np.array([m0[0] * Z0 / fx, m0[1] * Z0 / fx, Z0])
    t = M0_camara - multiplicar(R, M0)
    return eje_i, eje_j, eje_k, Z0, R, t


def proyectarPuntos(R, t, obj_vecs, K):
    """
    Proyecta los puntos del objeto con la pose (R, t) y la matriz K.
    Auxiliar de evaluarPose; también sirve para graficar reproyecciones.

    Parámetros
    ----------
    R        : ndarray (3, 3) — rotación.
    t        : ndarray (3,)   — traslación.
    obj_vecs : ndarray (n, 3) — puntos en el sistema del objeto.
    K        : ndarray (3, 3) — matriz intrínseca.

    Devuelve
    --------
    proyectados   : ndarray (n, 2) — píxeles (u, v); nan si Z <= 0.
    profundidades : ndarray (n,)   — Z de cada punto en la cámara.
    """
    # Puntos en el sistema de la cámara: X_cam = R·X + t (una fila por punto)
    puntos_camara = multiplicar(obj_vecs, R.T) + t
    profundidades = puntos_camara[:, 2]
    validos = profundidades > 0

    # Proyección con K solo de los puntos delante de la cámara
    proyectados = np.zeros((obj_vecs.shape[0], 2)) + np.nan
    if np.any(validos):
        homogeneos = multiplicar(puntos_camara[validos], K.T)
        proyectados[validos, 0] = homogeneos[:, 0] / homogeneos[:, 2]
        proyectados[validos, 1] = homogeneos[:, 1] / homogeneos[:, 2]
    return proyectados, profundidades


def evaluarPose(R, t, obj_vecs, img_vecs, K):
    """
    Reproyecta los puntos del objeto con la pose (R, t) y la matriz K y
    mide el error contra los píxeles observados.

    Parámetros
    ----------
    R        : ndarray (3, 3) — rotación.
    t        : ndarray (3,)   — traslación.
    obj_vecs : ndarray (n, 3) — puntos en el sistema del objeto.
    img_vecs : ndarray (n, 2) — píxeles observados (u, v) originales.
    K        : ndarray (3, 3) — matriz intrínseca.

    Devuelve
    --------
    n_validos : int — cantidad de puntos con Z > 0 (proyecciones válidas).
    rms       : float — √(media de ‖(u,v)_proy - (u,v)_obs‖²) en píxeles;
                np.inf si algún punto queda con Z <= 0 (pose inválida).
    """
    assert obj_vecs.shape[0] == img_vecs.shape[0], "Cantidad de puntos distinta"

    proyectados, profundidades = proyectarPuntos(R, t, obj_vecs, K)
    n_validos = int(np.sum(profundidades > 0))

    # El RMS solo tiene sentido si todos los puntos son visibles
    if n_validos < img_vecs.shape[0]:
        return n_validos, np.inf
    rms = np.sqrt(np.mean(np.sum((proyectados - img_vecs)**2, axis=1)))
    return n_validos, float(rms)


def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol):
    """
    Inciso e: pose de la cámara por POSIT coplanar (ver iterarPOSIT).

    Parámetros
    ----------
    obj_vecs, img_vecs, K, MaxIt, atol : como en iterarPOSIT.

    Devuelve
    --------
    R : ndarray (3, 3) — rotación de la mejor pose válida.
    t : ndarray (3,)   — traslación de la mejor pose válida.
    """
    R, t, _ = iterarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol)
    return R, t


def iterarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol):
    """
    POSIT coplanar con diagnóstico: estima la pose de la cámara a partir de
    n >= 4 puntos coplanares (W constante) y sus imágenes, tomando
    obj_vecs[0] como M0. aplicarPOSIT devuelve solo (R, t) de esta función.

    En cada iteración resuelve A·I0 = b_u y A·J0 = b_v por mínimos
    cuadrados, arma las dos candidatas (±λ, ±μ), descarta las que dejan
    algún punto con Z <= 0, se queda con la de menor RMS y actualiza las
    correcciones de perspectiva ε_i = (a_i·k)/Z0.

    Parámetros
    ----------
    obj_vecs : ndarray (n, 3) — puntos del objeto; fila 0 = M0.
    img_vecs : ndarray (n, 2) — píxeles observados (u, v).
    K        : ndarray (3, 3) — matriz intrínseca (admite fx != fy).
    MaxIt    : int   — máximo de iteraciones (>= 1).
    atol     : float — se detiene cuando el RMS no mejora al menos atol
               píxeles respecto de la iteración anterior
               (RMS_anterior - RMS_actual < atol; un aumento también corta).

    Limitación del método: la iteración ε → ε' es un punto fijo que puede
    no ser atractor (radio espectral >= 1) con el plano casi paralelo a la
    imagen o perspectiva muy fuerte; ahí no converge a la pose real.

    Devuelve
    --------
    R, t : mejor pose válida encontrada (menor RMS en todas las iteraciones).
    info : dict con 'rms' (float), 'iteraciones' (int), 'motivo'
           ('atol' | 'MaxIt' | 'sin candidata válida') e 'historial_rms'
           (lista por iteración).

    Lanza ValueError si en la primera iteración ninguna candidata es válida.
    """
    obj_vecs = np.asarray(obj_vecs, dtype=float)
    img_vecs = np.asarray(img_vecs, dtype=float)
    assert obj_vecs.shape[0] == img_vecs.shape[0] >= 4, "Se necesitan n >= 4 correspondencias"
    assert MaxIt >= 1, "MaxIt debe ser >= 1"

    # --- Coordenadas virtuales centradas con focal común fx (K sin skew) ---
    fx, fy = K[0, 0], K[1, 1]
    assert K[0, 1] == 0, "Se asume K sin skew, como heli/K.csv"
    x_virtual = img_vecs[:, 0] - K[0, 2]
    y_virtual = (img_vecs[:, 1] - K[1, 2]) * fx / fy

    # --- Matriz del objeto A: filas a_i = M_i - M0 ---
    M0 = obj_vecs[0]
    vectores_objeto = obj_vecs[1:] - M0
    assert np.all(np.abs(vectores_objeto[:, 2]) < 1e-12), "POSIT coplanar requiere W constante"

    # A = [U V 0] tiene rango 2: A·I0 = b solo determina (I0_U, I0_V). La
    # solución de mínimos cuadrados de norma mínima tiene I0_W = 0 (es la
    # componente ⟂ al núcleo n), así que basta resolver sobre U, V con las
    # ecuaciones normales 2×2: (AᵀA)·(I0_U, I0_V) = Aᵀ·b.
    A_plano = vectores_objeto[:, :2]
    gram = multiplicar(A_plano.T, A_plano)
    det_gram = gram[0, 0] * gram[1, 1] - gram[0, 1] * gram[1, 0]
    assert det_gram > 1e-12 * (gram[0, 0] + gram[1, 1])**2, "Puntos del objeto colineales"
    gram_inversa = np.array([[gram[1, 1], -gram[0, 1]],
                             [-gram[1, 0], gram[0, 0]]]) / det_gram
    pseudoinversa = multiplicar(gram_inversa, A_plano.T)   # (2, n-1)

    normal = np.array([0.0, 0.0, 1.0])
    epsilon = np.zeros(vectores_objeto.shape[0])
    mejor = None
    historial_rms = []
    motivo = 'MaxIt'

    for iteracion in range(1, MaxIt + 1):
        # --- Sistemas lineales A·I0 = b_u, A·J0 = b_v ---
        b_u = x_virtual[1:] * (1 + epsilon) - x_virtual[0]
        b_v = y_virtual[1:] * (1 + epsilon) - y_virtual[0]
        I0 = np.hstack([multiplicar(pseudoinversa, b_u), 0.0])
        J0 = np.hstack([multiplicar(pseudoinversa, b_v), 0.0])

        # --- Dos candidatas; se elige la válida de menor RMS ---
        elegida = None
        for lam, mu in calculoLambdaMu(I0, J0):
            _, _, eje_k, Z0, R, t = calcularSalidaPOSIT(I0 + lam * normal, J0 + mu * normal, fx,
                                                        np.array([x_virtual[0], y_virtual[0]]), M0)
            n_validos, rms = evaluarPose(R, t, obj_vecs, img_vecs, K)
            if n_validos == obj_vecs.shape[0] and (elegida is None or rms < elegida[2]):
                elegida = (R, t, rms, eje_k, Z0)

        if elegida is None:
            if mejor is None:
                raise ValueError("POSIT: ninguna candidata deja todos los puntos con Z > 0")
            motivo = 'sin candidata válida'
            break

        rms_anterior = historial_rms[-1] if historial_rms else np.inf
        historial_rms.append(elegida[2])
        if mejor is None or elegida[2] < mejor[2]:
            mejor = elegida

        # --- Criterio de corte (inciso e): sin mejora de RMS >= atol, es decir
        # RMS_anterior - RMS_actual < atol (un aumento también corta). La
        # mejor pose ya quedó guardada en 'mejor' ---
        if rms_anterior - elegida[2] < atol:
            motivo = 'atol'
            break

        # --- Correcciones de perspectiva ε_i = (a_i·k)/Z0 ---
        eje_k, Z0 = elegida[3], elegida[4]
        epsilon = multiplicar(vectores_objeto, eje_k) / Z0

    info = {'rms': mejor[2], 'iteraciones': iteracion, 'motivo': motivo,
            'historial_rms': historial_rms}
    return mejor[0], mejor[1], info


def calcularPoseObjetivo(h):
    """
    Inciso f: pose de la cámara centrada sobre M0 a altura h, mirando en
    vertical hacia abajo, con la H derecha en la imagen (el punto blanco,
    junto a M1 = superior izquierda, queda arriba a la izquierda).

    Sistema del objeto: U a la derecha y V hacia arriba de la H, W = U × V
    sale del suelo hacia arriba. Ejes de la cámara buscados:
        i = +U (u crece a la derecha), j = -V (v crece hacia abajo),
        k = i × j = -W (mira hacia el suelo).

    Parámetros
    ----------
    h : float — altura del centro óptico sobre el plano de la H (h > 0).

    Devuelve
    --------
    R : ndarray (3, 3) — diag(1, -1, -1).
    t : ndarray (3,)   — (0, 0, h), ya que C = (0, 0, h) y t = -R·C.
    """
    assert h > 0, "La altura debe ser positiva"
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0]])
    t = np.array([0.0, 0.0, h])
    return R, t
