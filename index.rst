.. raw:: html

	<style type="text/css">.menu>li.comparison-on>a {border-color:#444;cursor: default;}</style>

.. toctree::
   :hidden:

   julia-cheatsheet
   python-cheatsheet
   stats-cheatsheet

MATLAB--Python--Julia--KDB+ cheatsheet
===========================================

Dependencies and Setup
--------------------------

In the Python code we assume that you have already run :code:`import numpy as np`

In the Julia, we assume you are using **v1.0.2 or later** with Compat **v1.3.0 or later** and have run :code:`using LinearAlgebra, Statistics, Compat`

    

Creating Vectors
----------------

.. container:: multilang-table

    +----------------------------+---------------------------+----------------------------------------+-----------------------+----------------------+
    |         Operation          |           MATLAB          |                 Python                 |         Julia         |         KDB+         |
    +============================+===========================+========================================+=======================+======================+
    |                            | .. code-block:: matlab    | .. code-block:: python                 | .. code-block:: julia | .. code-block:: q    |
    |                            |                           |                                        |                       |                      |
    | Row vector: size (1, n)    |     A = [1 2 3]           |  A = np.array([1, 2, 3]).reshape(1, 3) |     A = [1 2 3]       |  ##code goes here    |
    +----------------------------+---------------------------+----------------------------------------+-----------------------+----------------------+
    |                            | .. code-block:: matlab    | .. code-block:: python                 | .. code-block:: julia | .. code-block:: q    |
    |                            |                           |                                        |                       |                      |
    | Column vector: size (n, 1) |     A = [1; 2; 3]         |  A = np.array([1, 2, 3]).reshape(3, 1) |     A = [1 2 3]'      |  ##code goes here    |
    +----------------------------+---------------------------+----------------------------------------+-----------------------+----------------------+
    |                            | Not possible              | .. code-block:: python                 | .. code-block:: julia | .. code-block:: q    |
    |                            |                           |                                        |                       |                      |
    | 1d array: size (n, )       |                           |  A = np.array([1, 2, 3])               |     A = [1; 2; 3]     |  ##code goes here    |
    |                            |                           |                                        |                       |                      |
    |                            |                           |                                        | or                    |                      |
    |                            |                           |                                        |                       |                      |
    |                            |                           |                                        | .. code-block:: julia |                      |
    |                            |                           |                                        |                       |                      |
    |                            |                           |                                        |     A = [1, 2, 3]     |                      |
    +----------------------------+---------------------------+----------------------------------------+-----------------------+----------------------+
    |                            | .. code-block:: matlab    | .. code-block:: python                 | .. code-block:: julia | .. code-block:: q    |
    |                            |                           |                                        |                       |                      |
    | Integers from j to n with  |     A = j:k:n             |  A = np.arange(j, n+1, k)              |     A = j:k:n         |  ##code goes here    |
    | step size k                |                           |                                        |                       |                      |
    +----------------------------+---------------------------+----------------------------------------+-----------------------+----------------------+
    |                            | .. code-block:: matlab    | .. code-block:: python                 | .. code-block:: julia | .. code-block:: q    |
    |                            |                           |                                        |                       |                      |
    | Linearly spaced vector     |     A = linspace(1, 5, k) |  A = np.linspace(1, 5, k)              |     A = range(1, 5,   |  ##code goes here    |
    | of k points                |                           |                                        |     length = k)       |                      |
    +----------------------------+---------------------------+----------------------------------------+-----------------------+----------------------+



Creating Matrices
-----------------

.. container:: multilang-table

    +------------------------+--------------------------------+-----------------------------------------------------+-------------------------------+----------------------+
    |       Operation        |             MATLAB             |                        Python                       |             Julia             |         KDB+         |
    +========================+================================+=====================================================+===============================+======================+
    |                        | .. code-block:: matlab         | .. code-block:: python                              | .. code-block:: julia         | .. code-block:: q    |
    |                        |                                |                                                     |                               |                      |
    | Create a matrix        |     A = [1 2; 3 4]             |   A = np.array([[1, 2], [3, 4]])                    |     A = [1 2; 3 4]            |  ##code goes here    |
    +------------------------+--------------------------------+-----------------------------------------------------+-------------------------------+----------------------+
    |                        | .. code-block:: matlab         | .. code-block:: python                              | .. code-block:: julia         | .. code-block:: q    |
    |                        |                                |                                                     |                               |                      |
    | 2 x 2 matrix of zeros  |     A = zeros(2, 2)            |   A = np.zeros((2, 2))                              |     A = zeros(2, 2)           |  ##code goes here    |
    +------------------------+--------------------------------+-----------------------------------------------------+-------------------------------+----------------------+
    |                        | .. code-block:: matlab         | .. code-block:: python                              | .. code-block:: julia         | .. code-block:: q    |
    |                        |                                |                                                     |                               |                      |
    | 2 x 2 matrix of ones   |     A = ones(2, 2)             |   A = np.ones((2, 2))                               |     A = ones(2, 2)            |  ##code goes here    |
    +------------------------+--------------------------------+-----------------------------------------------------+-------------------------------+----------------------+
    |                        | .. code-block:: matlab         | .. code-block:: python                              | .. code-block:: julia         | .. code-block:: q    |
    |                        |                                |                                                     |                               |                      |
    | 2 x 2 identity matrix  |     A = eye(2, 2)              |   A = np.eye(2)                                     |     A = I # will adopt        |  ##code goes here    |
    |                        |                                |                                                     |     # 2x2 dims if demanded by |                      |
    |                        |                                |                                                     |     # neighboring matrices    |                      |
    +------------------------+--------------------------------+-----------------------------------------------------+-------------------------------+----------------------+
    |                        | .. code-block:: matlab         | .. code-block:: python                              | .. code-block:: julia         | .. code-block:: q    |
    |                        |                                |                                                     |                               |                      |
    | Diagonal matrix        |     A = diag([1 2 3])          |   A = np.diag([1, 2, 3])                            |    A = Diagonal([1, 2,        |  ##code goes here    |
    |                        |                                |                                                     |        3])                    |                      |
    +------------------------+--------------------------------+-----------------------------------------------------+-------------------------------+----------------------+
    |                        | .. code-block:: matlab         | .. code-block:: python                              | .. code-block:: julia         | .. code-block:: q    |
    |                        |                                |                                                     |                               |                      |
    | Uniform random numbers |     A = rand(2, 2)             |   A = np.random.rand(2, 2)                          |     A = rand(2, 2)            |  ##code goes here    |
    +------------------------+--------------------------------+-----------------------------------------------------+-------------------------------+----------------------+
    |                        | .. code-block:: matlab         | .. code-block:: python                              | .. code-block:: julia         | .. code-block:: q    |
    |                        |                                |                                                     |                               |                      |
    | Normal random numbers  |     A = randn(2, 2)            |   A = np.random.randn(2, 2)                         |     A = randn(2, 2)           |  ##code goes here    |
    +------------------------+--------------------------------+-----------------------------------------------------+-------------------------------+----------------------+
    |                        | .. code-block:: matlab         | .. code-block:: python                              | .. code-block:: julia         | .. code-block:: q    |
    |                        |                                |                                                     |                               |                      |
    | Sparse Matrices        |     A = sparse(2, 2)           |   from scipy.sparse import coo_matrix               |     using SparseArrays        |  ##code goes here    |
    |                        |     A(1, 2) = 4                |                                                     |     A = spzeros(2, 2)         |                      |
    |                        |     A(2, 2) = 1                |   A = coo_matrix(([4, 1],                           |     A[1, 2] = 4               |                      |
    |                        |                                |                   ([0, 1], [1, 1])),                |     A[2, 2] = 1               |                      |
    |                        |                                |                   shape=(2, 2))                     |                               |                      |
    +------------------------+--------------------------------+-----------------------------------------------------+-------------------------------+----------------------+
    |                        | .. code-block:: matlab         | .. code-block:: python                              | .. code-block:: julia         | .. code-block:: q    |
    |                        |                                |                                                     |                               |                      |
    | Tridiagonal Matrices   |     A = [1 2 3 NaN;            |   import sp.sparse as sp                            |     x = [1, 2, 3]             |  ##code goes here    |
    |                        |          4 5 6 7;              |   diagonals = [[4, 5, 6, 7], [1, 2, 3], [8, 9, 10]] |     y = [4, 5, 6, 7]          |                      |
    |                        |          NaN 8 9 0]            |   sp.diags(diagonals, [0, -1, 2]).toarray()         |     z = [8, 9, 10]            |                      |
    |                        |     spdiags(A',[-1 0 1], 4, 4) |                                                     |     Tridiagonal(x, y, z)      |                      |
    +------------------------+--------------------------------+-----------------------------------------------------+-------------------------------+----------------------+

Manipulating Vectors and Matrices
---------------------------------

.. container:: multilang-table

    +--------------------------------+---------------------------------------+------------------------------+----------------------------+----------------------+
    |           Operation            |                 MATLAB                |            Python            |           Julia            |         KDB+         |
    +================================+=======================================+==============================+============================+======================+
    |                                | .. code-block:: matlab                | .. code-block:: python       | .. code-block:: julia      | .. code-block:: q    |
    |                                |                                       |                              |                            |                      |
    | Transpose                      |     A.'                               |   A.T                        |     transpose(A)           |  ##code goes here    |
    +--------------------------------+---------------------------------------+------------------------------+----------------------------+----------------------+
    |                                | .. code-block:: matlab                | .. code-block:: python       | .. code-block:: julia      | .. code-block:: q    |
    | Complex conjugate transpose    |                                       |                              |                            |                      |
    | (Adjoint)                      |     A'                                |   A.conj()                   |     A'                     |  ##code goes here    |
    +--------------------------------+---------------------------------------+------------------------------+----------------------------+----------------------+
    |                                | .. code-block:: matlab                | .. code-block:: python       | .. code-block:: julia      | .. code-block:: q    |
    |                                |                                       |                              |                            |                      |
    | Concatenate horizontally       |     A = [[1 2] [1 2]]                 |    B = np.array([1, 2])      |     A = [[1 2] [1 2]]      |  ##code goes here    |
    |                                |                                       |    A = np.hstack((B, B))     |                            |                      |
    |                                | or                                    |                              | or                         |                      |
    |                                |                                       |                              |                            |                      |
    |                                | .. code-block:: matlab                |                              | .. code-block:: julia      |                      |
    |                                |                                       |                              |                            |                      |
    |                                |     A = horzcat([1 2], [1 2])         |                              |    A = hcat([1 2], [1 2])  |                      |
    +--------------------------------+---------------------------------------+------------------------------+----------------------------+----------------------+
    |                                | .. code-block:: matlab                | .. code-block:: python       | .. code-block:: julia      | .. code-block:: q    |
    |                                |                                       |                              |                            |                      |
    | Concatenate vertically         |     A = [[1 2]; [1 2]]                |    B = np.array([1, 2])      |     A = [[1 2]; [1 2]]     |  ##code goes here    |
    |                                |                                       |    A = np.vstack((B, B))     |                            |                      |
    |                                | or                                    |                              | or                         |                      |
    |                                |                                       |                              |                            |                      |
    |                                | .. code-block:: matlab                |                              | .. code-block:: julia      |                      |
    |                                |                                       |                              |                            |                      |
    |                                |     A = vertcat([1 2], [1 2])         |                              |    A = vcat([1 2], [1 2])  |                      |
    +--------------------------------+---------------------------------------+------------------------------+----------------------------+----------------------+
    |                                | .. code-block:: matlab                | .. code-block:: python       | .. code-block:: julia      | .. code-block:: q    |
    |                                |                                       |                              |                            |                      |
    | Reshape (to 5 rows, 2 columns) |    A = reshape(1:10, 5, 2)            |    A = A.reshape(5, 2)       |    A = reshape(1:10, 5, 2) |  ##code goes here    |
    +--------------------------------+---------------------------------------+------------------------------+----------------------------+----------------------+
    |                                | .. code-block:: matlab                | .. code-block:: python       | .. code-block:: julia      | .. code-block:: q    |
    |                                |                                       |                              |                            |                      |
    | Convert matrix to vector       |    A(:)                               |    A = A.flatten()           |    A[:]                    |  ##code goes here    |
    +--------------------------------+---------------------------------------+------------------------------+----------------------------+----------------------+
    |                                | .. code-block:: matlab                | .. code-block:: python       | .. code-block:: julia      | .. code-block:: q    |
    |                                |                                       |                              |                            |                      |
    | Flip left/right                |    fliplr(A)                          |    np.fliplr(A)              |    reverse(A, dims = 2)    |  ##code goes here    |
    +--------------------------------+---------------------------------------+------------------------------+----------------------------+----------------------+
    |                                | .. code-block:: matlab                | .. code-block:: python       | .. code-block:: julia      | .. code-block:: q    |
    |                                |                                       |                              |                            |                      |
    | Flip up/down                   |    flipud(A)                          |    np.flipud(A)              |    reverse(A, dims = 1)    |  ##code goes here    |
    +--------------------------------+---------------------------------------+------------------------------+----------------------------+----------------------+
    |                                | .. code-block:: matlab                | .. code-block:: python       | .. code-block:: julia      | .. code-block:: q    |
    |                                |                                       |                              |                            |                      |
    | Repeat matrix (3 times in the  |    repmat(A, 3, 4)                    |    np.tile(A, (4, 3))        |    repeat(A, 3, 4)         |  ##code goes here    |
    | row dimension, 4 times in the  |                                       |                              |                            |                      |
    | column dimension)              |                                       |                              |                            |                      |
    +--------------------------------+---------------------------------------+------------------------------+----------------------------+----------------------+
    |                                | .. code-block:: matlab                | .. code-block:: python       | .. code-block:: julia      | .. code-block:: q    |
    |                                |                                       |                              |                            |                      |
    | Preallocating/Similar          |     x = rand(10)                      |    x = np.random.rand(3, 3)  |                            |  ##code goes here    |
    |                                |     y = zeros(size(x, 1), size(x, 2)) |    y = np.empty_like(x)      |     x = rand(3, 3)         |                      |
    |                                |                                       |                              |     y = similar(x)         |                      |
    |                                |                                       |    # new dims                |     # new dims             |                      |
    |                                |                                       |    y = np.empty((2, 3))      |     y = similar(x, 2, 2)   |                      |
    |                                | N/A similar type                      |                              |                            |                      |
    +--------------------------------+---------------------------------------+------------------------------+----------------------------+----------------------+
    |                                |                                       |                              |                            |                      |
    |                                | .. code-block:: matlab                | .. code-block:: python       |                            | .. code-block:: q    |
    |                                |                                       |                              |                            |                      |
    | Broadcast a function over a    |                                       |    def f(x):                 | .. code-block:: julia      |  ##code goes here    |
    | collection/matrix/vector       |                                       |        return x**2           |                            |                      |
    |                                |      f = @(x) x.^2                    |    def g(x, y):              |     f(x) = x^2             |                      |
    |                                |      g = @(x, y) x + 2 + y.^2         |        return x + 2 + y**2   |     g(x, y) = x + 2 + y^2  |                      |
    |                                |      x = 1:10                         |    x = np.arange(1, 10, 1)   |     x = 1:10               |                      |
    |                                |      y = 2:11                         |    y = np.arange(2, 11, 1)   |     y = 2:11               |                      |
    |                                |      f(x)                             |    f(x)                      |     f.(x)                  |                      |
    |                                |      g(x, y)                          |    g(x, y)                   |     g.(x, y)               |                      |
    |                                |                                       |                              |                            |                      |
    |                                | Functions broadcast directly          | Functions broadcast directly |                            |                      |
    +--------------------------------+---------------------------------------+------------------------------+----------------------------+----------------------+


Accessing Vector/Matrix Elements
--------------------------------

.. container:: multilang-table

    +--------------------------------+-------------------------------+-------------------------------+---------------------------+----------------------+
    | Operation                      |         MATLAB                | Python                        | Julia                     |         KDB+         |
    +================================+===============================+===============================+===========================+======================+
    |                                | .. code-block:: matlab        | .. code-block:: python        | .. code-block:: julia     | .. code-block:: q    |
    |                                |                               |                               |                           |                      |
    | Access one element             |     A(2, 2)                   |    A[1, 1]                    |     A[2, 2]               |  ##code goes here    |
    +--------------------------------+-------------------------------+-------------------------------+---------------------------+----------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python        | .. code-block:: julia     | .. code-block:: q    |
    |                                |                               |                               |                           |                      |
    | Access specific rows           |    A(1:4, :)                  |    A[0:4, :]                  |    A[1:4, :]              |  ##code goes here    |
    +--------------------------------+-------------------------------+-------------------------------+---------------------------+----------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python        | .. code-block:: julia     | .. code-block:: q    |
    |                                |                               |                               |                           |                      |
    | Access specific columns        |    A(:, 1:4)                  |    A[:, 0:4]                  |    A[:, 1:4]              |  ##code goes here    |
    +--------------------------------+-------------------------------+-------------------------------+---------------------------+----------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python        | .. code-block:: julia     | .. code-block:: q    |
    |                                |                               |                               |                           |                      |
    | Remove a row                   |    A([1 2 4], :)              |    A[[0, 1, 3], :]            |    A[[1, 2, 4], :]        |  ##code goes here    |
    +--------------------------------+-------------------------------+-------------------------------+---------------------------+----------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python        | .. code-block:: julia     | .. code-block:: q    |
    |                                |                               |                               |                           |                      |
    | Diagonals of matrix            |    diag(A)                    |    np.diag(A)                 |    diag(A)                |  ##code goes here    |
    +--------------------------------+-------------------------------+-------------------------------+---------------------------+----------------------+
    |                                | .. code-block:: matlab        | .. code-block:: python        | .. code-block:: julia     | .. code-block:: q    |
    |                                |                               |                               |                           |                      |
    | Get dimensions of matrix       |    [nrow ncol] = size(A)      |    nrow, ncol = np.shape(A)   |    nrow, ncol = size(A)   |  ##code goes here    |
    +--------------------------------+-------------------------------+-------------------------------+---------------------------+----------------------+



Mathematical Operations
-----------------------

.. container:: multilang-table

    +--------------------------------+-------------------------+---------------------------------------+------------------------+----------------------+
    |           Operation            |          MATLAB         |                 Python                |         Julia          |         KDB+         |
    +================================+=========================+=======================================+========================+======================+
    |                                | .. code-block:: matlab  | .. code-block:: python3               | .. code-block:: julia  | .. code-block:: q    |
    |                                |                         |                                       |                        |                      |
    | Dot product                    |     dot(A, B)           |    np.dot(A, B) or A @ B              |     dot(A, B)          |  ##code goes here    |
    |                                |                         |                                       |                        |                      |
    |                                |                         |                                       |     A ⋅ B # \cdot<TAB> |                      |
    +--------------------------------+-------------------------+---------------------------------------+------------------------+----------------------+
    |                                | .. code-block:: matlab  | .. code-block:: python3               | .. code-block:: julia  | .. code-block:: q    |
    |                                |                         |                                       |                        |                      |
    | Matrix multiplication          |     A * B               |     A @ B                             |     A * B              |  ##code goes here    |
    +--------------------------------+-------------------------+---------------------------------------+------------------------+----------------------+
    | Inplace matrix multiplication  | Not possible            | .. code-block:: python                | .. code-block:: julia  | .. code-block:: q    |
    |                                |                         |                                       |                        |                      |
    |                                |                         |    x = np.array([1, 2]).reshape(2, 1) |     x = [1, 2]         |  ##code goes here    |
    |                                |                         |    A = np.array(([1, 2], [3, 4]))     |     A = [1 2; 3 4]     |                      |
    |                                |                         |    y = np.empty_like(x)               |     y = similar(x)     |                      |
    |                                |                         |    np.matmul(A, x, y)                 |     mul!(y, A, x)      |                      |
    +--------------------------------+-------------------------+---------------------------------------+------------------------+----------------------+
    |                                | .. code-block:: matlab  | .. code-block:: python                | .. code-block:: julia  | .. code-block:: q    |
    |                                |                         |                                       |                        |                      |
    | Element-wise multiplication    |     A .* B              |    A * B                              |     A .* B             |  ##code goes here    |
    +--------------------------------+-------------------------+---------------------------------------+------------------------+----------------------+
    |                                | .. code-block:: matlab  | .. code-block:: python                | .. code-block:: julia  | .. code-block:: q    |
    |                                |                         |                                       |                        |                      |
    | Matrix to a power              |     A^2                 |    np.linalg.matrix_power(A, 2)       |     A^2                |  ##code goes here    |
    +--------------------------------+-------------------------+---------------------------------------+------------------------+----------------------+
    |                                | .. code-block:: matlab  | .. code-block:: python                | .. code-block:: julia  | .. code-block:: q    |
    |                                |                         |                                       |                        |                      |
    | Matrix to a power, elementwise |     A.^2                |    A**2                               |     A.^2               |  ##code goes here    |
    +--------------------------------+-------------------------+---------------------------------------+------------------------+----------------------+
    |                                | .. code-block:: matlab  | .. code-block:: python                | .. code-block:: julia  | .. code-block:: q    |
    |                                |                         |                                       |                        |                      |
    | Inverse                        |     inv(A)              |    np.linalg.inv(A)                   |     inv(A)             |  ##code goes here    |
    |                                |                         |                                       |                        |                      |
    |                                | or                      |                                       | or                     |                      |
    |                                |                         |                                       |                        |                      |
    |                                | .. code-block:: matlab  |                                       | .. code-block:: julia  |                      |
    |                                |                         |                                       |                        |                      |
    |                                |     A^(-1)              |                                       |    A^(-1)              |                      |
    +--------------------------------+-------------------------+---------------------------------------+------------------------+----------------------+
    |                                | .. code-block:: matlab  | .. code-block:: python                | .. code-block:: julia  | .. code-block:: q    |
    |                                |                         |                                       |                        |                      |
    | Determinant                    |     det(A)              |    np.linalg.det(A)                   |     det(A)             |  ##code goes here    |
    +--------------------------------+-------------------------+---------------------------------------+------------------------+----------------------+
    |                                | .. code-block:: matlab  | .. code-block:: python                | .. code-block:: julia  | .. code-block:: q    |
    |                                |                         |                                       |                        |                      |
    | Eigenvalues and eigenvectors   |     [vec, val] = eig(A) |    val, vec = np.linalg.eig(A)        |     val, vec = eigen(A)|  ##code goes here    |
    +--------------------------------+-------------------------+---------------------------------------+------------------------+----------------------+
    |                                | .. code-block:: matlab  | .. code-block:: python                | .. code-block:: julia  | .. code-block:: q    |
    |                                |                         |                                       |                        |                      |
    | Euclidean norm                 |     norm(A)             |    np.linalg.norm(A)                  |     norm(A)            |  ##code goes here    |
    +--------------------------------+-------------------------+---------------------------------------+------------------------+----------------------+
    |                                | .. code-block:: matlab  | .. code-block:: python                | .. code-block:: julia  | .. code-block:: q    |
    |                                |                         |                                       |                        |                      |
    | Solve linear system            |     A\b                 |    np.linalg.solve(A, b)              |     A\b                |  ##code goes here    |
    | :math:`Ax=b` (when :math:`A`   |                         |                                       |                        |                      |
    | is square)                     |                         |                                       |                        |                      |
    +--------------------------------+-------------------------+---------------------------------------+------------------------+----------------------+
    |                                | .. code-block:: matlab  | .. code-block:: python                | .. code-block:: julia  | .. code-block:: q    |
    |                                |                         |                                       |                        |                      |
    | Solve least squares problem    |     A\b                 |    np.linalg.lstsq(A, b)              |     A\b                |  ##code goes here    |
    | :math:`Ax=b` (when :math:`A`   |                         |                                       |                        |                      |
    | is rectangular)                |                         |                                       |                        |                      |
    +--------------------------------+-------------------------+---------------------------------------+------------------------+----------------------+


Sum / max / min
-------------------

.. container:: multilang-table

    +-----------------------------+------------------------+--------------------------------+-----------------------------------+----------------------+
    |          Operation          |         MATLAB         |             Python             |          Julia                    |         KDB+         |
    +=============================+========================+================================+===================================+======================+
    |                             | .. code-block:: matlab | .. code-block:: python         | .. code-block:: julia             | .. code-block:: q    |
    |                             |                        |                                |                                   |                      |
    | Sum / max / min of          |     sum(A, 1)          |    sum(A, 0)                   |     sum(A, dims = 1)              |  ##code goes here    |
    | each column                 |     max(A, [], 1)      |    np.amax(A, 0)               |     maximum(A, dims = 1)          |                      |
    |                             |     min(A, [], 1)      |    np.amin(A, 0)               |     minimum(A, dims = 1)          |                      |
    +-----------------------------+------------------------+--------------------------------+-----------------------------------+----------------------+
    |                             | .. code-block:: matlab | .. code-block:: python         | .. code-block:: julia             | .. code-block:: q    |
    |                             |                        |                                |                                   |                      |
    | Sum / max / min of each row |     sum(A, 2)          |    sum(A, 1)                   |     sum(A, dims = 2)              |  ##code goes here    |
    |                             |     max(A, [], 2)      |    np.amax(A, 1)               |     maximum(A, dims = 2)          |                      |
    |                             |     min(A, [], 2)      |    np.amin(A, 1)               |     minimum(A, dims = 2)          |                      |
    +-----------------------------+------------------------+--------------------------------+-----------------------------------+----------------------+
    |                             | .. code-block:: matlab | .. code-block:: python         | .. code-block:: julia             | .. code-block:: q    |
    |                             |                        |                                |                                   |                      |
    | Sum / max / min of          |     sum(A(:))          |    np.sum(A)                   |     sum(A)                        |  ##code goes here    |
    | entire matrix               |     max(A(:))          |    np.amax(A)                  |     maximum(A)                    |                      |
    |                             |     min(A(:))          |    np.amin(A)                  |     minimum(A)                    |                      |
    +-----------------------------+------------------------+--------------------------------+-----------------------------------+----------------------+
    |                             | .. code-block:: matlab | .. code-block:: python         | .. code-block:: julia             | .. code-block:: q    |
    |                             |                        |                                |                                   |                      |
    | Cumulative sum / max / min  |     cumsum(A, 1)       |    np.cumsum(A, 0)             |     cumsum(A, dims = 1)           |  ##code goes here    |
    | by row                      |     cummax(A, 1)       |    np.maximum.accumulate(A, 0) |     accumulate(max, A, dims = 1)  |                      |
    |                             |     cummin(A, 1)       |    np.minimum.accumulate(A, 0) |     accumulate(min, A, dims = 1)  |                      |
    +-----------------------------+------------------------+--------------------------------+-----------------------------------+----------------------+
    |                             | .. code-block:: matlab | .. code-block:: python         | .. code-block:: julia             | .. code-block:: q    |
    |                             |                        |                                |                                   |                      |
    | Cumulative sum / max / min  |     cumsum(A, 2)       |    np.cumsum(A, 1)             |     cumsum(A, dims = 2)           |  ##code goes here    |
    | by column                   |     cummax(A, 2)       |    np.maximum.accumulate(A, 1) |     accumulate(max, A, dims = 2)  |                      |
    |                             |     cummin(A, 2)       |    np.minimum.accumulate(A, 1) |     accumulate(min, A, dims = 2)  |                      |
    +-----------------------------+------------------------+--------------------------------+-----------------------------------+----------------------+



Programming
-----------

.. container:: multilang-table

    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+
    |        Operation        |                MATLAB                |                 Python                |             Julia             |         KDB+         |
    +=========================+======================================+=======================================+===============================+======================+
    |                         | .. code-block:: matlab               | .. code-block:: python                | .. code-block:: julia         | .. code-block:: q    |
    |                         |                                      |                                       |                               |                      |
    | Comment one line        |     % This is a comment              |    # This is a comment                |     # This is a comment       |  ##code goes here    |
    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+
    |                         | .. code-block:: matlab               | .. code-block:: python                | .. code-block:: julia         | .. code-block:: q    |
    |                         |                                      |                                       |                               |                      |
    | Comment block           |     %{                               |    # Block                            |     #=                        |  ##code goes here    |
    |                         |     Comment block                    |    # comment                          |     Comment block             |                      |
    |                         |     %}                               |    # following PEP8                   |     =#                        |                      |
    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+
    |                         | .. code-block:: matlab               | .. code-block:: python                | .. code-block:: julia         | .. code-block:: q    |                      
    |                         |                                      |                                       |                               |                      |
    | For loop                |     for i = 1:N                      |    for i in range(n):                 |     for i in 1:N              |  ##code goes here    |
    |                         |        % do something                |        # do something                 |        # do something         |                      |
    |                         |     end                              |                                       |     end                       |                      |
    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+
    |                         | .. code-block:: matlab               | .. code-block:: python                | .. code-block:: julia         | .. code-block:: q    |
    |                         |                                      |                                       |                               |                      |
    | While loop              |     while i <= N                     |    while i <= N:                      |     while i <= N              |  ##code goes here    |
    |                         |        % do something                |        # do something                 |        # do something         |                      |
    |                         |     end                              |                                       |     end                       |                      |
    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+
    |                         | .. code-block:: matlab               | .. code-block:: python                | .. code-block:: julia         | .. code-block:: q    |
    |                         |                                      |                                       |                               |                      |
    | If                      |     if i <= N                        |    if i <= N:                         |     if i <= N                 |  ##code goes here    |
    |                         |        % do something                |       # do something                  |        # do something         |                      |
    |                         |     end                              |                                       |     end                       |                      |
    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+
    |                         | .. code-block:: matlab               | .. code-block:: python                | .. code-block:: julia         | .. code-block:: q    |
    |                         |                                      |                                       |                               |                      |
    | If / else               |     if i <= N                        |   if i <= N:                          |    if i <= N                  |  ##code goes here    |
    |                         |        % do something                |       # do something                  |       # do something          |                      |
    |                         |     else                             |   else:                               |    else                       |                      |
    |                         |        % do something else           |       # so something else             |       # do something else     |                      |
    |                         |     end                              |                                       |    end                        |                      |
    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+
    |                         | .. code-block:: matlab               | .. code-block:: python                | .. code-block:: julia         | .. code-block:: q    |
    |                         |                                      |                                       |                               |                      |
    | Print text and variable |     x = 10                           |   x = 10                              |    x = 10                     |  ##code goes here    |
    |                         |     fprintf('x = %d \n', x)          |   print(f'x = {x}')                   |    println("x = $x")          |                      |
    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+
    |                         | .. code-block:: matlab               | .. code-block:: python                | .. code-block:: julia         | .. code-block:: q    |
    |                         |                                      |                                       |                               |                      |
    | Function: anonymous     |     f = @(x) x^2                     |    f = lambda x: x**2                 |     f = x -> x^2              |  ##code goes here    |
    |                         |                                      |                                       |     # can be rebound          |                      |
    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+
    |                         | .. code-block:: matlab               | .. code-block:: python                | .. code-block:: julia         | .. code-block:: q    |
    |                         |                                      |                                       |                               |                      |
    | Function                |     function out  = f(x)             |    def f(x):                          |     function f(x)             |  ##code goes here    |
    |                         |        out = x^2                     |        return x**2                    |        return x^2             |                      |
    |                         |     end                              |                                       |     end                       |                      |
    |                         |                                      |                                       |                               |                      |
    |                         |                                      |                                       |     f(x) = x^2 # not anon!    |                      |
    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+
    |                         |                                      |                                       |                               |                      |
    |                         |                                      |                                       |                               |                      |
    | Tuples                  |                                      | .. code-block:: python                | .. code-block:: julia         | .. code-block:: q    |
    |                         |  .. code-block:: matlab              |                                       |                               |                      |
    |                         |                                      |    t = (1, 2.0, "test")               |    t = (1, 2.0, "test")       |  ##code goes here    |
    |                         |     t = {1 2.0 "test"}               |    t[0]                               |    t[1]                       |                      |
    |                         |     t{1}                             |                                       |                               |                      |
    |                         |                                      |                                       |                               |                      |
    |                         |  Can use cells but watch performance |                                       |                               |                      |
    |                         |                                      |                                       |                               |                      |
    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+
    |                         |                                      |                                       |                               |                      |
    |                         |                                      |                                       |                               |                      |
    | Named Tuples/           |                                      | .. code-block:: python                | .. code-block:: julia         | .. code-block:: q    |
    | Anonymous Structures    |                                      |                                       |                               |                      |
    |                         | .. code-block:: matlab               |    from collections import namedtuple |    # vanilla                  |  ##code goes here    |
    |                         |                                      |                                       |    m = (x = 1, y = 2)         |                      |
    |                         |                                      |    mdef = namedtuple('m', 'x y')      |    m.x                        |                      |
    |                         |      m.x = 1                         |    m = mdef(1, 2)                     |                               |                      |
    |                         |      m.y = 2                         |                                       |    # constructor              |                      |
    |                         |                                      |    m.x                                |    using Parameters           |                      |
    |                         |      m.x                             |                                       |    mdef = @with_kw (x=1, y=2) |                      |
    |                         |                                      |                                       |    m = mdef() # same as above |                      |
    |                         |                                      |                                       |    m = mdef(x = 3)            |                      |
    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+
    |                         |                                      |                                       |                               |                      |
    |                         |                                      | .. code-block:: julia                 |                               |                      |
    | Closures                |  .. code-block:: matlab              |                                       | .. code-block:: julia         | .. code-block:: q    |
    |                         |                                      |    a = 2.0                            |                               |                      |
    |                         |     a = 2.0                          |    def f(x):                          |        a = 2.0                |  ##code goes here    |
    |                         |     f = @(x) a + x                   |        return a + x                   |        f(x) = a + x           |                      |
    |                         |     f(1.0)                           |    f(1.0)                             |        f(1.0)                 |                      |
    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+
    |                         |  No consistent or simple syntax      | .. code-block:: python                | .. code-block:: julia         | .. code-block:: q    |
    |                         |  to achieve |inplace-matlab|_        |                                       |                               |                      |
    | Inplace Modification    |                                      |    def f(x):                          |    function f!(out, x)        |  ##code goes here    |
    |                         |                                      |        x **=2                         |        out .= x.^2            |                      |
    |                         |                                      |        return                         |    end                        |                      |
    |                         |                                      |                                       |    x = rand(10)               |                      |
    |                         |                                      |    x = np.random.rand(10)             |    y = similar(x)             |                      |
    |                         |                                      |    f(x)                               |    f!(y, x)                   |                      |
    +-------------------------+--------------------------------------+---------------------------------------+-------------------------------+----------------------+

.. |inplace-matlab| replace:: this
.. _inplace-matlab: https://blogs.mathworks.com/loren/2007/03/22/in-place-operations-on-data/