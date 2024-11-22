import networkx as nx

def flow_matrix_row(G, weight='weight', dtype=None, solver='lu'):
    """Returns a function that solves the current-flow betweenness equation.

    Returns a function that solves the current-flow betweenness equation
    for a row of the Laplacian matrix.

    Parameters
    ----------
    G : NetworkX graph
      The algorithm works for all types of graphs, including directed
      graphs and multigraphs.

    weight : string or function
      Key for edge data used as the edge weight.
      If None, then use 1 as each edge weight.
      The weight function can be a function that takes (edge) as input.
      If it is a string then use this key to obtain the weight value.

    dtype : data-type, optional (default=None)
      Default data type for internal matrices.
      Set to np.float32 for lower memory consumption.

    solver : string (default='lu')
        Type of linear solver to use for computing the flow matrix.
        Options are "full" (uses most memory), "lu" (recommended), and
        "cg" (uses least memory).

    Returns
    -------
    solver : function
        Function that solves the current-flow betweenness equation
        for a row of the Laplacian matrix.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> solver = flow_matrix_row(G)
    >>> for row in range(G.number_of_nodes()):
    ...     x = solver(row)
    ...     # do something with row of current-flow matrix

    Notes
    -----
    The solvers use different methods to compute the flow matrix.
    The full solver uses the full inverse of the Laplacian matrix,
    the LU solver uses LU decomposition, and the CG solver uses the
    conjugate gradient method.

    See Also
    --------
    current_flow_betweenness_centrality
    edge_current_flow_betweenness_centrality
    approximate_current_flow_betweenness_centrality
    """
    import numpy as np
    L = nx.laplacian_matrix(G, weight=weight, dtype=dtype).todense()
    n = G.number_of_nodes()
    if solver == 'full':
        # Use full matrix inverse
        IL = FullInverseLaplacian(L, width=1, dtype=dtype)
    elif solver == 'lu':
        # Use sparse LU decomposition
        IL = SuperLUInverseLaplacian(L, width=1, dtype=dtype)
    elif solver == 'cg':
        # Use conjugate gradient
        IL = CGInverseLaplacian(L, width=1, dtype=dtype)
    else:
        raise nx.NetworkXError(f"Unknown solver {solver}")

    def solve_row(r):
        rhs = np.zeros(IL.n, dtype=dtype)
        rhs[r] = 1
        return IL.solve(rhs)

    return solve_row

class InverseLaplacian:
    """Base class for computing the inverse of the Laplacian matrix."""

    def __init__(self, L, width=None, dtype=None):
        """Initialize the InverseLaplacian solver.

        Parameters
        ----------
        L : NumPy matrix
            Laplacian of the graph.

        width : integer, optional
            Width of the solver.

        dtype : NumPy data type, optional
            Data type of the solver.
        """
        global np
        import numpy as np
        n, n = L.shape
        self.dtype = dtype
        self.n = n
        if width is None:
            self.w = self.width(L)
        else:
            self.w = width
        self.C = np.zeros((self.w, n), dtype=dtype)
        self.L1 = L[1:, 1:]
        self.init_solver(L)

    def init_solver(self, L):
        """Initialize the solver with the Laplacian matrix.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("init_solver not implemented")

    def solve(self, rhs):
        """Solve the system Lx = b.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("solve not implemented")

    def width(self, L):
        """Compute the width of the inverse Laplacian.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("width not implemented")

class FullInverseLaplacian(InverseLaplacian):
    """Inverse Laplacian using full matrix inverse."""

    def init_solver(self, L):
        """Initialize the solver with the Laplacian matrix."""
        self.IL = np.zeros((self.n, self.n), dtype=self.dtype)
        self.IL[1:, 1:] = np.linalg.inv(self.L1)
        self.IL[0, 0] = 1.0 / self.n
        self.IL[0, 1:] = -1.0 / self.n
        self.IL[1:, 0] = -1.0 / self.n

    def solve(self, rhs):
        """Solve the system Lx = b."""
        return np.dot(self.IL, rhs)

    def width(self, L):
        """Compute the width of the inverse Laplacian."""
        return 1

class SuperLUInverseLaplacian(InverseLaplacian):
    """Inverse Laplacian using LU decomposition."""

    def init_solver(self, L):
        """Initialize the solver with the Laplacian matrix."""
        from scipy.sparse.linalg import splu
        self.lu = splu(self.L1.tocsc(), permc_spec='MMD_AT_PLUS_A')

    def solve(self, rhs):
        """Solve the system Lx = b."""
        x = np.zeros(self.n, dtype=self.dtype)
        x[1:] = self.lu.solve(rhs[1:])
        x[0] = -sum(x[1:])
        return x

    def width(self, L):
        """Compute the width of the inverse Laplacian."""
        return 1

class CGInverseLaplacian(InverseLaplacian):
    """Inverse Laplacian using conjugate gradient method."""

    def init_solver(self, L):
        """Initialize the solver with the Laplacian matrix."""
        from scipy.sparse.linalg import cg
        self.cg = cg

    def solve(self, rhs):
        """Solve the system Lx = b."""
        x = np.zeros(self.n, dtype=self.dtype)
        x[1:], info = self.cg(self.L1, rhs[1:])
        if info != 0:
            raise nx.NetworkXError("Conjugate gradient solver did not converge.")
        x[0] = -sum(x[1:])
        return x

    def width(self, L):
        """Compute the width of the inverse Laplacian."""
        return 1