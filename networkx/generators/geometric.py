# -*- coding: utf-8 -*-
#    Copyright (C) 2004-2017 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
#
# Authors: Aric Hagberg (hagberg@lanl.gov)
#          Dan Schult (dschult@colgate.edu)
#          Ben Edwards (BJEdwards@gmail.com)
#          Arya McCarthy (admccarthy@smu.edu)
"""Generators for geometric graphs.
"""
from __future__ import division

from bisect import bisect_left
from itertools import combinations
from itertools import product
from math import sqrt
import math
import random
from random import uniform
try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    _is_scipy_available = False
else:
    _is_scipy_available = True

import networkx as nx
from networkx.utils import nodes_or_number

__all__ = ['geographical_threshold_graph', 'waxman_graph',
           'navigable_small_world_graph', 'random_geometric_graph']


def euclidean(x, y):
    """Returns the Euclidean distance between the vectors ``x`` and ``y``.

    Each of ``x`` and ``y`` can be any iterable of numbers. The
    iterables must be of the same length.

    """
    return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


def _fast_construct_edges(G, radius, p):
    """Construct edges for random geometric graph.

    Requires scipy to be installed.
    """
    pos = nx.get_node_attributes(G, 'pos')
    nodes, coords = list(zip(*pos.items()))
    kdtree = KDTree(coords)  # Cannot provide generator.
    edge_indexes = kdtree.query_pairs(radius, p)
    edges = ((nodes[u], nodes[v]) for u, v in edge_indexes)
    G.add_edges_from(edges)


def _slow_construct_edges(G, radius, p):
    """Construct edges for random geometric graph.

    Works without scipy, but in `O(n^2)` time.
    """
    # TODO This can be parallelized.
    for (u, pu), (v, pv) in combinations(G.nodes(data='pos'), 2):
        if sum(abs(a - b) ** p for a, b in zip(pu, pv)) <= radius ** p:
            G.add_edge(u, v)


@nodes_or_number(0)
def random_geometric_graph(n, radius, dim=2, pos=None, p=2):
    """Returns a random geometric graph in the unit cube.

    The random geometric graph model places `n` nodes uniformly at
    random in the unit cube. Two nodes are joined by an edge if the
    distance between the nodes is at most `radius`.

    Edges are determined using a KDTree when SciPy is available.
    This reduces the time complexity from $O(n^2)$ to $O(n)$.

    Parameters
    ----------
    n : int or iterable
        Number of nodes or iterable of nodes
    radius: float
        Distance threshold value
    dim : int, optional
        Dimension of graph
    pos : dict, optional
        A dictionary keyed by node with node positions as values.
    p : float
        Which Minkowski distance metric to use.  `p` has to meet the condition
        ``1 <= p <= infinity``.

        If this argument is not specified, the $L^2$ metric
        (the Euclidean distance metric) is used.

        This should not be confused with the `p` of an Erdős-Rényi random
        graph, which represents probability.

    Returns
    -------
    Graph
        A random geometric graph, undirected and without self-loops.
        Each node has a node attribute ``'pos'`` that stores the
        position of that node in Euclidean space as provided by the
        ``pos`` keyword argument or, if ``pos`` was not provided, as
        generated by this function.

    Examples
    --------
    Create a random geometric graph on twenty nodes where nodes are joined by
    an edge if their distance is at most 0.1::

    >>> G = nx.random_geometric_graph(20, 0.1)

    Notes
    -----
    This uses a *k*-d tree to build the graph.

    The `pos` keyword argument can be used to specify node positions so you
    can create an arbitrary distribution and domain for positions.

    For example, to use a 2D Gaussian distribution of node positions with mean
    (0, 0) and standard deviation 2::

    >>> import random
    >>> n = 20
    >>> p = {i: (random.gauss(0, 2), random.gauss(0, 2)) for i in range(n)}
    >>> G = nx.random_geometric_graph(n, 0.2, pos=p)

    References
    ----------
    .. [1] Penrose, Mathew, *Random Geometric Graphs*,
           Oxford Studies in Probability, 5, 2003.

    """
    # TODO Is this function just a special case of the geographical
    # threshold graph?
    #
    #     n_name, nodes = n
    #     half_radius = {v: radius / 2 for v in nodes}
    #     return geographical_threshold_graph(nodes, theta=1, alpha=1,
    #                                         weight=half_radius)
    #
    n_name, nodes = n
    G = nx.Graph()
    G.add_nodes_from(nodes)
    # If no positions are provided, choose uniformly random vectors in
    # Euclidean space of the specified dimension.
    if pos is None:
        pos = {v: [random.random() for i in range(dim)] for v in nodes}
    nx.set_node_attributes(G, pos, 'pos')

    if _is_scipy_available:
        _fast_construct_edges(G, radius, p)
    else:
        _slow_construct_edges(G, radius, p)

    return G


@nodes_or_number(0)
def geographical_threshold_graph(n, theta, alpha=2, dim=2, pos=None,
                                 weight=None, metric=None):
    r"""Returns a geographical threshold graph.

    The geographical threshold graph model places $n$ nodes uniformly at
    random in a rectangular domain.  Each node $u$ is assigned a weight
    $w_u$. Two nodes $u$ and $v$ are joined by an edge if

    .. math::

       w_u + w_v \ge \theta r^{\alpha}

    where $r$ is the distance between $u$ and $v$, and $\theta$,
    $\alpha$ are parameters.

    Parameters
    ----------
    n : int or iterable
        Number of nodes or iterable of nodes
    theta: float
        Threshold value
    alpha: float, optional
        Exponent of distance function
    dim : int, optional
        Dimension of graph
    pos : dict
        Node positions as a dictionary of tuples keyed by node.
    weight : dict
        Node weights as a dictionary of numbers keyed by node.
    metric : function
        A metric on vectors of numbers (represented as lists or
        tuples). This must be a function that accepts two lists (or
        tuples) as input and yields a number as output. The function
        must also satisfy the four requirements of a `metric`_.
        Specifically, if *d* is the function and *x*, *y*,
        and *z* are vectors in the graph, then *d* must satisfy

        1. *d*(*x*, *y*) ≥ 0,
        2. *d*(*x*, *y*) = 0 if and only if *x* = *y*,
        3. *d*(*x*, *y*) = *d*(*y*, *x*),
        4. *d*(*x*, *z*) ≤ *d*(*x*, *y*) + *d*(*y*, *z*).

        If this argument is not specified, the Euclidean distance metric is
        used.

        .. _metric: https://en.wikipedia.org/wiki/Metric_%28mathematics%29

    Returns
    -------
    Graph
        A random geographic threshold graph, undirected and without
        self-loops.

        Each node has a node attribute ``pos`` that stores the
        position of that node in Euclidean space as provided by the
        ``pos`` keyword argument or, if ``pos`` was not provided, as
        generated by this function. Similarly, each node has a node
        attribute ``weight`` that stores the weight of that node as
        provided or as generated.

    Examples
    --------
    Specify an alternate distance metric using the ``metric`` keyword
    argument. For example, to use the `taxicab metric`_ instead of the
    default `Euclidean metric`_::

        >>> dist = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
        >>> G = nx.geographical_threshold_graph(10, 0.1, metric=dist)

    .. _taxicab metric: https://en.wikipedia.org/wiki/Taxicab_geometry
    .. _Euclidean metric: https://en.wikipedia.org/wiki/Euclidean_distance

    Notes
    -----
    If weights are not specified they are assigned to nodes by drawing randomly
    from the exponential distribution with rate parameter $\lambda=1$.
    To specify weights from a different distribution, use the `weight` keyword
    argument::

    >>> import random
    >>> n = 20
    >>> w = {i: random.expovariate(5.0) for i in range(n)}
    >>> G = nx.geographical_threshold_graph(20, 50, weight=w)

    If node positions are not specified they are randomly assigned from the
    uniform distribution.

    References
    ----------
    .. [1] Masuda, N., Miwa, H., Konno, N.:
       Geographical threshold graphs with small-world and scale-free
       properties.
       Physical Review E 71, 036108 (2005)
    .. [2]  Milan Bradonjić, Aric Hagberg and Allon G. Percus,
       Giant component and connectivity in geographical threshold graphs,
       in Algorithms and Models for the Web-Graph (WAW 2007),
       Antony Bonato and Fan Chung (Eds), pp. 209--216, 2007
    """
    n_name, nodes = n
    G = nx.Graph()
    G.add_nodes_from(nodes)
    # If no weights are provided, choose them from an exponential
    # distribution.
    if weight is None:
        weight = {v: random.expovariate(1) for v in G}
    # If no positions are provided, choose uniformly random vectors in
    # Euclidean space of the specified dimension.
    if pos is None:
        pos = {v: [random.random() for i in range(dim)] for v in nodes}
    # If no distance metric is provided, use Euclidean distance.
    if metric is None:
        metric = euclidean
    nx.set_node_attributes(G, weight, 'weight')
    nx.set_node_attributes(G, pos, 'pos')

    # Returns ``True`` if and only if the nodes whose attributes are
    # ``du`` and ``dv`` should be joined, according to the threshold
    # condition.
    def should_join(pair):
        u, v = pair
        u_pos, v_pos = pos[u], pos[v]
        u_weight, v_weight = weight[u], weight[v]
        return theta * metric(u_pos, v_pos) ** alpha <= u_weight + v_weight

    G.add_edges_from(filter(should_join, combinations(G, 2)))
    return G


@nodes_or_number(0)
def waxman_graph(n, beta=0.4, alpha=0.1, L=None, domain=(0, 0, 1, 1),
                 metric=None):
    r"""Return a Waxman random graph.

    The Waxman random graph model places `n` nodes uniformly at random
    in a rectangular domain. Each pair of nodes at distance `d` is
    joined by an edge with probability

    .. math::
            p = \beta \exp(-d / \alpha L).

    This function implements both Waxman models, using the `L` keyword
    argument.

    * Waxman-1: if `L` is not specified, it is set to be the maximum distance
      between any pair of nodes.
    * Waxman-2: if `L` is specified, the distance between a pair of nodes is
      chosen uniformly at random from the interval `[0, L]`.

    Parameters
    ----------
    n : int or iterable
        Number of nodes or iterable of nodes
    beta: float
        Model parameter
    alpha: float
        Model parameter
    L : float, optional
        Maximum distance between nodes.  If not specified, the actual distance
        is calculated.
    domain : four-tuple of numbers, optional
        Domain size, given as a tuple of the form `(x_min, y_min, x_max,
        y_max)`.
    metric : function
        A metric on vectors of numbers (represented as lists or
        tuples). This must be a function that accepts two lists (or
        tuples) as input and yields a number as output. The function
        must also satisfy the four requirements of a `metric`_.
        Specifically, if *d* is the function and *x*, *y*,
        and *z* are vectors in the graph, then *d* must satisfy

        1. *d*(*x*, *y*) ≥ 0,
        2. *d*(*x*, *y*) = 0 if and only if *x* = *y*,
        3. *d*(*x*, *y*) = *d*(*y*, *x*),
        4. *d*(*x*, *z*) ≤ *d*(*x*, *y*) + *d*(*y*, *z*).

        If this argument is not specified, the Euclidean distance metric is
        used.

        .. _metric: https://en.wikipedia.org/wiki/Metric_%28mathematics%29

    Returns
    -------
    Graph
        A random Waxman graph, undirected and without self-loops. Each
        node has a node attribute ``'pos'`` that stores the position of
        that node in Euclidean space as generated by this function.

    Examples
    --------
    Specify an alternate distance metric using the ``metric`` keyword
    argument. For example, to use the "`taxicab metric`_" instead of the
    default `Euclidean metric`_::

        >>> dist = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
        >>> G = nx.waxman_graph(10, 0.5, 0.1, metric=dist)

    .. _taxicab metric: https://en.wikipedia.org/wiki/Taxicab_geometry
    .. _Euclidean metric: https://en.wikipedia.org/wiki/Euclidean_distance

    Notes
    -----
    Starting in NetworkX 2.0 the parameters alpha and beta align with their
    usual roles in the probability distribution. In earlier versions their
    positions in the expresssion were reversed. Their position in the calling
    sequence reversed as well to minimize backward incompatibility.

    References
    ----------
    .. [1]  B. M. Waxman, *Routing of multipoint connections*.
       IEEE J. Select. Areas Commun. 6(9),(1988) 1617--1622.
    """
    n_name, nodes = n
    G = nx.Graph()
    G.add_nodes_from(nodes)
    (xmin, ymin, xmax, ymax) = domain
    # Each node gets a uniformly random position in the given rectangle.
    pos = {v: (uniform(xmin, xmax), uniform(ymin, ymax)) for v in G}
    nx.set_node_attributes(G, pos, 'pos')
    # If no distance metric is provided, use Euclidean distance.
    if metric is None:
        metric = euclidean
    # If the maximum distance L is not specified (that is, we are in the
    # Waxman-1 model), then find the maximum distance between any pair
    # of nodes.
    #
    # In the Waxman-1 model, join nodes randomly based on distance. In
    # the Waxman-2 model, join randomly based on random l.
    if L is None:
        L = max(metric(x, y) for x, y in combinations(pos.values(), 2))
        dist = lambda u, v: metric(pos[u], pos[v])
    else:
        dist = lambda u, v: random.random() * L

    # `pair` is the pair of nodes to decide whether to join.
    def should_join(pair):
        return random.random() < beta * math.exp(-dist(*pair) / (alpha * L))

    G.add_edges_from(filter(should_join, combinations(G, 2)))
    return G


def navigable_small_world_graph(n, p=1, q=1, r=2, dim=2, seed=None):
    r"""Return a navigable small-world graph.

    A navigable small-world graph is a directed grid with additional long-range
    connections that are chosen randomly.

      [...] we begin with a set of nodes [...] that are identified with the set
      of lattice points in an $n \times n$ square,
      $\{(i, j): i \in \{1, 2, \ldots, n\}, j \in \{1, 2, \ldots, n\}\}$,
      and we define the *lattice distance* between two nodes $(i, j)$ and
      $(k, l)$ to be the number of "lattice steps" separating them:
      $d((i, j), (k, l)) = |k - i| + |l - j|$.

      For a universal constant $p >= 1$, the node $u$ has a directed edge to
      every other node within lattice distance $p$---these are its *local
      contacts*. For universal constants $q >= 0$ and $r >= 0$ we also
      construct directed edges from $u$ to $q$ other nodes (the *long-range
      contacts*) using independent random trials; the $i$th directed edge from
      $u$ has endpoint $v$ with probability proportional to $[d(u,v)]^{-r}$.

      -- [1]_

    Parameters
    ----------
    n : int
        The length of one side of the lattice; the number of nodes in
        the graph is therefore $n^2$.
    p : int
        The diameter of short range connections. Each node is joined with every
        other node within this lattice distance.
    q : int
        The number of long-range connections for each node.
    r : float
        Exponent for decaying probability of connections.  The probability of
        connecting to a node at lattice distance $d$ is $1/d^r$.
    dim : int
        Dimension of grid
    seed : int, optional
        Seed for random number generator (default=None).

    References
    ----------
    .. [1] J. Kleinberg. The small-world phenomenon: An algorithmic
       perspective. Proc. 32nd ACM Symposium on Theory of Computing, 2000.
    """
    if (p < 1):
        raise nx.NetworkXException("p must be >= 1")
    if (q < 0):
        raise nx.NetworkXException("q must be >= 0")
    if (r < 0):
        raise nx.NetworkXException("r must be >= 1")
    if seed is not None:
        random.seed(seed)
    G = nx.DiGraph()
    nodes = list(product(range(n), repeat=dim))
    for p1 in nodes:
        probs = [0]
        for p2 in nodes:
            if p1 == p2:
                continue
            d = sum((abs(b - a) for a, b in zip(p1, p2)))
            if d <= p:
                G.add_edge(p1, p2)
            probs.append(d**-r)
        cdf = list(nx.utils.accumulate(probs))
        for _ in range(q):
            target = nodes[bisect_left(cdf, random.uniform(0, cdf[-1]))]
            G.add_edge(p1, target)
    return G
