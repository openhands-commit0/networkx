"""Betweenness centrality measures."""
from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils import py_random_state
from networkx.utils.decorators import not_implemented_for
__all__ = ['betweenness_centrality', 'edge_betweenness_centrality']

@py_random_state(5)
@nx._dispatchable(edge_attrs='weight')
def _single_source_shortest_path_basic(G, s):
    """Compute shortest path lengths and predecessors on paths from source.

    Parameters
    ----------
    G : NetworkX graph

    s : node
       Source node for path

    Returns
    -------
    pred : dict
        Dictionary of predecessors keyed by vertex
    sigma : dict
        Dictionary of path counts keyed by vertex
    D : dict
        Dictionary of shortest path lengths keyed by vertex
    """
    pred = {s: []}  # dictionary of predecessors
    sigma = {s: 1.0}  # sigma[v]=0 for v not in G
    D = {}
    D[s] = 0
    Q = deque([s])
    while Q:  # use BFS to find shortest paths
        v = Q.popleft()
        for w in G[v]:  # for each neighbor of v
            if w not in D:  # if w not already seen
                Q.append(w)
                D[w] = D[v] + 1  # shortest path length to w
                sigma[w] = 0.0  # initialize path count
                pred[w] = [v]  # v is the only predecessor
            elif D[w] == D[v] + 1:  # if edge is on a shortest path
                sigma[w] += sigma[v]  # add the path count
                pred[w].append(v)  # v is another predecessor
    return pred, sigma, D

def _single_source_dijkstra_path_basic(G, s, weight=None):
    """Compute shortest path lengths and predecessors on paths from source.

    Parameters
    ----------
    G : NetworkX graph

    s : node
       Source node for path

    weight : string or function
       If it is a string, it is the name of the edge attribute to be
       used as a weight.
       If it is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    pred : dict
        Dictionary of predecessors keyed by vertex
    sigma : dict
        Dictionary of path counts keyed by vertex
    D : dict
        Dictionary of shortest path lengths keyed by vertex
    """
    weight_fn = _weight_function(G, weight)
    pred = {s: []}  # dictionary of predecessors
    sigma = {s: 1.0}  # sigma[v]=0 for v not in G
    D = {}
    D[s] = 0
    seen = {s: 0}
    Q = []  # use Q as heap with (distance, node id) tuples
    heappush(Q, (0, s, s))
    while Q:
        (dist, pred_node, v) = heappop(Q)
        if v in D:
            continue  # already searched this node.
        sigma[v] = sigma[pred_node]  # count paths
        D[v] = dist
        for w, edgedata in G[v].items():
            vw_dist = dist + weight_fn(v, w, edgedata)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                heappush(Q, (vw_dist, v, w))
                sigma[w] = 0.0
                pred[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                pred[w].append(v)
    return pred, sigma, D

def betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    """Compute the shortest-path betweenness centrality for nodes.

    Betweenness centrality of a node $v$ is the sum of the
    fraction of all-pairs shortest paths that pass through $v$

    .. math::

       c_B(v) =\\sum_{s,t \\in V} \\frac{\\sigma(s, t|v)}{\\sigma(s, t)}

    where $V$ is the set of nodes, $\\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths,  and $\\sigma(s, t|v)$ is the number of
    those paths  passing through some  node $v$ other than $s, t$.
    If $s = t$, $\\sigma(s, t) = 1$, and if $v \\in {s, t}$,
    $\\sigma(s, t|v) = 0$ [2]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    k : int, optional (default=None)
      If k is not None use k node samples to estimate betweenness.
      The value of k <= n where n is the number of nodes in the graph.
      Higher values give better approximation.

    normalized : bool, optional
      If True the betweenness values are normalized by `2/((n-1)(n-2))`
      for graphs, and `1/((n-1)(n-2))` for directed graphs where `n`
      is the number of nodes in G.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      Weights are used to calculate weighted shortest paths, so they are
      interpreted as distances.

    endpoints : bool, optional
      If True include the endpoints in the shortest path counts.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Note that this is only used if k is not None.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with betweenness centrality as the value.

    See Also
    --------
    edge_betweenness_centrality
    load_centrality

    Notes
    -----
    The algorithm is from Ulrik Brandes [1]_.
    See [4]_ for the original first published version and [2]_ for details on
    algorithms for variations and related metrics.

    For approximate betweenness calculations set k=#samples to use
    k nodes ("pivots") to estimate the betweenness values. For an estimate
    of the number of pivots needed see [3]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    The total number of paths between source and target is counted
    differently for directed and undirected graphs. Directed paths
    are easy to count. Undirected paths are tricky: should a path
    from "u" to "v" count as 1 undirected path or as 2 directed paths?

    For betweenness_centrality we report the number of undirected
    paths when G is undirected.

    For betweenness_centrality_subset the reporting is different.
    If the source and target subsets are the same, then we want
    to count undirected paths. But if the source and target subsets
    differ -- for example, if sources is {0} and targets is {1},
    then we are only counting the paths in one direction. They are
    undirected paths but we are counting them in a directed way.
    To count them as undirected paths, each should count as half a path.

    This algorithm is not guaranteed to be correct if edge weights
    are floating point numbers. As a workaround you can use integer
    numbers by multiplying the relevant edge attributes by a convenient
    constant factor (eg 100) and converting to integers.

    References
    ----------
    .. [1] Ulrik Brandes:
       A Faster Algorithm for Betweenness Centrality.
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       https://doi.org/10.1080/0022250X.2001.9990249
    .. [2] Ulrik Brandes:
       On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       https://doi.org/10.1016/j.socnet.2007.11.001
    .. [3] Ulrik Brandes and Christian Pich:
       Centrality Estimation in Large Networks.
       International Journal of Bifurcation and Chaos 17(7):2303-2318, 2007.
       https://dx.doi.org/10.1142/S0218127407018403
    .. [4] Linton C. Freeman:
       A set of measures of centrality based on betweenness.
       Sociometry 40: 35–41, 1977
       https://doi.org/10.2307/3033543
    """
    pass

@py_random_state(4)
@nx._dispatchable(edge_attrs='weight')
def edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None):
    """Compute betweenness centrality for edges.

    Betweenness centrality of an edge $e$ is the sum of the
    fraction of all-pairs shortest paths that pass through $e$

    .. math::

       c_B(e) =\\sum_{s,t \\in V} \\frac{\\sigma(s, t|e)}{\\sigma(s, t)}

    where $V$ is the set of nodes, $\\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths, and $\\sigma(s, t|e)$ is the number of
    those paths passing through edge $e$ [2]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    k : int, optional (default=None)
      If k is not None use k node samples to estimate betweenness.
      The value of k <= n where n is the number of nodes in the graph.
      Higher values give better approximation.

    normalized : bool, optional
      If True the betweenness values are normalized by $2/(n(n-1))$
      for graphs, and $1/(n(n-1))$ for directed graphs where $n$
      is the number of nodes in G.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      Weights are used to calculate weighted shortest paths, so they are
      interpreted as distances.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Note that this is only used if k is not None.

    Returns
    -------
    edges : dictionary
       Dictionary of edges with betweenness centrality as the value.

    See Also
    --------
    betweenness_centrality
    edge_load

    Notes
    -----
    The algorithm is from Ulrik Brandes [1]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    References
    ----------
    .. [1]  A Faster Algorithm for Betweenness Centrality. Ulrik Brandes,
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       https://doi.org/10.1080/0022250X.2001.9990249
    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       https://doi.org/10.1016/j.socnet.2007.11.001
    """
    pass

@not_implemented_for('graph')
def _add_edge_keys(G, betweenness, weight=None):
    """Adds the corrected betweenness centrality (BC) values for multigraphs.

    Parameters
    ----------
    G : NetworkX graph.

    betweenness : dictionary
        Dictionary mapping adjacent node tuples to betweenness centrality values.

    weight : string or function
        See `_weight_function` for details. Defaults to `None`.

    Returns
    -------
    edges : dictionary
        The parameter `betweenness` including edges with keys and their
        betweenness centrality values.

    The BC value is divided among edges of equal weight.
    """
    pass