"""
NetworkX utilizes a plugin-dispatch architecture, which means we can plug in and
out of backends with minimal code changes. A valid NetworkX backend specifies
`entry points <https://packaging.python.org/en/latest/specifications/entry-points>`_,
named ``networkx.backends`` and an optional ``networkx.backend_info`` when it is
installed (not imported). This allows NetworkX to dispatch (redirect) function calls
to the backend so the execution flows to the designated backend
implementation, similar to how plugging a charger into a socket redirects the
electricity to your phone. This design enhances flexibility and integration, making
NetworkX more adaptable and efficient. 

There are three main ways to use a backend after the package is installed.
You can set environment variables and run the exact same code you run for
NetworkX. You can use a keyword argument ``backend=...`` with the NetworkX
function. Or, you can convert the NetworkX Graph to a backend graph type and
call a NetworkX function supported by that backend. Environment variables
and backend keywords automatically convert your NetworkX Graph to the
backend type. Manually converting it yourself allows you to use that same
backend graph for more than one function call, reducing conversion time.

For example, you can set an environment variable before starting python to request
all dispatchable functions automatically dispatch to the given backend::

    bash> NETWORKX_AUTOMATIC_BACKENDS=cugraph python my_networkx_script.py

or you can specify the backend as a kwarg::

    nx.betweenness_centrality(G, k=10, backend="parallel")

or you can convert the NetworkX Graph object ``G`` into a Graph-like
object specific to the backend and then pass that in the NetworkX function::

    H = nx_parallel.ParallelGraph(G)
    nx.betweenness_centrality(H, k=10)

How it works: You might have seen the ``@nx._dispatchable`` decorator on
many of the NetworkX functions in the codebase. It decorates the function
with code that redirects execution to the function's backend implementation.
The code also manages any ``backend_kwargs`` you provide to the backend
version of the function. The code looks for the environment variable or
a ``backend`` keyword argument and if found, converts the input NetworkX
graph to the backend format before calling the backend's version of the
function. If no environment variable or backend keyword are found, the
dispatching code checks the input graph object for an attribute
called ``__networkx_backend__`` which tells it which backend provides this
graph type. That backend's version of the function is then called.
The backend system relies on Python ``entry_point`` system to signal
NetworkX that a backend is installed (even if not imported yet). Thus no
code needs to be changed between running with NetworkX and running with
a backend to NetworkX. The attribute ``__networkx_backend__`` holds a
string with the name of the ``entry_point``. If none of these options
are being used, the decorator code simply calls the NetworkX function
on the NetworkX graph as usual.

The NetworkX library does not need to know that a backend exists for it
to work. So long as the backend package creates the entry_point, and
provides the correct interface, it will be called when the user requests
it using one of the three approaches described above. Some backends have
been working with the NetworkX developers to ensure smooth operation.
They are the following::

- `graphblas <https://github.com/python-graphblas/graphblas-algorithms>`_
- `cugraph <https://github.com/rapidsai/cugraph/tree/branch-24.04/python/nx-cugraph>`_
- `parallel <https://github.com/networkx/nx-parallel>`_
- ``loopback`` is for testing purposes only and is not a real backend.

Note that the ``backend_name`` is e.g. ``parallel``, the package installed
is ``nx-parallel``, and we use ``nx_parallel`` while importing the package.

Creating a Custom backend
-------------------------

1.  To be a valid backend that is discoverable by NetworkX, your package must
    register an `entry-point <https://packaging.python.org/en/latest/specifications/entry-points/#entry-points>`_
    ``networkx.backends`` in the package's metadata, with a `key pointing to your
    dispatch object <https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata>`_ .
    For example, if you are using ``setuptools`` to manage your backend package,
    you can `add the following to your pyproject.toml file <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_::

        [project.entry-points."networkx.backends"]
        backend_name = "your_dispatcher_class"

    You can also add the ``backend_info`` entry-point. It points towards the ``get_info``
    function that returns all the backend information, which is then used to build the
    "Additional Backend Implementation" box at the end of algorithm's documentation
    page (e.g. `nx-cugraph's get_info function <https://github.com/rapidsai/cugraph/blob/branch-24.04/python/nx-cugraph/_nx_cugraph/__init__.py>`_)::

        [project.entry-points."networkx.backend_info"]
        backend_name = "your_get_info_function"

    Note that this would only work if your backend is a trusted backend of NetworkX,
    and is present in the `.circleci/config.yml` and
    `.github/workflows/deploy-docs.yml` files in the NetworkX repository.

2.  The backend must create an ``nx.Graph``-like object which contains an attribute
    ``__networkx_backend__`` with a value of the entry point name::

        class BackendGraph:
            __networkx_backend__ = "backend_name"
            ...


Testing the Custom backend
--------------------------

To test your custom backend, you can run the NetworkX test suite with your backend.
This also ensures that the custom backend is compatible with NetworkX's API.

Testing Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~

To enable automatic testing with your custom backend, follow these steps:

1. Set Backend Environment Variables: 
    - ``NETWORKX_TEST_BACKEND`` : Setting this to your registered backend key will let
      the NetworkX's dispatch machinery automatically convert a regular NetworkX
      ``Graph``, ``DiGraph``, ``MultiGraph``, etc. to their backend equivalents, using
      ``your_dispatcher_class.convert_from_nx(G, ...)`` function.
    - ``NETWORKX_FALLBACK_TO_NX`` (default=False) : Setting this variable to `True` will
      instruct tests to use a NetworkX ``Graph`` for algorithms not implemented by your
      custom backend. Setting this to `False` will only run the tests for algorithms
      implemented by your custom backend and tests for other algorithms will ``xfail``.

2. Defining ``convert_from_nx`` and ``convert_to_nx`` methods:
    The arguments to ``convert_from_nx`` are:

    - ``G`` : NetworkX Graph
    - ``edge_attrs`` : dict, optional
        Dictionary mapping edge attributes to default values if missing in ``G``.
        If None, then no edge attributes will be converted and default may be 1.
    - ``node_attrs``: dict, optional
        Dictionary mapping node attributes to default values if missing in ``G``.
        If None, then no node attributes will be converted.
    - ``preserve_edge_attrs`` : bool
        Whether to preserve all edge attributes.
    - ``preserve_node_attrs`` : bool
        Whether to preserve all node attributes.
    - ``preserve_graph_attrs`` : bool
        Whether to preserve all graph attributes.
    - ``preserve_all_attrs`` : bool
        Whether to preserve all graph, node, and edge attributes.
    - ``name`` : str
        The name of the algorithm.
    - ``graph_name`` : str
        The name of the graph argument being converted.

Running Tests
~~~~~~~~~~~~~

You can invoke NetworkX tests for your custom backend with the following commands::

    NETWORKX_TEST_BACKEND=<backend_name>
    NETWORKX_FALLBACK_TO_NX=True # or False
    pytest --pyargs networkx

Conversions while running tests :

- Convert NetworkX graphs using ``<your_dispatcher_class>.convert_from_nx(G, ...)`` into
  the backend graph.
- Pass the backend graph objects to the backend implementation of the algorithm.
- Convert the result back to a form expected by NetworkX tests using 
  ``<your_dispatcher_class>.convert_to_nx(result, ...)``.

Notes
~~~~~

-   Dispatchable algorithms that are not implemented by the backend
    will cause a ``pytest.xfail``, giving some indication that not all
    tests are running, while avoiding causing an explicit failure.

-   If a backend only partially implements some algorithms, it can define
    a ``can_run(name, args, kwargs)`` function that returns True or False
    indicating whether it can run the algorithm with the given arguments.
    It may also return a string indicating why the algorithm can't be run;
    this string may be used in the future to give helpful info to the user.

-   A backend may also define ``should_run(name, args, kwargs)`` that is similar
    to ``can_run``, but answers whether the backend *should* be run (converting
    if necessary). Like ``can_run``, it receives the original arguments so it
    can decide whether it should be run by inspecting the arguments. ``can_run``
    runs before ``should_run``, so ``should_run`` may assume ``can_run`` is True.
    If not implemented by the backend, ``can_run`` and ``should_run`` are
    assumed to always return True if the backend implements the algorithm.

-   A special ``on_start_tests(items)`` function may be defined by the backend.
    It will be called with the list of NetworkX tests discovered. Each item
    is a test object that can be marked as xfail if the backend does not support
    the test using ``item.add_marker(pytest.mark.xfail(reason=...))``.

-   A backend graph instance may have a ``G.__networkx_cache__`` dict to enable
    caching, and care should be taken to clear the cache when appropriate.
"""
import inspect
import itertools
import os
import warnings
from functools import partial
from importlib.metadata import entry_points
import networkx as nx
from .decorators import argmap
__all__ = ['_dispatchable']

class _dispatchable:
    """Allow any of the following decorator forms:
    - @_dispatchable
    - @_dispatchable()
    - @_dispatchable(name="override_name")
    - @_dispatchable(graphs="graph")
    - @_dispatchable(edge_attrs="weight")
    - @_dispatchable(graphs={"G": 0, "H": 1}, edge_attrs={"weight": "default"})

    These class attributes are currently used to allow backends to run networkx tests.
    For example: `PYTHONPATH=. pytest --backend graphblas --fallback-to-nx`
    Future work: add configuration to control these.
    """
    _is_testing = False
    _fallback_to_nx = os.environ.get('NETWORKX_FALLBACK_TO_NX', 'true').strip().lower() == 'true'

    def __new__(cls, func=None, *, name=None, graphs='G', edge_attrs=None, node_attrs=None, preserve_edge_attrs=False, preserve_node_attrs=False, preserve_graph_attrs=False, preserve_all_attrs=False, mutates_input=False, returns_graph=False):
        """A decorator that makes certain input graph types dispatch to ``func``'s
        backend implementation.

        Usage can be any of the following decorator forms:
        - @_dispatchable
        - @_dispatchable()
        - @_dispatchable(name="override_name")
        - @_dispatchable(graphs="graph_var_name")
        - @_dispatchable(edge_attrs="weight")
        - @_dispatchable(graphs={"G": 0, "H": 1}, edge_attrs={"weight": "default"})
        with 0 and 1 giving the position in the signature function for graph objects.
        When edge_attrs is a dict, keys are keyword names and values are defaults.

        The class attributes are used to allow backends to run networkx tests.
        For example: `PYTHONPATH=. pytest --backend graphblas --fallback-to-nx`
        Future work: add configuration to control these.

        Parameters
        ----------
        func : callable, optional
            The function to be decorated. If ``func`` is not provided, returns a
            partial object that can be used to decorate a function later. If ``func``
            is provided, returns a new callable object that dispatches to a backend
            algorithm based on input graph types.

        name : str, optional
            The name of the algorithm to use for dispatching. If not provided,
            the name of ``func`` will be used. ``name`` is useful to avoid name
            conflicts, as all dispatched algorithms live in a single namespace.
            For example, ``tournament.is_strongly_connected`` had a name conflict
            with the standard ``nx.is_strongly_connected``, so we used
            ``@_dispatchable(name="tournament_is_strongly_connected")``.

        graphs : str or dict or None, default "G"
            If a string, the parameter name of the graph, which must be the first
            argument of the wrapped function. If more than one graph is required
            for the algorithm (or if the graph is not the first argument), provide
            a dict of parameter name to argument position for each graph argument.
            For example, ``@_dispatchable(graphs={"G": 0, "auxiliary?": 4})``
            indicates the 0th parameter ``G`` of the function is a required graph,
            and the 4th parameter ``auxiliary`` is an optional graph.
            To indicate an argument is a list of graphs, do e.g. ``"[graphs]"``.
            Use ``graphs=None`` if *no* arguments are NetworkX graphs such as for
            graph generators, readers, and conversion functions.

        edge_attrs : str or dict, optional
            ``edge_attrs`` holds information about edge attribute arguments
            and default values for those edge attributes.
            If a string, ``edge_attrs`` holds the function argument name that
            indicates a single edge attribute to include in the converted graph.
            The default value for this attribute is 1. To indicate that an argument
            is a list of attributes (all with default value 1), use e.g. ``"[attrs]"``.
            If a dict, ``edge_attrs`` holds a dict keyed by argument names, with
            values that are either the default value or, if a string, the argument
            name that indicates the default value.

        node_attrs : str or dict, optional
            Like ``edge_attrs``, but for node attributes.

        preserve_edge_attrs : bool or str or dict, optional
            For bool, whether to preserve all edge attributes.
            For str, the name of the argument that indicates whether to preserve all edge attributes.
            For dict, a mapping of edge attribute names to their default values.

        preserve_node_attrs : bool or str or dict, optional
            Like ``preserve_edge_attrs``, but for node attributes.

        preserve_graph_attrs : bool or str or dict, optional
            Like ``preserve_edge_attrs``, but for graph attributes.

        preserve_all_attrs : bool or str or dict, optional
            Like ``preserve_edge_attrs``, but for all attributes (graph, node, and edge).

        mutates_input : bool, optional
            Whether the function modifies its input graph(s).

        returns_graph : bool, optional
            Whether the function returns a graph.
        """
        if func is None:
            return partial(cls, name=name, graphs=graphs, edge_attrs=edge_attrs,
                         node_attrs=node_attrs, preserve_edge_attrs=preserve_edge_attrs,
                         preserve_node_attrs=preserve_node_attrs,
                         preserve_graph_attrs=preserve_graph_attrs,
                         preserve_all_attrs=preserve_all_attrs,
                         mutates_input=mutates_input, returns_graph=returns_graph)

        # Register the function with the dispatcher
        if name is None:
            name = func.__name__
        _registered_algorithms[name] = {
            'func': func,
            'graphs': graphs,
            'edge_attrs': edge_attrs,
            'node_attrs': node_attrs,
            'preserve_edge_attrs': preserve_edge_attrs,
            'preserve_node_attrs': preserve_node_attrs,
            'preserve_graph_attrs': preserve_graph_attrs,
            'preserve_all_attrs': preserve_all_attrs,
            'mutates_input': mutates_input,
            'returns_graph': returns_graph,
        }

        # Return the original function
        return func

def _do_nothing():
    """This does nothing at all, yet it helps turn `_dispatchable` into functions."""
    pass

def _get_backends(group, *, load_and_call=False):
    """
    Retrieve NetworkX ``backends`` and ``backend_info`` from the entry points.

    Parameters
    -----------
    group : str
        The entry_point to be retrieved.
    load_and_call : bool, optional
        If True, load and call the backend. Defaults to False.

    Returns
    --------
    dict
        A dictionary mapping backend names to their respective backend objects.

    Notes
    ------
    If a backend is defined more than once, a warning is issued.
    The `nx-loopback` backend is removed if it exists, as it is only available during testing.
    A warning is displayed if an error occurs while loading a backend.
    """
    # Get all entry points for the specified group
    eps = entry_points()
    if group not in eps.groups:
        return {}

    # Dictionary to store backends
    result = {}

    # Process each entry point
    for ep in eps.select(group=group):
        try:
            # Load the entry point
            backend = ep.load()
            if load_and_call:
                backend = backend()

            # Check for duplicate backends
            if ep.name in result:
                warnings.warn(
                    f"Backend {ep.name} is defined more than once. "
                    f"Using {ep.value} instead of {result[ep.name]}"
                )

            # Store the backend
            result[ep.name] = backend

        except Exception as e:
            warnings.warn(f"Error loading backend {ep.name}: {str(e)}")

    # Remove loopback backend if not testing
    if not _dispatchable._is_testing and 'nx-loopback' in result:
        del result['nx-loopback']

    return result
backends = _get_backends('networkx.backends')
backend_info = _get_backends('networkx.backend_info', load_and_call=True)
from .configs import Config, config
config.backend_priority = [x.strip() for x in os.environ.get('NETWORKX_BACKEND_PRIORITY', os.environ.get('NETWORKX_AUTOMATIC_BACKENDS', '')).split(',') if x.strip()]
config.backends = Config(**{backend: (cfg if isinstance((cfg := info['default_config']), Config) else Config(**cfg)) if 'default_config' in info else Config() for backend, info in backend_info.items()})
type(config.backends).__doc__ = 'All installed NetworkX backends and their configs.'
_loaded_backends = {}
_registered_algorithms = {}

class _dispatchable:
    """Allow any of the following decorator forms:
    - @_dispatchable
    - @_dispatchable()
    - @_dispatchable(name="override_name")
    - @_dispatchable(graphs="graph")
    - @_dispatchable(edge_attrs="weight")
    - @_dispatchable(graphs={"G": 0, "H": 1}, edge_attrs={"weight": "default"})

    These class attributes are currently used to allow backends to run networkx tests.
    For example: `PYTHONPATH=. pytest --backend graphblas --fallback-to-nx`
    Future work: add configuration to control these.
    """
    _is_testing = False
    _fallback_to_nx = os.environ.get('NETWORKX_FALLBACK_TO_NX', 'true').strip().lower() == 'true'

    def __new__(cls, func=None, *, name=None, graphs='G', edge_attrs=None, node_attrs=None, preserve_edge_attrs=False, preserve_node_attrs=False, preserve_graph_attrs=False, preserve_all_attrs=False, mutates_input=False, returns_graph=False):
        """A decorator that makes certain input graph types dispatch to ``func``'s
        backend implementation.

        Usage can be any of the following decorator forms:
        - @_dispatchable
        - @_dispatchable()
        - @_dispatchable(name="override_name")
        - @_dispatchable(graphs="graph_var_name")
        - @_dispatchable(edge_attrs="weight")
        - @_dispatchable(graphs={"G": 0, "H": 1}, edge_attrs={"weight": "default"})
        with 0 and 1 giving the position in the signature function for graph objects.
        When edge_attrs is a dict, keys are keyword names and values are defaults.

        The class attributes are used to allow backends to run networkx tests.
        For example: `PYTHONPATH=. pytest --backend graphblas --fallback-to-nx`
        Future work: add configuration to control these.

        Parameters
        ----------
        func : callable, optional
            The function to be decorated. If ``func`` is not provided, returns a
            partial object that can be used to decorate a function later. If ``func``
            is provided, returns a new callable object that dispatches to a backend
            algorithm based on input graph types.

        name : str, optional
            The name of the algorithm to use for dispatching. If not provided,
            the name of ``func`` will be used. ``name`` is useful to avoid name
            conflicts, as all dispatched algorithms live in a single namespace.
            For example, ``tournament.is_strongly_connected`` had a name conflict
            with the standard ``nx.is_strongly_connected``, so we used
            ``@_dispatchable(name="tournament_is_strongly_connected")``.

        graphs : str or dict or None, default "G"
            If a string, the parameter name of the graph, which must be the first
            argument of the wrapped function. If more than one graph is required
            for the algorithm (or if the graph is not the first argument), provide
            a dict of parameter name to argument position for each graph argument.
            For example, ``@_dispatchable(graphs={"G": 0, "auxiliary?": 4})``
            indicates the 0th parameter ``G`` of the function is a required graph,
            and the 4th parameter ``auxiliary`` is an optional graph.
            To indicate an argument is a list of graphs, do e.g. ``"[graphs]"``.
            Use ``graphs=None`` if *no* arguments are NetworkX graphs such as for
            graph generators, readers, and conversion functions.

        edge_attrs : str or dict, optional
            ``edge_attrs`` holds information about edge attribute arguments
            and default values for those edge attributes.
            If a string, ``edge_attrs`` holds the function argument name that
            indicates a single edge attribute to include in the converted graph.
            The default value for this attribute is 1. To indicate that an argument
            is a list of attributes (all with default value 1), use e.g. ``"[attrs]"``.
            If a dict, ``edge_attrs`` holds a dict keyed by argument names, with
            values that are either the default value or, if a string, the argument
            name that indicates the default value.

        node_attrs : str or dict, optional
            Like ``edge_attrs``, but for node attributes.

        preserve_edge_attrs : bool or str or dict, optional
            For bool, whether to preserve all edge attributes.
            For str, the parameter name that may indicate (with ``True`` or a
            callable argument) whether all edge attributes should be preserved
            when converting.
            For dict of ``{graph_name: {attr: default}}``, indicate pre-determined
            edge attributes (and defaults) to preserve for input graphs.

        preserve_node_attrs : bool or str or dict, optional
            Like ``preserve_edge_attrs``, but for node attributes.

        preserve_graph_attrs : bool or set
            For bool, whether to preserve all graph attributes.
            For set, which input graph arguments to preserve graph attributes.

        preserve_all_attrs : bool
            Whether to preserve all edge, node and graph attributes.
            This overrides all the other preserve_*_attrs.

        mutates_input : bool or dict, default False
            For bool, whether the functions mutates an input graph argument.
            For dict of ``{arg_name: arg_pos}``, arguments that indicates whether an
            input graph will be mutated, and ``arg_name`` may begin with ``"not "``
            to negate the logic (for example, this is used by ``copy=`` arguments).
            By default, dispatching doesn't convert input graphs to a different
            backend for functions that mutate input graphs.

        returns_graph : bool, default False
            Whether the function can return or yield a graph object. By default,
            dispatching doesn't convert input graphs to a different backend for
            functions that return graphs.
        """
        if func is None:
            return partial(_dispatchable, name=name, graphs=graphs, edge_attrs=edge_attrs, node_attrs=node_attrs, preserve_edge_attrs=preserve_edge_attrs, preserve_node_attrs=preserve_node_attrs, preserve_graph_attrs=preserve_graph_attrs, preserve_all_attrs=preserve_all_attrs, mutates_input=mutates_input, returns_graph=returns_graph)
        if isinstance(func, str):
            raise TypeError("'name' and 'graphs' must be passed by keyword") from None
        if name is None:
            name = func.__name__
        self = object.__new__(cls)
        self.__name__ = func.__name__
        self.__defaults__ = func.__defaults__
        if func.__kwdefaults__:
            self.__kwdefaults__ = {**func.__kwdefaults__, 'backend': None}
        else:
            self.__kwdefaults__ = {'backend': None}
        self.__module__ = func.__module__
        self.__qualname__ = func.__qualname__
        self.__dict__.update(func.__dict__)
        self.__wrapped__ = func
        self._orig_doc = func.__doc__
        self._cached_doc = None
        self.orig_func = func
        self.name = name
        self.edge_attrs = edge_attrs
        self.node_attrs = node_attrs
        self.preserve_edge_attrs = preserve_edge_attrs or preserve_all_attrs
        self.preserve_node_attrs = preserve_node_attrs or preserve_all_attrs
        self.preserve_graph_attrs = preserve_graph_attrs or preserve_all_attrs
        self.mutates_input = mutates_input
        self._returns_graph = returns_graph
        if edge_attrs is not None and (not isinstance(edge_attrs, str | dict)):
            raise TypeError(f'Bad type for edge_attrs: {type(edge_attrs)}. Expected str or dict.') from None
        if node_attrs is not None and (not isinstance(node_attrs, str | dict)):
            raise TypeError(f'Bad type for node_attrs: {type(node_attrs)}. Expected str or dict.') from None
        if not isinstance(self.preserve_edge_attrs, bool | str | dict):
            raise TypeError(f'Bad type for preserve_edge_attrs: {type(self.preserve_edge_attrs)}. Expected bool, str, or dict.') from None
        if not isinstance(self.preserve_node_attrs, bool | str | dict):
            raise TypeError(f'Bad type for preserve_node_attrs: {type(self.preserve_node_attrs)}. Expected bool, str, or dict.') from None
        if not isinstance(self.preserve_graph_attrs, bool | set):
            raise TypeError(f'Bad type for preserve_graph_attrs: {type(self.preserve_graph_attrs)}. Expected bool or set.') from None
        if not isinstance(self.mutates_input, bool | dict):
            raise TypeError(f'Bad type for mutates_input: {type(self.mutates_input)}. Expected bool or dict.') from None
        if not isinstance(self._returns_graph, bool):
            raise TypeError(f'Bad type for returns_graph: {type(self._returns_graph)}. Expected bool.') from None
        if isinstance(graphs, str):
            graphs = {graphs: 0}
        elif graphs is None:
            pass
        elif not isinstance(graphs, dict):
            raise TypeError(f'Bad type for graphs: {type(graphs)}. Expected str or dict.') from None
        elif len(graphs) == 0:
            raise KeyError("'graphs' must contain at least one variable name") from None
        self.optional_graphs = set()
        self.list_graphs = set()
        if graphs is None:
            self.graphs = {}
        else:
            self.graphs = {self.optional_graphs.add((val := k[:-1])) or val if (last := k[-1]) == '?' else self.list_graphs.add((val := k[1:-1])) or val if last == ']' else k: v for k, v in graphs.items()}
        self._sig = None
        self.backends = {backend for backend, info in backend_info.items() if 'functions' in info and name in info['functions']}
        if name in _registered_algorithms:
            raise KeyError(f'Algorithm already exists in dispatch registry: {name}') from None
        self = argmap(_do_nothing)(self)
        _registered_algorithms[name] = self
        return self

    @property
    def __doc__(self):
        """If the cached documentation exists, it is returned.
        Otherwise, the documentation is generated using _make_doc() method,
        cached, and then returned."""
        if (rv := self._cached_doc) is not None:
            return rv
        rv = self._cached_doc = self._make_doc()
        return rv

    @__doc__.setter
    def __doc__(self, val):
        """Sets the original documentation to the given value and resets the
        cached documentation."""
        self._orig_doc = val
        self._cached_doc = None

    @property
    def __signature__(self):
        """Return the signature of the original function, with the addition of
        the `backend` and `backend_kwargs` parameters."""
        if self._sig is None:
            sig = inspect.signature(self.orig_func)
            if not any((p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())):
                sig = sig.replace(parameters=[*sig.parameters.values(), inspect.Parameter('backend', inspect.Parameter.KEYWORD_ONLY, default=None), inspect.Parameter('backend_kwargs', inspect.Parameter.VAR_KEYWORD)])
            else:
                *parameters, var_keyword = sig.parameters.values()
                sig = sig.replace(parameters=[*parameters, inspect.Parameter('backend', inspect.Parameter.KEYWORD_ONLY, default=None), var_keyword])
            self._sig = sig
        return self._sig

    def __call__(self, /, *args, backend=None, **kwargs):
        """Returns the result of the original function, or the backend function if
        the backend is specified and that backend implements `func`."""
        if not backends:
            return self.orig_func(*args, **kwargs)
        backend_name = backend
        if backend_name is not None and backend_name not in backends:
            raise ImportError(f'Unable to load backend: {backend_name}')
        graphs_resolved = {}
        for gname, pos in self.graphs.items():
            if pos < len(args):
                if gname in kwargs:
                    raise TypeError(f'{self.name}() got multiple values for {gname!r}')
                val = args[pos]
            elif gname in kwargs:
                val = kwargs[gname]
            elif gname not in self.optional_graphs:
                raise TypeError(f'{self.name}() missing required graph argument: {gname}')
            else:
                continue
            if val is None:
                if gname not in self.optional_graphs:
                    raise TypeError(f'{self.name}() required graph argument {gname!r} is None; must be a graph')
            else:
                graphs_resolved[gname] = val
        if self.list_graphs:
            args = list(args)
            for gname in self.list_graphs & graphs_resolved.keys():
                val = list(graphs_resolved[gname])
                graphs_resolved[gname] = val
                if gname in kwargs:
                    kwargs[gname] = val
                else:
                    args[self.graphs[gname]] = val
            has_backends = any((hasattr(g, '__networkx_backend__') if gname not in self.list_graphs else any((hasattr(g2, '__networkx_backend__') for g2 in g)) for gname, g in graphs_resolved.items()))
            if has_backends:
                graph_backend_names = {getattr(g, '__networkx_backend__', 'networkx') for gname, g in graphs_resolved.items() if gname not in self.list_graphs}
                for gname in self.list_graphs & graphs_resolved.keys():
                    graph_backend_names.update((getattr(g, '__networkx_backend__', 'networkx') for g in graphs_resolved[gname]))
        else:
            has_backends = any((hasattr(g, '__networkx_backend__') for g in graphs_resolved.values()))
            if has_backends:
                graph_backend_names = {getattr(g, '__networkx_backend__', 'networkx') for g in graphs_resolved.values()}
        backend_priority = config.backend_priority
        if self._is_testing and backend_priority and (backend_name is None):
            return self._convert_and_call_for_tests(backend_priority[0], args, kwargs, fallback_to_nx=self._fallback_to_nx)
        if has_backends:
            backend_names = graph_backend_names - {'networkx'}
            if len(backend_names) != 1:
                raise TypeError(f'{self.name}() graphs must all be from the same backend, found {backend_names}')
            [graph_backend_name] = backend_names
            if backend_name is not None and backend_name != graph_backend_name:
                raise TypeError(f'{self.name}() is unable to convert graph from backend {graph_backend_name!r} to the specified backend {backend_name!r}.')
            if graph_backend_name not in backends:
                raise ImportError(f'Unable to load backend: {graph_backend_name}')
            if 'networkx' in graph_backend_names and graph_backend_name not in backend_priority:
                raise TypeError(f'Unable to convert inputs and run {self.name}. {self.name}() has networkx and {graph_backend_name} graphs, but NetworkX is not configured to automatically convert graphs from networkx to {graph_backend_name}.')
            backend = _load_backend(graph_backend_name)
            if hasattr(backend, self.name):
                if 'networkx' in graph_backend_names:
                    return self._convert_and_call(graph_backend_name, args, kwargs, fallback_to_nx=self._fallback_to_nx)
                return getattr(backend, self.name)(*args, **kwargs)
            raise nx.NetworkXNotImplemented(f"'{self.name}' not implemented by {graph_backend_name}")
        if backend_name is not None:
            return self._convert_and_call(backend_name, args, kwargs, fallback_to_nx=False)
        if not self._returns_graph and (not self.mutates_input or (isinstance(self.mutates_input, dict) and any((not (args[arg_pos] if len(args) > arg_pos else kwargs.get(arg_name[4:], True)) if arg_name.startswith('not ') else (args[arg_pos] if len(args) > arg_pos else kwargs.get(arg_name)) is not None for arg_name, arg_pos in self.mutates_input.items())))):
            for backend_name in backend_priority:
                if self._should_backend_run(backend_name, *args, **kwargs):
                    return self._convert_and_call(backend_name, args, kwargs, fallback_to_nx=self._fallback_to_nx)
        return self.orig_func(*args, **kwargs)

    def _can_backend_run(self, backend_name, /, *args, **kwargs):
        """Can the specified backend run this algorithm with these arguments?"""
        pass

    def _should_backend_run(self, backend_name, /, *args, **kwargs):
        """Can/should the specified backend run this algorithm with these arguments?"""
        pass

    def _convert_arguments(self, backend_name, args, kwargs, *, use_cache):
        """Convert graph arguments to the specified backend.

        Returns
        -------
        args tuple and kwargs dict
        """
        pass

    def _convert_and_call(self, backend_name, args, kwargs, *, fallback_to_nx=False):
        """Call this dispatchable function with a backend, converting graphs if necessary."""
        pass

    def _convert_and_call_for_tests(self, backend_name, args, kwargs, *, fallback_to_nx=False):
        """Call this dispatchable function with a backend; for use with testing."""
        pass

    def _make_doc(self):
        """Generate the backends section at the end for functions having an alternate
        backend implementation(s) using the `backend_info` entry-point."""
        pass

    def __reduce__(self):
        """Allow this object to be serialized with pickle.

        This uses the global registry `_registered_algorithms` to deserialize.
        """
        return (_restore_dispatchable, (self.name,))
if os.environ.get('_NETWORKX_BUILDING_DOCS_'):
    _orig_dispatchable = _dispatchable
    _dispatchable.__doc__ = _orig_dispatchable.__new__.__doc__
    _sig = inspect.signature(_orig_dispatchable.__new__)
    _dispatchable.__signature__ = _sig.replace(parameters=[v for k, v in _sig.parameters.items() if k != 'cls'])