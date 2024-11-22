import bz2
import collections
import gzip
import inspect
import itertools
import re
import warnings
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from inspect import Parameter, signature
from os.path import splitext
from pathlib import Path
import networkx as nx
from networkx.utils import create_py_random_state, create_random_state
__all__ = ['not_implemented_for', 'open_file', 'nodes_or_number', 'np_random_state', 'py_random_state', 'argmap', 'deprecate_positional_args']

def not_implemented_for(*graph_types):
    """Decorator to mark algorithms as not implemented

    Parameters
    ----------
    graph_types : container of strings
        Entries must be one of "directed", "undirected", "multigraph", or "graph".

    Returns
    -------
    _require : function
        The decorated function.

    Raises
    ------
    NetworkXNotImplemented
    If any of the packages cannot be imported

    Notes
    -----
    Multiple types are joined logically with "and".
    For "or" use multiple @not_implemented_for() lines.

    Examples
    --------
    Decorate functions like this::

       @not_implemented_for("directed")
       def sp_function(G):
           pass


       # rule out MultiDiGraph
       @not_implemented_for("directed", "multigraph")
       def sp_np_function(G):
           pass


       # rule out all except DiGraph
       @not_implemented_for("undirected")
       @not_implemented_for("multigraph")
       def sp_np_function(G):
           pass
    """
    def _not_implemented_for(func):
        @wraps(func)
        def _not_implemented_for_func(*args, **kwargs):
            graph = args[0]
            terms = {
                "directed": graph.is_directed(),
                "undirected": not graph.is_directed(),
                "multigraph": graph.is_multigraph(),
                "graph": not graph.is_multigraph(),
            }
            match = True
            try:
                for t in graph_types:
                    if terms[t]:
                        match = False
            except KeyError as e:
                raise KeyError(f"use one of {', '.join(terms)}") from e
            if match:
                return func(*args, **kwargs)
            msg = f"{func.__name__} not implemented for {' '.join(graph_types)} type"
            raise nx.NetworkXNotImplemented(msg)
        return _not_implemented_for_func
    return _not_implemented_for
fopeners = {'.gz': gzip.open, '.gzip': gzip.open, '.bz2': bz2.BZ2File}
_dispatch_dict = defaultdict(lambda: open, **fopeners)

def open_file(path_arg, mode='r'):
    """Decorator to ensure clean opening and closing of files.

    Parameters
    ----------
    path_arg : string or int
        Name or index of the argument that is a path.

    mode : str
        String for opening mode.

    Returns
    -------
    _open_file : function
        Function which cleanly executes the io.

    Examples
    --------
    Decorate functions like this::

       @open_file(0, "r")
       def read_function(pathname):
           pass


       @open_file(1, "w")
       def write_function(G, pathname):
           pass


       @open_file(1, "w")
       def write_function(G, pathname="graph.dot"):
           pass


       @open_file("pathname", "w")
       def write_function(G, pathname="graph.dot"):
           pass


       @open_file("path", "w+")
       def another_function(arg, **kwargs):
           path = kwargs["path"]
           pass

    Notes
    -----
    Note that this decorator solves the problem when a path argument is
    specified as a string, but it does not handle the situation when the
    function wants to accept a default of None (and then handle it).

    Here is an example of how to handle this case::

      @open_file("path")
      def some_function(arg1, arg2, path=None):
          if path is None:
              fobj = tempfile.NamedTemporaryFile(delete=False)
          else:
              # `path` could have been a string or file object or something
              # similar. In any event, the decorator has given us a file object
              # and it will close it for us, if it should.
              fobj = path

          try:
              fobj.write("blah")
          finally:
              if path is None:
                  fobj.close()

    Normally, we'd want to use "with" to ensure that fobj gets closed.
    However, the decorator will make `path` a file object for us,
    and using "with" would undesirably close that file object.
    Instead, we use a try block, as shown above.
    When we exit the function, fobj will be closed, if it should be, by the decorator.
    """
    def _open_file(func):
        def _file_opener(*args, **kwargs):
            # Get the path argument
            if isinstance(path_arg, int):
                # path_arg is a position argument
                if path_arg >= len(args):
                    # If the path_arg index is greater than the number of arguments,
                    # try to find the argument in the kwargs using its name
                    sig = signature(func)
                    param_names = list(sig.parameters.keys())
                    if path_arg < len(param_names):
                        path = kwargs.get(param_names[path_arg], None)
                    else:
                        raise ValueError(f"path argument at index {path_arg} not found")
                else:
                    path = args[path_arg]
            else:
                # path_arg is a keyword argument
                path = kwargs.get(path_arg, None)

            # Return quickly if no path argument was found or it's None
            if path is None:
                return func(*args, **kwargs)

            # If the path argument is already a file object, just pass it through
            if hasattr(path, 'write') and hasattr(path, 'read'):
                return func(*args, **kwargs)

            # Convert path to Path object to handle both string and Path inputs
            if isinstance(path, str):
                path = Path(path)

            # Get the file extension and corresponding opener
            ext = path.suffix.lower()
            fopen = _dispatch_dict[ext]

            # Open the file with the appropriate opener
            try:
                fobj = fopen(path, mode=mode)
            except Exception as e:
                raise ValueError(f"Failed to open file {path}: {str(e)}")

            # Replace the path argument with the file object
            if isinstance(path_arg, int):
                # Convert args to list for modification
                args = list(args)
                if path_arg < len(args):
                    args[path_arg] = fobj
                else:
                    kwargs[list(signature(func).parameters.keys())[path_arg]] = fobj
                args = tuple(args)
            else:
                kwargs[path_arg] = fobj

            # Call the function and ensure the file is closed afterward
            try:
                result = func(*args, **kwargs)
            finally:
                fobj.close()

            return result

        return _file_opener

    return _open_file

def nodes_or_number(which_args):
    """Decorator to allow number of nodes or container of nodes.

    With this decorator, the specified argument can be either a number or a container
    of nodes. If it is a number, the nodes used are `range(n)`.
    This allows `nx.complete_graph(50)` in place of `nx.complete_graph(list(range(50)))`.
    And it also allows `nx.complete_graph(any_list_of_nodes)`.

    Parameters
    ----------
    which_args : string or int or sequence of strings or ints
        If string, the name of the argument to be treated.
        If int, the index of the argument to be treated.
        If more than one node argument is allowed, can be a list of locations.

    Returns
    -------
    _nodes_or_numbers : function
        Function which replaces int args with ranges.

    Examples
    --------
    Decorate functions like this::

       @nodes_or_number("nodes")
       def empty_graph(nodes):
           # nodes is converted to a list of nodes

       @nodes_or_number(0)
       def empty_graph(nodes):
           # nodes is converted to a list of nodes

       @nodes_or_number(["m1", "m2"])
       def grid_2d_graph(m1, m2, periodic=False):
           # m1 and m2 are each converted to a list of nodes

       @nodes_or_number([0, 1])
       def grid_2d_graph(m1, m2, periodic=False):
           # m1 and m2 are each converted to a list of nodes

       @nodes_or_number(1)
       def full_rary_tree(r, n)
           # presumably r is a number. It is not handled by this decorator.
           # n is converted to a list of nodes
    """
    def _nodes_or_number(func):
        # If which_args is not a list, make it a list
        arglist = which_args if isinstance(which_args, (list, tuple)) else [which_args]

        @wraps(func)
        def _nodes_or_numbers_func(*args, **kwargs):
            # Get the names of the arguments
            sig = signature(func)
            param_names = list(sig.parameters.keys())

            # Convert args to a list for modification
            args = list(args)

            # Process each argument that needs conversion
            for arg_loc in arglist:
                if isinstance(arg_loc, int):
                    # If arg_loc is an integer, it's an index into args
                    if arg_loc < len(args):
                        if isinstance(args[arg_loc], int):
                            args[arg_loc] = range(args[arg_loc])
                    else:
                        # If the argument is not in args, it might be in kwargs
                        # Find the name of the argument at this position
                        if arg_loc < len(param_names):
                            arg_name = param_names[arg_loc]
                            if arg_name in kwargs and isinstance(kwargs[arg_name], int):
                                kwargs[arg_name] = range(kwargs[arg_name])
                elif isinstance(arg_loc, str):
                    # If arg_loc is a string, it's the name of a kwarg
                    if arg_loc in kwargs:
                        if isinstance(kwargs[arg_loc], int):
                            kwargs[arg_loc] = range(kwargs[arg_loc])
                    else:
                        # If not in kwargs, check if it's in args
                        try:
                            idx = param_names.index(arg_loc)
                            if idx < len(args) and isinstance(args[idx], int):
                                args[idx] = range(args[idx])
                        except ValueError:
                            pass

            # Convert args back to tuple and call the function
            return func(*args, **kwargs)

        return _nodes_or_numbers_func

    return _nodes_or_number

def np_random_state(random_state_argument):
    """Decorator to generate a numpy RandomState or Generator instance.

    The decorator processes the argument indicated by `random_state_argument`
    using :func:`nx.utils.create_random_state`.
    The argument value can be a seed (integer), or a `numpy.random.RandomState`
    or `numpy.random.RandomState` instance or (`None` or `numpy.random`).
    The latter two options use the global random number generator for `numpy.random`.

    The returned instance is a `numpy.random.RandomState` or `numpy.random.Generator`.

    Parameters
    ----------
    random_state_argument : string or int
        The name or index of the argument to be converted
        to a `numpy.random.RandomState` instance.

    Returns
    -------
    _random_state : function
        Function whose random_state keyword argument is a RandomState instance.

    Examples
    --------
    Decorate functions like this::

       @np_random_state("seed")
       def random_float(seed=None):
           return seed.rand()


       @np_random_state(0)
       def random_float(rng=None):
           return rng.rand()


       @np_random_state(1)
       def random_array(dims, random_state=1):
           return random_state.rand(*dims)

    See Also
    --------
    py_random_state
    """
    def _random_state(func):
        # Local import to avoid circular import
        from networkx.utils import create_random_state

        # Get the name of the random_state argument
        if isinstance(random_state_argument, str):
            random_state_name = random_state_argument
        elif isinstance(random_state_argument, int):
            # Get the name of the argument at the given index
            sig = signature(func)
            param_names = list(sig.parameters.keys())
            if random_state_argument < len(param_names):
                random_state_name = param_names[random_state_argument]
            else:
                # Handle variadic arguments (*args)
                for param in sig.parameters.values():
                    if param.kind == Parameter.VAR_POSITIONAL:
                        random_state_name = random_state_argument
                        break
                else:
                    msg = f"Argument index {random_state_argument} is out of range"
                    raise ValueError(msg)
        else:
            raise ValueError("random_state_argument must be string or integer")

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the random_state argument value
            if isinstance(random_state_name, int):
                # Handle variadic arguments
                if random_state_name < len(args):
                    random_state = args[random_state_name]
                    args = list(args)  # Convert to list to allow modification
                    args[random_state_name] = create_random_state(random_state)
                    args = tuple(args)  # Convert back to tuple
                else:
                    raise ValueError(f"No argument at index {random_state_name}")
            else:
                # Handle named arguments
                try:
                    random_state = kwargs[random_state_name]
                    kwargs[random_state_name] = create_random_state(random_state)
                except KeyError:
                    # If not in kwargs, check if it's in args
                    sig = signature(func)
                    param_names = list(sig.parameters.keys())
                    try:
                        idx = param_names.index(random_state_name)
                        if idx < len(args):
                            random_state = args[idx]
                            args = list(args)
                            args[idx] = create_random_state(random_state)
                            args = tuple(args)
                        else:
                            # Use default value if available
                            param = sig.parameters[random_state_name]
                            if param.default is not Parameter.empty:
                                kwargs[random_state_name] = create_random_state(param.default)
                            else:
                                raise ValueError(f"No value provided for {random_state_name}")
                    except ValueError:
                        raise ValueError(f"No argument named {random_state_name}")

            return func(*args, **kwargs)

        return wrapper

    return _random_state

def py_random_state(random_state_argument):
    """Decorator to generate a random.Random instance (or equiv).

    This decorator processes `random_state_argument` using
    :func:`nx.utils.create_py_random_state`.
    The input value can be a seed (integer), or a random number generator::

        If int, return a random.Random instance set with seed=int.
        If random.Random instance, return it.
        If None or the `random` package, return the global random number
        generator used by `random`.
        If np.random package, or the default numpy RandomState instance,
        return the default numpy random number generator wrapped in a
        `PythonRandomViaNumpyBits`  class.
        If np.random.Generator instance, return it wrapped in a
        `PythonRandomViaNumpyBits`  class.

        # Legacy options
        If np.random.RandomState instance, return it wrapped in a
        `PythonRandomInterface` class.
        If a `PythonRandomInterface` instance, return it

    Parameters
    ----------
    random_state_argument : string or int
        The name of the argument or the index of the argument in args that is
        to be converted to the random.Random instance or numpy.random.RandomState
        instance that mimics basic methods of random.Random.

    Returns
    -------
    _random_state : function
        Function whose random_state_argument is converted to a Random instance.

    Examples
    --------
    Decorate functions like this::

       @py_random_state("random_state")
       def random_float(random_state=None):
           return random_state.rand()


       @py_random_state(0)
       def random_float(rng=None):
           return rng.rand()


       @py_random_state(1)
       def random_array(dims, seed=12345):
           return seed.rand(*dims)

    See Also
    --------
    np_random_state
    """
    def _random_state(func):
        # Local import to avoid circular import
        from networkx.utils import create_py_random_state

        # Get the name of the random_state argument
        if isinstance(random_state_argument, str):
            random_state_name = random_state_argument
        elif isinstance(random_state_argument, int):
            # Get the name of the argument at the given index
            sig = signature(func)
            param_names = list(sig.parameters.keys())
            if random_state_argument < len(param_names):
                random_state_name = param_names[random_state_argument]
            else:
                # Handle variadic arguments (*args)
                for param in sig.parameters.values():
                    if param.kind == Parameter.VAR_POSITIONAL:
                        random_state_name = random_state_argument
                        break
                else:
                    msg = f"Argument index {random_state_argument} is out of range"
                    raise ValueError(msg)
        else:
            raise ValueError("random_state_argument must be string or integer")

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the random_state argument value
            if isinstance(random_state_name, int):
                # Handle variadic arguments
                if random_state_name < len(args):
                    random_state = args[random_state_name]
                    args = list(args)  # Convert to list to allow modification
                    args[random_state_name] = create_py_random_state(random_state)
                    args = tuple(args)  # Convert back to tuple
                else:
                    raise ValueError(f"No argument at index {random_state_name}")
            else:
                # Handle named arguments
                try:
                    random_state = kwargs[random_state_name]
                    kwargs[random_state_name] = create_py_random_state(random_state)
                except KeyError:
                    # If not in kwargs, check if it's in args
                    sig = signature(func)
                    param_names = list(sig.parameters.keys())
                    try:
                        idx = param_names.index(random_state_name)
                        if idx < len(args):
                            random_state = args[idx]
                            args = list(args)
                            args[idx] = create_py_random_state(random_state)
                            args = tuple(args)
                        else:
                            # Use default value if available
                            param = sig.parameters[random_state_name]
                            if param.default is not Parameter.empty:
                                kwargs[random_state_name] = create_py_random_state(param.default)
                            else:
                                raise ValueError(f"No value provided for {random_state_name}")
                    except ValueError:
                        raise ValueError(f"No argument named {random_state_name}")

            return func(*args, **kwargs)

        return wrapper

    return _random_state

class argmap:
    """A decorator to apply a map to arguments before calling the function

    This class provides a decorator that maps (transforms) arguments of the function
    before the function is called. Thus for example, we have similar code
    in many functions to determine whether an argument is the number of nodes
    to be created, or a list of nodes to be handled. The decorator provides
    the code to accept either -- transforming the indicated argument into a
    list of nodes before the actual function is called.

    This decorator class allows us to process single or multiple arguments.
    The arguments to be processed can be specified by string, naming the argument,
    or by index, specifying the item in the args list.

    Parameters
    ----------
    func : callable
        The function to apply to arguments

    *args : iterable of (int, str or tuple)
        A list of parameters, specified either as strings (their names), ints
        (numerical indices) or tuples, which may contain ints, strings, and
        (recursively) tuples. Each indicates which parameters the decorator
        should map. Tuples indicate that the map function takes (and returns)
        multiple parameters in the same order and nested structure as indicated
        here.

    try_finally : bool (default: False)
        When True, wrap the function call in a try-finally block with code
        for the finally block created by `func`. This is used when the map
        function constructs an object (like a file handle) that requires
        post-processing (like closing).

        Note: try_finally decorators cannot be used to decorate generator
        functions.

    Examples
    --------
    Most of these examples use `@argmap(...)` to apply the decorator to
    the function defined on the next line.
    In the NetworkX codebase however, `argmap` is used within a function to
    construct a decorator. That is, the decorator defines a mapping function
    and then uses `argmap` to build and return a decorated function.
    A simple example is a decorator that specifies which currency to report money.
    The decorator (named `convert_to`) would be used like::

        @convert_to("US_Dollars", "income")
        def show_me_the_money(name, income):
            print(f"{name} : {income}")

    And the code to create the decorator might be::

        def convert_to(currency, which_arg):
            def _convert(amount):
                if amount.currency != currency:
                    amount = amount.to_currency(currency)
                return amount

            return argmap(_convert, which_arg)

    Despite this common idiom for argmap, most of the following examples
    use the `@argmap(...)` idiom to save space.

    Here's an example use of argmap to sum the elements of two of the functions
    arguments. The decorated function::

        @argmap(sum, "xlist", "zlist")
        def foo(xlist, y, zlist):
            return xlist - y + zlist

    is syntactic sugar for::

        def foo(xlist, y, zlist):
            x = sum(xlist)
            z = sum(zlist)
            return x - y + z

    and is equivalent to (using argument indexes)::

        @argmap(sum, "xlist", 2)
        def foo(xlist, y, zlist):
            return xlist - y + zlist

    or::

        @argmap(sum, "zlist", 0)
        def foo(xlist, y, zlist):
            return xlist - y + zlist

    Transforming functions can be applied to multiple arguments, such as::

        def swap(x, y):
            return y, x

        # the 2-tuple tells argmap that the map `swap` has 2 inputs/outputs.
        @argmap(swap, ("a", "b")):
        def foo(a, b, c):
            return a / b * c

    is equivalent to::

        def foo(a, b, c):
            a, b = swap(a, b)
            return a / b * c

    More generally, the applied arguments can be nested tuples of strings or ints.
    The syntax `@argmap(some_func, ("a", ("b", "c")))` would expect `some_func` to
    accept 2 inputs with the second expected to be a 2-tuple. It should then return
    2 outputs with the second a 2-tuple. The returns values would replace input "a"
    "b" and "c" respectively. Similarly for `@argmap(some_func, (0, ("b", 2)))`.

    Also, note that an index larger than the number of named parameters is allowed
    for variadic functions. For example::

        def double(a):
            return 2 * a


        @argmap(double, 3)
        def overflow(a, *args):
            return a, args


        print(overflow(1, 2, 3, 4, 5, 6))  # output is 1, (2, 3, 8, 5, 6)

    **Try Finally**

    Additionally, this `argmap` class can be used to create a decorator that
    initiates a try...finally block. The decorator must be written to return
    both the transformed argument and a closing function.
    This feature was included to enable the `open_file` decorator which might
    need to close the file or not depending on whether it had to open that file.
    This feature uses the keyword-only `try_finally` argument to `@argmap`.

    For example this map opens a file and then makes sure it is closed::

        def open_file(fn):
            f = open(fn)
            return f, lambda: f.close()

    The decorator applies that to the function `foo`::

        @argmap(open_file, "file", try_finally=True)
        def foo(file):
            print(file.read())

    is syntactic sugar for::

        def foo(file):
            file, close_file = open_file(file)
            try:
                print(file.read())
            finally:
                close_file()

    and is equivalent to (using indexes)::

        @argmap(open_file, 0, try_finally=True)
        def foo(file):
            print(file.read())

    Here's an example of the try_finally feature used to create a decorator::

        def my_closing_decorator(which_arg):
            def _opener(path):
                if path is None:
                    path = open(path)
                    fclose = path.close
                else:
                    # assume `path` handles the closing
                    fclose = lambda: None
                return path, fclose

            return argmap(_opener, which_arg, try_finally=True)

    which can then be used as::

        @my_closing_decorator("file")
        def fancy_reader(file=None):
            # this code doesn't need to worry about closing the file
            print(file.read())

    Decorators with try_finally = True cannot be used with generator functions,
    because the `finally` block is evaluated before the generator is exhausted::

        @argmap(open_file, "file", try_finally=True)
        def file_to_lines(file):
            for line in file.readlines():
                yield line

    is equivalent to::

        def file_to_lines_wrapped(file):
            for line in file.readlines():
                yield line


        def file_to_lines_wrapper(file):
            try:
                file = open_file(file)
                return file_to_lines_wrapped(file)
            finally:
                file.close()

    which behaves similarly to::

        def file_to_lines_whoops(file):
            file = open_file(file)
            file.close()
            for line in file.readlines():
                yield line

    because the `finally` block of `file_to_lines_wrapper` is executed before
    the caller has a chance to exhaust the iterator.

    Notes
    -----
    An object of this class is callable and intended to be used when
    defining a decorator. Generally, a decorator takes a function as input
    and constructs a function as output. Specifically, an `argmap` object
    returns the input function decorated/wrapped so that specified arguments
    are mapped (transformed) to new values before the decorated function is called.

    As an overview, the argmap object returns a new function with all the
    dunder values of the original function (like `__doc__`, `__name__`, etc).
    Code for this decorated function is built based on the original function's
    signature. It starts by mapping the input arguments to potentially new
    values. Then it calls the decorated function with these new values in place
    of the indicated arguments that have been mapped. The return value of the
    original function is then returned. This new function is the function that
    is actually called by the user.

    Three additional features are provided.
        1) The code is lazily compiled. That is, the new function is returned
        as an object without the code compiled, but with all information
        needed so it can be compiled upon it's first invocation. This saves
        time on import at the cost of additional time on the first call of
        the function. Subsequent calls are then just as fast as normal.

        2) If the "try_finally" keyword-only argument is True, a try block
        follows each mapped argument, matched on the other side of the wrapped
        call, by a finally block closing that mapping.  We expect func to return
        a 2-tuple: the mapped value and a function to be called in the finally
        clause.  This feature was included so the `open_file` decorator could
        provide a file handle to the decorated function and close the file handle
        after the function call. It even keeps track of whether to close the file
        handle or not based on whether it had to open the file or the input was
        already open. So, the decorated function does not need to include any
        code to open or close files.

        3) The maps applied can process multiple arguments. For example,
        you could swap two arguments using a mapping, or transform
        them to their sum and their difference. This was included to allow
        a decorator in the `quality.py` module that checks that an input
        `partition` is a valid partition of the nodes of the input graph `G`.
        In this example, the map has inputs `(G, partition)`. After checking
        for a valid partition, the map either raises an exception or leaves
        the inputs unchanged. Thus many functions that make this check can
        use the decorator rather than copy the checking code into each function.
        More complicated nested argument structures are described below.

    The remaining notes describe the code structure and methods for this
    class in broad terms to aid in understanding how to use it.

    Instantiating an `argmap` object simply stores the mapping function and
    the input identifiers of which arguments to map. The resulting decorator
    is ready to use this map to decorate any function. Calling that object
    (`argmap.__call__`, but usually done via `@my_decorator`) a lazily
    compiled thin wrapper of the decorated function is constructed,
    wrapped with the necessary function dunder attributes like `__doc__`
    and `__name__`. That thinly wrapped function is returned as the
    decorated function. When that decorated function is called, the thin
    wrapper of code calls `argmap._lazy_compile` which compiles the decorated
    function (using `argmap.compile`) and replaces the code of the thin
    wrapper with the newly compiled code. This saves the compilation step
    every import of networkx, at the cost of compiling upon the first call
    to the decorated function.

    When the decorated function is compiled, the code is recursively assembled
    using the `argmap.assemble` method. The recursive nature is needed in
    case of nested decorators. The result of the assembly is a number of
    useful objects.

      sig : the function signature of the original decorated function as
          constructed by :func:`argmap.signature`. This is constructed
          using `inspect.signature` but enhanced with attribute
          strings `sig_def` and `sig_call`, and other information
          specific to mapping arguments of this function.
          This information is used to construct a string of code defining
          the new decorated function.

      wrapped_name : a unique internally used name constructed by argmap
          for the decorated function.

      functions : a dict of the functions used inside the code of this
          decorated function, to be used as `globals` in `exec`.
          This dict is recursively updated to allow for nested decorating.

      mapblock : code (as a list of strings) to map the incoming argument
          values to their mapped values.

      finallys : code (as a list of strings) to provide the possibly nested
          set of finally clauses if needed.

      mutable_args : a bool indicating whether the `sig.args` tuple should be
          converted to a list so mutation can occur.

    After this recursive assembly process, the `argmap.compile` method
    constructs code (as strings) to convert the tuple `sig.args` to a list
    if needed. It joins the defining code with appropriate indents and
    compiles the result.  Finally, this code is evaluated and the original
    wrapper's implementation is replaced with the compiled version (see
    `argmap._lazy_compile` for more details).

    Other `argmap` methods include `_name` and `_count` which allow internally
    generated names to be unique within a python session.
    The methods `_flatten` and `_indent` process the nested lists of strings
    into properly indented python code ready to be compiled.

    More complicated nested tuples of arguments also allowed though
    usually not used. For the simple 2 argument case, the argmap
    input ("a", "b") implies the mapping function will take 2 arguments
    and return a 2-tuple of mapped values. A more complicated example
    with argmap input `("a", ("b", "c"))` requires the mapping function
    take 2 inputs, with the second being a 2-tuple. It then must output
    the 3 mapped values in the same nested structure `(newa, (newb, newc))`.
    This level of generality is not often needed, but was convenient
    to implement when handling the multiple arguments.

    See Also
    --------
    not_implemented_for
    open_file
    nodes_or_number
    py_random_state
    networkx.algorithms.community.quality.require_partition

    """

    def __init__(self, func, *args, try_finally=False):
        self._func = func
        self._args = args
        self._finally = try_finally

    @staticmethod
    def _lazy_compile(func):
        """Compile the source of a wrapped function

        Assemble and compile the decorated function, and intrusively replace its
        code with the compiled version's.  The thinly wrapped function becomes
        the decorated function.

        Parameters
        ----------
        func : callable
            A function returned by argmap.__call__ which is in the process
            of being called for the first time.

        Returns
        -------
        func : callable
            The same function, with a new __code__ object.

        Notes
        -----
        It was observed in NetworkX issue #4732 [1] that the import time of
        NetworkX was significantly bloated by the use of decorators: over half
        of the import time was being spent decorating functions.  This was
        somewhat improved by a change made to the `decorator` library, at the
        cost of a relatively heavy-weight call to `inspect.Signature.bind`
        for each call to the decorated function.

        The workaround we arrived at is to do minimal work at the time of
        decoration.  When the decorated function is called for the first time,
        we compile a function with the same function signature as the wrapped
        function.  The resulting decorated function is faster than one made by
        the `decorator` library, so that the overhead of the first call is
        'paid off' after a small number of calls.

        References
        ----------

        [1] https://github.com/networkx/networkx/issues/4732

        """
        pass

    def __call__(self, f):
        """Construct a lazily decorated wrapper of f.

        The decorated function will be compiled when it is called for the first time,
        and it will replace its own __code__ object so subsequent calls are fast.

        Parameters
        ----------
        f : callable
            A function to be decorated.

        Returns
        -------
        func : callable
            The decorated function.

        See Also
        --------
        argmap._lazy_compile
        """

        def func(*args, __wrapper=None, **kwargs):
            return argmap._lazy_compile(__wrapper)(*args, **kwargs)
        func.__name__ = f.__name__
        func.__doc__ = f.__doc__
        func.__defaults__ = f.__defaults__
        func.__kwdefaults__.update(f.__kwdefaults__ or {})
        func.__module__ = f.__module__
        func.__qualname__ = f.__qualname__
        func.__dict__.update(f.__dict__)
        func.__wrapped__ = f
        func.__kwdefaults__['_argmap__wrapper'] = func
        func.__self__ = func
        func.__argmap__ = self
        if hasattr(f, '__argmap__'):
            func.__is_generator = f.__is_generator
        else:
            func.__is_generator = inspect.isgeneratorfunction(f)
        if self._finally and func.__is_generator:
            raise nx.NetworkXError('argmap cannot decorate generators with try_finally')
        return func
    __count = 0

    @classmethod
    def _count(cls):
        """Maintain a globally-unique identifier for function names and "file" names

        Note that this counter is a class method reporting a class variable
        so the count is unique within a Python session. It could differ from
        session to session for a specific decorator depending on the order
        that the decorators are created. But that doesn't disrupt `argmap`.

        This is used in two places: to construct unique variable names
        in the `_name` method and to construct unique fictitious filenames
        in the `_compile` method.

        Returns
        -------
        count : int
            An integer unique to this Python session (simply counts from zero)
        """
        pass
    _bad_chars = re.compile('[^a-zA-Z0-9_]')

    @classmethod
    def _name(cls, f):
        """Mangle the name of a function to be unique but somewhat human-readable

        The names are unique within a Python session and set using `_count`.

        Parameters
        ----------
        f : str or object

        Returns
        -------
        name : str
            The mangled version of `f.__name__` (if `f.__name__` exists) or `f`

        """
        pass

    def compile(self, f):
        """Compile the decorated function.

        Called once for a given decorated function -- collects the code from all
        argmap decorators in the stack, and compiles the decorated function.

        Much of the work done here uses the `assemble` method to allow recursive
        treatment of multiple argmap decorators on a single decorated function.
        That flattens the argmap decorators, collects the source code to construct
        a single decorated function, then compiles/executes/returns that function.

        The source code for the decorated function is stored as an attribute
        `_code` on the function object itself.

        Note that Python's `compile` function requires a filename, but this
        code is constructed without a file, so a fictitious filename is used
        to describe where the function comes from. The name is something like:
        "argmap compilation 4".

        Parameters
        ----------
        f : callable
            The function to be decorated

        Returns
        -------
        func : callable
            The decorated file

        """
        pass

    def assemble(self, f):
        """Collects components of the source for the decorated function wrapping f.

        If `f` has multiple argmap decorators, we recursively assemble the stack of
        decorators into a single flattened function.

        This method is part of the `compile` method's process yet separated
        from that method to allow recursive processing. The outputs are
        strings, dictionaries and lists that collect needed info to
        flatten any nested argmap-decoration.

        Parameters
        ----------
        f : callable
            The function to be decorated.  If f is argmapped, we assemble it.

        Returns
        -------
        sig : argmap.Signature
            The function signature as an `argmap.Signature` object.
        wrapped_name : str
            The mangled name used to represent the wrapped function in the code
            being assembled.
        functions : dict
            A dictionary mapping id(g) -> (mangled_name(g), g) for functions g
            referred to in the code being assembled. These need to be present
            in the ``globals`` scope of ``exec`` when defining the decorated
            function.
        mapblock : list of lists and/or strings
            Code that implements mapping of parameters including any try blocks
            if needed. This code will precede the decorated function call.
        finallys : list of lists and/or strings
            Code that implements the finally blocks to post-process the
            arguments (usually close any files if needed) after the
            decorated function is called.
        mutable_args : bool
            True if the decorator needs to modify positional arguments
            via their indices. The compile method then turns the argument
            tuple into a list so that the arguments can be modified.
        """
        pass

    @classmethod
    def signature(cls, f):
        """Construct a Signature object describing `f`

        Compute a Signature so that we can write a function wrapping f with
        the same signature and call-type.

        Parameters
        ----------
        f : callable
            A function to be decorated

        Returns
        -------
        sig : argmap.Signature
            The Signature of f

        Notes
        -----
        The Signature is a namedtuple with names:

            name : a unique version of the name of the decorated function
            signature : the inspect.signature of the decorated function
            def_sig : a string used as code to define the new function
            call_sig : a string used as code to call the decorated function
            names : a dict keyed by argument name and index to the argument's name
            n_positional : the number of positional arguments in the signature
            args : the name of the VAR_POSITIONAL argument if any, i.e. \\*theseargs
            kwargs : the name of the VAR_KEYWORDS argument if any, i.e. \\*\\*kwargs

        These named attributes of the signature are used in `assemble` and `compile`
        to construct a string of source code for the decorated function.

        """
        pass
    Signature = collections.namedtuple('Signature', ['name', 'signature', 'def_sig', 'call_sig', 'names', 'n_positional', 'args', 'kwargs'])

    @staticmethod
    def _flatten(nestlist, visited):
        """flattens a recursive list of lists that doesn't have cyclic references

        Parameters
        ----------
        nestlist : iterable
            A recursive list of objects to be flattened into a single iterable

        visited : set
            A set of object ids which have been walked -- initialize with an
            empty set

        Yields
        ------
        Non-list objects contained in nestlist

        """
        pass
    _tabs = ' ' * 64

    @staticmethod
    def _indent(*lines):
        """Indent list of code lines to make executable Python code

        Indents a tree-recursive list of strings, following the rule that one
        space is added to the tab after a line that ends in a colon, and one is
        removed after a line that ends in an hashmark.

        Parameters
        ----------
        *lines : lists and/or strings
            A recursive list of strings to be assembled into properly indented
            code.

        Returns
        -------
        code : str

        Examples
        --------

            argmap._indent(*["try:", "try:", "pass#", "finally:", "pass#", "#",
                             "finally:", "pass#"])

        renders to

            '''try:
             try:
              pass#
             finally:
              pass#
             #
            finally:
             pass#'''
        """
        pass

def deprecate_positional_args(*, version=None):
    """Decorator to enforce keyword-only arguments.

    Using this decorator on a function will raise a TypeError if positional arguments
    are used instead of keyword arguments for any parameter after the first one.

    Parameters
    ----------
    version : str, optional
        The version in which positional arguments will be fully deprecated.
        If None, positional arguments are deprecated immediately.

    Returns
    -------
    _deprecate_positional_args : function
        Function that enforces keyword-only arguments.

    Examples
    --------
    >>> @deprecate_positional_args(version='3.4')
    ... def add(a, b=2, c=3):
    ...     return a + b + c
    >>> add(1, b=2, c=3)  # OK
    6
    >>> add(1, 2, c=3)  # Raises TypeError
    Traceback (most recent call last):
        ...
    TypeError: After the first argument, only keyword arguments are allowed.
    """
    def _deprecate_positional_args(func):
        sig = signature(func)
        first_param = next(iter(sig.parameters.values()))
        has_var_args = any(p.kind == Parameter.VAR_POSITIONAL for p in sig.parameters.values())

        @wraps(func)
        def inner(*args, **kwargs):
            if len(args) > 1 and not has_var_args:
                warnings.warn(
                    f"After the first argument, only keyword arguments are allowed. "
                    f"Using positional arguments will be deprecated in version {version}.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if version is None:
                    raise TypeError("After the first argument, only keyword arguments are allowed.")
            return func(*args, **kwargs)

        return inner

    return _deprecate_positional_args