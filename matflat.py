"""
matflat — flatten deeply nested MATLAB .mat structures into pandas.

Reader-agnostic: works on output from mat-io, scipy.io.loadmat, pymatreader,
mat73, or on an already-loaded dict.

Core ideas
----------
1. Normalize first. Every reader hands back a different mess (structured
   object arrays, mat_struct objects, h5py references). `normalize()` turns
   all of them into plain dicts / lists / ndarrays / str.
2. Struct arrays become **list-of-dicts**, not dict-of-lists, so
   `trials(i).emg.rms` in MATLAB is `d['trials'][i]['emg']['rms']` in Python.
3. Then walk to the leaves and reshape into DataFrames.

Typical use
-----------
    import matflat as mf

    d = mf.load("subject.mat")
    mf.inventory(d)              # what's in here, one row per leaf
    frames = mf.tabulate(d)      # dict of DataFrames
    mf.to_long(d)                # one tidy frame, everything stacked
    mf.get(d, "subject.trials[0].emg.rms")
"""

from __future__ import annotations

import fnmatch
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "load", "normalize", "walk", "get", "inventory",
    "tabulate", "to_long", "to_wide",
]

_SCALAR_TYPES = (str, bytes, bool, int, float, complex, np.generic)


# ---------------------------------------------------------------- loading

def load(path, reader: str = "auto", **kwargs):
    """Read a .mat file and return a normalized nested structure.

    reader : "auto" | "matio" | "scipy" | "pymatreader" | "mat73" | "h5py"
    """
    raw = _read_raw(path, reader, **kwargs)
    raw = {k: v for k, v in raw.items() if not k.startswith("__")}
    return normalize(raw)


def _read_raw(path, reader, **kwargs):
    path = str(path)
    order = (
        ["matio", "pymatreader", "scipy", "mat73", "h5py"]
        if reader == "auto" else [reader]
    )
    errs = []
    for name in order:
        try:
            if name == "matio":
                from matio import load_from_mat
                return load_from_mat(path, **kwargs)
            if name == "pymatreader":
                from pymatreader import read_mat
                return read_mat(path, **kwargs)
            if name == "scipy":
                from scipy.io import loadmat
                return loadmat(path, struct_as_record=True,
                               squeeze_me=False, **kwargs)
            if name == "mat73":
                import mat73
                return mat73.loadmat(path, **kwargs)
            if name == "h5py":
                import h5py
                with h5py.File(path, "r") as f:
                    return {k: _h5_to_py(f[k], f) for k in f.keys()}
            raise ValueError(f"unknown reader {name!r}")
        except Exception as e:  # noqa: BLE001 - try the next reader
            errs.append(f"{name}: {type(e).__name__}: {e}")
    raise OSError("no reader succeeded:\n  " + "\n  ".join(errs))


def _h5_to_py(node, fh):
    """Minimal v7.3 fallback: deref object references, decode char arrays."""
    import h5py

    if isinstance(node, h5py.Group):
        keys = [k for k in node.keys() if not k.startswith("#")]
        return {k: _h5_to_py(node[k], fh) for k in keys}

    arr = node[()]
    cls = node.attrs.get("MATLAB_class", b"")
    cls = cls.decode() if isinstance(cls, bytes) else cls
    if cls == "char":
        a = np.atleast_2d(np.asarray(arr))
        return "".join(chr(int(c)) for c in a.ravel(order="F") if int(c))
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        out = [_h5_to_py(fh[r], fh) for r in arr.ravel(order="F")]
        return out[0] if len(out) == 1 else out
    if isinstance(arr, np.ndarray) and arr.ndim >= 2:
        return arr.T  # undo MATLAB column-major
    return arr


# ------------------------------------------------------------ normalizing

def normalize(obj, squeeze: bool = True):
    """Recursively convert any reader's output into plain Python types.

    struct scalar -> dict
    struct array  -> list[dict]        (one dict per element)
    cell array    -> list
    char array    -> str
    numeric       -> np.ndarray (squeezed) or Python scalar
    """
    # scipy's struct_as_record=False objects
    if hasattr(obj, "_fieldnames"):
        return {f: normalize(getattr(obj, f), squeeze) for f in obj._fieldnames}

    if isinstance(obj, Mapping):
        return {str(k): normalize(v, squeeze) for k, v in obj.items()}

    if isinstance(obj, np.ndarray):
        # struct / struct array
        if obj.dtype.names:
            recs = [
                {n: normalize(el[n], squeeze) for n in obj.dtype.names}
                for el in obj.ravel(order="F")
            ]
            return recs[0] if obj.size == 1 else recs
        # cell array
        if obj.dtype == object:
            items = [normalize(el, squeeze) for el in obj.ravel(order="F")]
            if obj.size == 1:
                return items[0]
            return items
        # char
        if obj.dtype.kind in "US":
            if obj.size == 1:
                return str(obj.ravel()[0])
            return ["".join(str(x)) for x in obj.ravel(order="F").tolist()]
        # numeric / bool
        a = np.squeeze(obj) if squeeze else obj
        if a.ndim == 0:
            return a.item()
        return a

    if isinstance(obj, np.void) and obj.dtype.names:
        return {n: normalize(obj[n], squeeze) for n in obj.dtype.names}

    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")

    if isinstance(obj, (list, tuple)):
        return [normalize(v, squeeze) for v in obj]

    return obj


# --------------------------------------------------------------- walking

def _is_leaf(v) -> bool:
    if isinstance(v, Mapping):
        return False
    if isinstance(v, list) and v and all(isinstance(x, Mapping) for x in v):
        return False
    return True


def walk(obj, prefix: str = "", *, leaves_only: bool = True):
    """Yield (path, value) pairs. Paths look like `a.b[2].c`."""
    if isinstance(obj, Mapping):
        if not leaves_only and prefix:
            yield prefix, obj
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}" if prefix else str(k),
                            leaves_only=leaves_only)
    elif isinstance(obj, list) and obj and all(isinstance(x, Mapping) for x in obj):
        if not leaves_only and prefix:
            yield prefix, obj
        for i, v in enumerate(obj):
            yield from walk(v, f"{prefix}[{i}]", leaves_only=leaves_only)
    else:
        yield prefix, obj


def get(obj, path: str):
    """Fetch by dotted/indexed path: `subject.trials[0].emg.rms`."""
    cur = obj
    for tok in path.replace("[", ".[").split("."):
        if not tok:
            continue
        if tok.startswith("["):
            cur = cur[int(tok[1:-1])]
        else:
            cur = cur[tok]
    return cur


# ------------------------------------------------------------- inventory

def _kind(v) -> str:
    if isinstance(v, str):
        return "str"
    if isinstance(v, _SCALAR_TYPES):
        return "scalar"
    if isinstance(v, np.ndarray):
        return f"{v.ndim}d-array"
    if isinstance(v, list):
        return "list"
    return type(v).__name__


def inventory(obj, pattern: str | None = None) -> pd.DataFrame:
    """One row per leaf: path, kind, shape, dtype, n. Use this first."""
    rows = []
    for path, v in walk(obj):
        if pattern and not fnmatch.fnmatch(path, pattern):
            continue
        arr = v if isinstance(v, np.ndarray) else None
        rows.append({
            "path": path,
            "field": path.split(".")[-1],
            "parent": path.rsplit(".", 1)[0] if "." in path else "",
            "kind": _kind(v),
            "shape": tuple(arr.shape) if arr is not None else (),
            "dtype": str(arr.dtype) if arr is not None
                     else type(v).__name__,
            "n": int(arr.size) if arr is not None
                 else (len(v) if isinstance(v, (list, str)) else 1),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------- reshaping

def _flat_scalars(d: Mapping, prefix: str = "") -> dict:
    """All scalar/str leaves beneath a dict, keyed by dotted name."""
    out = {}
    for path, v in walk(d, prefix):
        if isinstance(v, _SCALAR_TYPES) and not isinstance(v, (bytes,)):
            out[path] = v
    return out


def _struct_array_frame(recs: Sequence[Mapping]) -> pd.DataFrame:
    """list-of-dicts -> one row per element, scalar leaves as columns."""
    rows = [_flat_scalars(r) for r in recs]
    df = pd.DataFrame(rows)
    df.insert(0, "_i", range(len(recs)))
    return df


def _columnar_frame(parent_path: str, obj, labels_field: str | None = None):
    """Sibling 1-D leaves of equal length -> DataFrame columns."""
    cols, scalars, lengths = {}, {}, set()
    for k, v in obj.items():
        if isinstance(v, np.ndarray) and v.ndim == 1:
            cols[k] = v
            lengths.add(v.size)
        elif isinstance(v, _SCALAR_TYPES):
            scalars[k] = v
    if len(lengths) != 1 or not cols:
        return None
    df = pd.DataFrame(cols)
    for k, v in scalars.items():
        df[k] = v
    return df


def _matrix_frame(name: str, arr: np.ndarray, labels=None,
                  orient: str = "auto") -> pd.DataFrame:
    """2-D array -> DataFrame, auto-oriented so the long axis is rows.

    Column naming, in priority order:
      1. a sibling cell of strings (`labels`) whose length matches an axis —
         that axis becomes the columns, transposing if needed
      2. orient="auto": the *shorter* axis becomes the columns, so a
         4 x 50000 NMF activation matrix lands as 50000 rows x 4 columns
         instead of a 50000-column frame
      3. orient="asis": keep MATLAB's orientation untouched
    """
    a = arr if arr.ndim == 2 else arr.reshape(arr.shape[0], -1)

    if labels is not None:
        n = len(labels)
        if n == a.shape[1] and n != a.shape[0]:
            return pd.DataFrame(a, columns=[str(x) for x in labels])
        if n == a.shape[0] and n != a.shape[1]:
            return pd.DataFrame(a.T, columns=[str(x) for x in labels])
        if n == a.shape[1] == a.shape[0]:  # square: believe MATLAB
            return pd.DataFrame(a, columns=[str(x) for x in labels])

    if orient == "auto" and a.shape[1] > a.shape[0]:
        a = a.T
    return pd.DataFrame(a, columns=[f"{name}_{i}" for i in range(a.shape[1])])


def tabulate(obj, *, labels_key: str = "labels", orient: str = "auto",
             max_matrix_cols: int | None = None) -> dict[str, pd.DataFrame]:
    """Reshape a nested structure into a dict of DataFrames.

    Produces, keyed by path:
      - `<path>` for every struct array  -> one row per element,
        all scalar leaves beneath flattened into dotted columns
      - `<path>` for every dict whose 1-D leaves share a length
        -> columnar frame (scalars broadcast)
      - `<path>` for every 2-D array -> matrix frame, oriented long-axis-as-rows
        and named from a sibling cell `labels_key` when the size matches

    orient : "auto" transposes so the shorter axis is columns; "asis" keeps
        MATLAB's orientation. A sibling label cell always overrides both.
    max_matrix_cols : None (default) means no cap — every matrix is emitted.
        Set an int to skip frames still wider than that after orientation;
        skipped paths raise a UserWarning rather than disappearing silently.
    """
    frames: dict[str, pd.DataFrame] = {}

    def visit(node, path):
        if isinstance(node, list) and node and all(isinstance(x, Mapping) for x in node):
            f = _struct_array_frame(node)
            if f.shape[1] > 1:
                frames[path or "root"] = f
            for i, el in enumerate(node):
                visit(el, f"{path}[{i}]")
            return

        if isinstance(node, Mapping):
            f = _columnar_frame(path, node)
            if f is not None:
                frames[path or "root"] = f
            labels = node.get(labels_key)
            if isinstance(labels, str):
                labels = [labels]
            for k, v in node.items():
                child = f"{path}.{k}" if path else k
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    f = _matrix_frame(k, v, labels, orient)
                    if max_matrix_cols is not None and f.shape[1] > max_matrix_cols:
                        warnings.warn(
                            f"{child}: {f.shape[1]} columns after orientation "
                            f"exceeds max_matrix_cols={max_matrix_cols}; skipped",
                            UserWarning, stacklevel=2,
                        )
                    else:
                        frames[child] = f
                else:
                    visit(v, child)
            return

    visit(obj, "")
    return frames


def to_long(obj, *, include_matrices: bool = True) -> pd.DataFrame:
    """Everything stacked into one tidy frame: path, i, j, value.

    Scalars get i=j=NaN, vectors get i, matrices get i and j.
    Safe on ragged data — this is the format to use when nothing lines up.
    """
    rows = []
    for path, v in walk(obj):
        if isinstance(v, str) or isinstance(v, _SCALAR_TYPES):
            rows.append((path, np.nan, np.nan, v))
        elif isinstance(v, np.ndarray) and v.ndim == 1:
            rows.extend((path, i, np.nan, x) for i, x in enumerate(v.tolist()))
        elif isinstance(v, np.ndarray) and v.ndim == 2 and include_matrices:
            for i in range(v.shape[0]):
                rows.extend((path, i, j, v[i, j]) for j in range(v.shape[1]))
        elif isinstance(v, list):
            rows.extend((path, i, np.nan, x) for i, x in enumerate(v))
    df = pd.DataFrame(rows, columns=["path", "i", "j", "value"])
    df["field"] = df["path"].str.rsplit(".", n=1).str[-1]
    return df


def to_wide(obj) -> pd.DataFrame:
    """Single-row frame of every scalar leaf, dotted column names.

    Useful for stacking many subject files: pd.concat([to_wide(load(p)) ...]).
    """
    return pd.DataFrame([_flat_scalars(obj)])
