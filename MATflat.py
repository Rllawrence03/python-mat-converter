import hdf5storage
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from scipy.io.matlab import mat_struct, MatlabOpaque
from dotenv import load_dotenv


def mat_to_dataframes(path=None, show_tree=True):
    """Load a .mat file, flatten every struct/struct-array into its own
    DataFrame, and draw the resulting hierarchy as a tree chart.

    `path` defaults to the DIR env var (loaded from .env). Returns
    (tables, variable_names, fig).
    """
    load_dotenv()
    path = path or os.getenv("DIR")

    raw = hdf5storage.loadmat(path) # type: ignore

    def mat_to_py(obj):
        """Normalize MATLAB struct/cell data from scipy.io.loadmat or
        hdf5storage.loadmat into plain dicts/lists/scalars/ndarrays."""
        def decode(x):
            return x.decode() if isinstance(x, bytes) else x

        if isinstance(obj, mat_struct):
            return {name: mat_to_py(getattr(obj, name)) for name in obj._fieldnames} # type: ignore

        if isinstance(obj, MatlabOpaque):
            # MATLAB `string`/classdef values are MCOS references neither loader
            # can decode; resave as char/cellstr in MATLAB to get real data.
            rec = obj[0]
            names = obj.dtype.names or ()
            if "_Class" in names:
                return {
                    "_matlab_unsupported_class": decode(rec["_Class"]),
                    "_matlab_type_system": decode(rec["_TypeSystem"]) if "_TypeSystem" in names else None,
                }
            if "s2" in names:
                return {
                    "_matlab_unsupported_class": decode(rec["s2"]),
                    "_matlab_variable_name": decode(rec["s0"]) if "s0" in names else None,
                }
            return {"_matlab_unsupported_fields": names}

        if isinstance(obj, np.void):
            return {name: mat_to_py(obj[name]) for name in obj.dtype.names} # type: ignore

        if isinstance(obj, np.ndarray):
            if obj.dtype.names:
                items = [mat_to_py(rec) for rec in obj.ravel()]
                return items[0] if len(items) == 1 else items
            is_cell = obj.dtype == object
            squeezed = np.squeeze(obj)
            if squeezed.ndim == 0:
                return mat_to_py(squeezed.item()) if is_cell else squeezed.item()
            return [mat_to_py(x) for x in squeezed.ravel()] if is_cell else squeezed

        return obj

    data = {k: mat_to_py(v) for k, v in raw.items() if not k.startswith("__")}

    tables = {}
    variable_names = []
    name_counts = {}

    def assign_name(path_):
        """Name a variable after the leaf field in its struct path (e.g.
        'emgMatrix.hierarchy.emgBinNames' -> 'emgBinNames'), sanitized and
        deduped with a numeric suffix since MATLAB reuses field names at
        different nesting depths."""
        leaf = path_.rsplit(".", 1)[-1]
        name = re.sub(r"[^0-9a-zA-Z_]", "_", leaf)
        name = re.sub(r"_+", "_", name).strip("_") or "_"
        if name[0].isdigit():
            name = "_" + name
        n = name_counts.get(name, 0) + 1
        name_counts[name] = n
        return name if n == 1 else f"{name}_{n}"

    def hoist_structs(obj, path_):
        """Recursively hoist every struct/struct-array into its own DataFrame
        in `tables` instead of nesting it inside a parent cell. Returns
        (hoisted, result): result is the saved variable name if hoisted,
        else the original value."""
        if isinstance(obj, dict):
            items, indexed = [obj], False
        elif isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj) and all(x.keys() == obj[0].keys() for x in obj):
            items, indexed = obj, True
        else:
            return False, obj

        rows = []
        for i, item in enumerate(items):
            row = {}
            for k, v in item.items():
                child_path = f"{path_}[{i}].{k}" if indexed else f"{path_}.{k}"
                hoisted, result = hoist_structs(v, child_path)
                row[k] = f"-> {result}" if hoisted else result
            rows.append(row)

        varname = assign_name(path_)
        tables[varname] = pd.DataFrame(rows)
        variable_names.append(varname)
        return True, varname

    for name, value in data.items():
        hoisted, result = hoist_structs(value, name)
        if not hoisted:
            if isinstance(result, np.ndarray) and result.ndim > 1:
                df = pd.DataFrame(result)
            elif isinstance(result, (np.ndarray, list)):
                df = pd.DataFrame({"value": result})
            else:
                df = pd.DataFrame({"value": [result]})
            varname = assign_name(name)
            tables[varname] = df
            variable_names.append(varname)

    print(len(variable_names), "DataFrames created:")
    for name in variable_names:
        df = tables[name]
        print(f"{name}: {df.shape} cols={list(df.columns)[:6]}{'...' if len(df.columns) > 6 else ''}")

    def build_tree_edges(names):
        """Reconstruct the struct hierarchy from the '-> childname'
        breadcrumbs hoist_structs left in each DataFrame's cells."""
        children = {name: [] for name in names}
        is_child = set()
        for name in names:
            df = tables[name]
            for col in df.columns:
                for val in df[col]:
                    if isinstance(val, str) and val.startswith("-> "):
                        child = val[3:]
                        if child in children:
                            children[name].append((col, child))
                            is_child.add(child)
        roots = [name for name in names if name not in is_child]
        return children, roots

    def assign_positions(children, roots):
        """Post-order layout: leaves get sequential x positions, parents
        center over their children; y is -depth."""
        positions = {}
        next_x = [0]

        def place(node, depth):
            kids = children.get(node, [])
            if not kids:
                x = next_x[0]
                next_x[0] += 1
            else:
                x = sum(place(child, depth + 1) for _, child in kids) / len(kids)
            positions[node] = (x, -depth)
            return x

        for root in roots:
            place(root, 0)
        return positions

    def draw_mat_tree(names):
        children, roots = build_tree_edges(names)
        positions = assign_positions(children, roots)

        max_depth = max(-y for _, y in positions.values())
        n_leaves = sum(1 for name in names if not children.get(name))
        fig, ax = plt.subplots(figsize=(max(10, n_leaves * 0.9), max(6, (max_depth + 1) * 1.4)))

        for name, (x, y) in positions.items():
            kids = children.get(name, [])
            for col, child in kids:
                cx, cy = positions[child]
                ax.plot([x, cx], [y, cy], color="gray", linewidth=0.8, zorder=1)
                ax.annotate(col, ((x + cx) / 2, (y + cy) / 2), fontsize=6, color="dimgray",
                            ha="center", va="center", backgroundcolor="white")

            face = "#fdf6e3" if not kids else "#e8f0fe"
            df = tables[name]
            dtypes = "\n".join(sorted({str(dt) for dt in df.dtypes}))
            ax.text(x, y, f"{name}\n{df.shape}\n{dtypes}", fontsize=7, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc=face, ec="#4a86e8"), zorder=2)

        ax.set_xlim(-1, n_leaves)
        ax.set_ylim(-max_depth - 1, 1)
        ax.axis("off")
        ax.set_title("MAT file structure")
        fig.tight_layout()
        return fig

    fig = None
    if show_tree:
        fig = draw_mat_tree(variable_names)
        plt.show()

    return tables, variable_names, fig


if __name__ == "__main__":
    mat_to_dataframes()
