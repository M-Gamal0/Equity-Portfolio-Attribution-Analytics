"""One-time development extraction; production code never loads notebooks.

Copies selected function definitions and their literal constants, without executing
notebook cells. Existing generated files are deliberately not overwritten.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
from pathlib import Path
import symtable

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src" / "portfolio_analytics"

# Notebook cell indexes are zero based. Each topic becomes a small public module.
GROUPS = {
    "06": ("data", {
        "contracts": [3, 5, 7, 11, 12, 14], "integration": [18],
    }),
    "07": ("performance", {
        "inputs": [1], "wealth": [2], "periods": [3],
        "relative": [4, 5], "windows": [6, 7, 8], "rolling": [9, 10],
        "capture": [11], "risk_adjusted": [12, 13],
        "contributions": [14, 15, 16], "return_basis": [17],
        "summaries": [18], "distributions": [20],
    }),
    "08": ("attribution", {
        "inputs": [1], "security": [2], "sectors": [3, 4],
        "brinson": [5], "carino": [8], "menchero": [10],
        "windows": [12], "periods": [13, 14], "linking": [15],
        "summaries": [16], "cumulative": [19],
    }),
    "09": ("risk", {
        "coverage": [1], "inputs": [2], "absolute": [3], "active": [4],
        "rolling": [5], "drawdowns": [6, 7], "tails": [8],
        "covariance": [9], "security": [10], "contributions": [11, 12, 13],
        "fixed_weights": [14], "windows": [15],
    }),
    "10": ("exposures", {
        "inputs": [1], "security": [2, 4], "sectors": [3, 5],
        "active_share": [6], "concentration": [7, 8], "overlap": [9],
        "changes": [10], "rankings": [11], "validation": [12, 13],
        "windows": [14], "chart_data": [15, 16],
    }),
    "11": ("factors", {
        "inputs": [1, 2], "ols": [3], "inference": [4, 5, 6, 7],
        "security": [8, 9], "lagging": [10], "exposures": [11],
        "attribution": [12, 13], "linking": [14], "direct": [16],
        "risk": [17], "comparison": [18], "rolling": [19], "sectors": [21],
    }),
    "12": ("turnover", {
        "inputs": [1], "schedule": [3], "implementation": [4],
        "trades": [5, 15], "sectors": [7], "drift": [9],
        "transitions": [11], "periods": [13, 16],
        "concentration": [17], "costs": [20],
    }),
}
EXCLUDE = {
    "run_external_contract_scenario", "assert_period_return", "maximum_absolute",
    "format_return", "format_pp", "format_ratio", "format_capture",
    "style_rolling_axes", "plot_cumulative_brinson",
    "expect_concentration_rejection", "expect_sector_concentration_rejection",
    "expect_holdings_overlap_rejection", "expect_exposure_change_rejection",
    "plot_historical_exposure_comparison", "attach_factor_regression_provenance",
    "append_synthetic_holding", "expect_portable_reporting_failure",
}
IMPORTS = {
    "np": "import numpy as np", "pd": "import pandas as pd",
    "json": "import json", "hashlib": "import hashlib",
    "Mapping": "from collections.abc import Mapping",
    "sliding_window_view": "from numpy.lib.stride_tricks import sliding_window_view",
    "stats": "from scipy import stats", "norm": "from scipy.stats import norm",
    "student_t": "from scipy.stats import t as student_t",
    "chi2": "from scipy.stats import chi2", "fisher_f": "from scipy.stats import f as fisher_f",
}


def global_reads(table):
    names = {s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced()}
    for child in table.get_children():
        names |= global_reads(child)
    return names


def main():
    provenance = []
    for prefix, (domain, groups) in GROUPS.items():
        notebook = next((ROOT / "local/notebooks").glob(prefix + "*.ipynb"))
        raw = notebook.read_bytes()
        cells = json.loads(raw)["cells"]
        functions, constants = {}, {}
        for i, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            constants[target.id] = node.value
                if isinstance(node, ast.FunctionDef) and node.name not in EXCLUDE:
                    for module, indexes in groups.items():
                        if i in indexes:
                            functions[node.name] = (module, i, node, ast.get_source_segment(source, node))

        def literal(name, seen=()):
            if name in seen or name not in constants:
                raise ValueError(f"Unresolved notebook dependency: {domain}.{name}")
            node = constants[name]
            if isinstance(node, ast.Name):
                return literal(node.id, (*seen, name))
            return ast.literal_eval(node)

        folder = DEST / domain
        folder.mkdir(parents=True, exist_ok=True)
        exports = []
        for module in groups:
            selected = [(name, spec) for name, spec in functions.items() if spec[0] == module]
            imports, values = {"import pandas as pd"}, {}
            sources = []
            for name, (_, cell, node, source) in selected:
                table = symtable.symtable(source, str(notebook), "exec").get_children()[0]
                globals_ = global_reads(table) - set(dir(builtins))
                for dep in sorted(globals_):
                    if dep in IMPORTS:
                        imports.add(IMPORTS[dep])
                    elif dep in functions:
                        dependency_module = functions[dep][0]
                        if dependency_module != module:
                            imports.add(f"from .{dependency_module} import {dep}")
                    else:
                        values[dep] = literal(dep)
                if any(isinstance(n, ast.Name) and n.id == "Mapping" for n in ast.walk(node.args)):
                    imports.add(IMPORTS["Mapping"])
                sources.append(f"# Research reference: {notebook.name}, code cell index {cell}.\n{source}")
                exports.append((module, name))
                provenance.append({
                    "domain": domain, "module": module, "function": name,
                    "notebook": notebook.name, "cell_index": cell,
                    "notebook_sha256": hashlib.sha256(raw).hexdigest(),
                    "function_ast_sha256": hashlib.sha256(ast.dump(node).encode()).hexdigest(),
                })
            target = folder / f"{module}.py"
            if target.exists():
                raise FileExistsError(target)
            header = f'"""{domain.title()} {module.replace("_", " ")}: validated research calculations."""\n\nfrom __future__ import annotations\n\n'
            target.write_text(header + "\n".join(sorted(imports)) + "\n\n" +
                              "\n".join(f"{k} = {v!r}" for k, v in sorted(values.items())) +
                              "\n\n" + "\n\n\n".join(sources) + "\n", encoding="utf-8")
        # Preserve the existing data package's loader export.
        init = f'"""Independently callable {domain} calculations."""\n\n'
        if domain == "data":
            init += "from .snapshots import load_standardized_inputs\n"
        init += "\n".join(f"from .{module} import {name} as {name}" for module, name in exports)
        init += "\n\n__all__ = " + repr([name for _, name in exports if not name.startswith("_")] +
                                      (["load_standardized_inputs"] if domain == "data" else [])) + "\n"
        (folder / "__init__.py").write_text(init, encoding="utf-8")
        print(domain, len(exports), "functions extracted")
    (ROOT / "docs/kernel_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
