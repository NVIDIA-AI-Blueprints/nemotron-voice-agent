# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The guard against hardcoding: what it finds in the code, read with ``ast``, never imported.

Every extension point is an interface whose implementations are named in config and picked by a composition root, so
no code branches on a kind. The guard walks the modules the import rules walk (``src/live``,
``src/examples/frontend_backend_live`` without its parked code, ``src/delegation_core`` and ``src/server.py``) and
finds four kinds of site:

* ``compare``: a comparison of a kind attribute (``.mode``, ``.target``, ``.provider``, ``.kind``, ``.adapter``,
  ``.delegation.type``, ``.transport``, ``.agent``, ...: :data:`KIND_ATTRIBUTES`), of a kind looked up by its name
  (``names["agent"]``), of a kind's own name (``settings.agent.name``), or of a local name that holds one of the
  kinds (``mode``, ``target``, ``provider``, ``adapter``, ``agent``: :data:`KIND_NAMES`), with a string,
  an UPPER_CASE name or an UPPER_CASE member, or a ``match`` on one with such a ``case``, outside the table modules;
* ``isinstance``: an ``isinstance`` or ``issubclass`` check against a class of another module of the repository
  (imported by name, or reached through its module), unless it is a port type, an exception, or the check sits in a
  composition root or the core's builder;
* ``flag``: a banned flag (:data:`BANNED_FLAGS`) as an attribute, an argument, a keyword or a field;
* ``pick``: a ``pick()`` from a table outside the composition roots, the core's builder and a ``from_settings``,
  ``from_config`` or ``parse`` method: choices are made once, when the object is built.

The test fails on any site.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
WALKED = ("live", "examples/frontend_backend_live", "delegation_core", "server.py")
PARKED = "examples.frontend_backend_live.parked"

# The kinds a config names: comparing one against a literal is a branch on a kind.
KIND_ATTRIBUTES = frozenset(
    {
        "mode",
        "target",
        "provider",
        "adapter",
        "transport",
        "protocol",
        "conductor",
        "when_idle",
        "steer",
        "session",
        "permission",
        "agent",
        "result_format",
        "result_wording",
        "prompt_version",
        "voice_note",
        "kind",
        "tts_mode",
        "turn_end",
        "backend_calls",
        "unlisted_models",
        "one_shot_adapter",
        "work_tools",
    }
)
# Local names that hold a kind: ``mode = config.mode`` then ``mode == "client"`` is the same branch. ``kind`` is
# left out: a local ``kind`` is an official event's or command's type, read by the protocol.
KIND_NAMES = frozenset({"mode", "target", "provider", "adapter", "agent"})
# Flags whose behavior is a class or a capability: none may appear.
BANNED_FLAGS = frozenset(
    {
        "server_tools",
        "stateless",
        "hide_calls",
        "stateful",
        "one_shot",
        "merge_pending",
        "fold_before_acting",
        "status_from_ledger",
    }
)
# Port modules: an isinstance check against their types (runtime-checkable protocols) is a capability check.
PORT_MODULES = frozenset(
    {
        "live.ports",
        "delegation_core.backend.port",
        "examples.frontend_backend_live.cascade.ports",
        "examples.frontend_backend_live.adapter.contract",
    }
)
# The composition roots and the core's builder: they turn names into objects.
ROOTS = frozenset(
    {
        "server",
        "examples.frontend_backend_live.factory",
        "examples.frontend_backend_live.pipeline",
        "delegation_core.app",
    }
)
TABLE_MODULES = frozenset({"delegation_core.tables", "examples.frontend_backend_live.tables"})
# The methods in which a class may pick from a table: when it is built.
BUILD_METHODS = frozenset({"from_settings", "from_config", "parse"})
INTERNAL_TOPS = frozenset({"live", "examples", "delegation_core", "utils", "examples_registry", "server", "tracing"})


@dataclass(frozen=True)
class Site:
    """A site of one kind and name in one module."""

    module: str
    kind: str  # compare | isinstance | flag | pick
    name: str  # the attribute compared, the class checked, the flag, or "pick"


def within(name: str, prefix: str) -> bool:
    """Whether ``name`` is the module ``prefix`` or inside it."""
    return name == prefix or name.startswith(prefix + ".")


def module_name(path: Path) -> str:
    """The module a file under ``src`` is."""
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def walked_files() -> list[Path]:
    """The files the guard reads: the walked trees, without the parked code."""
    files = []
    for name in WALKED:
        root = SRC / name
        files += [root] if root.is_file() else sorted(root.rglob("*.py"))
    return [path for path in files if not within(module_name(path), PARKED)]


def _is_kind(node: ast.AST) -> str | None:
    """The kind ``node`` reads, or ``None``.

    A kind attribute (``x.mode``), a kind looked up by its name (``names["agent"]``, as ``agent``), a kind's own name
    (``x.agent.name``, as ``agent.name``), ``x.delegation.type`` or ``delegation.type`` (as ``delegation.type``), or a
    local name of :data:`KIND_NAMES`.
    """
    if isinstance(node, ast.Name):
        return node.id if node.id in KIND_NAMES else None
    if isinstance(node, ast.Subscript):
        key = node.slice
        return key.value if isinstance(key, ast.Constant) and key.value in KIND_ATTRIBUTES else None
    if not isinstance(node, ast.Attribute):
        return None
    if node.attr in KIND_ATTRIBUTES:
        return node.attr
    owner = node.value
    if node.attr == "name" and (kind := _is_kind(owner)):
        return f"{kind}.name"
    owner_name = owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, "id", None)
    if node.attr == "type" and owner_name == "delegation":
        return "delegation.type"
    return None


def _is_literal(node: ast.AST) -> bool:
    """A string, an UPPER_CASE name or member (``CLIENT``, ``Mode.CLIENT``), or a collection of them (for ``in``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.Name | ast.Attribute) and (getattr(node, "id", None) or node.attr).isupper():
        return True
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return bool(node.elts) and all(_is_literal(element) for element in node.elts)
    return False


def _matches_a_literal(pattern: ast.AST) -> bool:
    """A ``case`` pattern that matches a string or an UPPER_CASE member: a branch on a kind's name."""
    if isinstance(pattern, ast.MatchValue):
        return _is_literal(pattern.value)
    if isinstance(pattern, ast.MatchOr):
        return any(_matches_a_literal(each) for each in pattern.patterns)
    if isinstance(pattern, ast.MatchAs) and pattern.pattern is not None:
        return _matches_a_literal(pattern.pattern)
    return False


def _module_aliases(tree: ast.AST) -> dict[str, str]:
    """Each name a plain ``import`` binds, and its module (``import a.b`` binds ``a``; ``as c`` binds ``c``)."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                aliases[alias.asname or top] = alias.name if alias.asname else top
    return aliases


def _class_reference(node: ast.AST, imports: dict[str, str], aliases: dict[str, str]) -> tuple[str | None, str, bool]:
    """A checked class as ``(name, the module it comes from, named bare)``; the module is ``""`` when unknown."""
    if isinstance(node, ast.Name):
        return node.id, imports.get(node.id, ""), True
    if not isinstance(node, ast.Attribute):
        return None, "", False
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return parts[0], "", False
    head, *middle, name = [node.id, *reversed(parts)]
    if head in aliases:
        base = aliases[head]
    elif head in imports:
        base = f"{imports[head]}.{head}" if imports[head] else head  # ``from a import b`` then ``b.Class``
    else:
        return name, "", False
    return name, ".".join([base, *middle]), False


def _imported_from(tree: ast.AST, module: str, is_package: bool) -> dict[str, str]:
    """Each name a ``from`` import binds, and the module it came from."""
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                parts = module.split(".") if is_package else module.split(".")[:-1]
                parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join([*parts, node.module] if node.module else parts)
            else:
                base = node.module or ""
            for alias in node.names:
                names[alias.asname or alias.name] = base
    return names


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return node.name
    return None


def sites_in(source: str, module: str, is_package: bool = False) -> list[tuple[int, Site]]:
    """Every site in ``source`` as ``(line, Site)``, the module named ``module``."""
    tree = ast.parse(source)
    imports = _imported_from(tree, module, is_package)
    aliases = _module_aliases(tree)
    local_classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    parents = _parents(tree)
    found: list[tuple[int, Site]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and module not in TABLE_MODULES:
            sides = [node.left, *node.comparators]
            kinds = [kind for side in sides if (kind := _is_kind(side))]
            if kinds and any(_is_literal(side) for side in sides):
                found.append((node.lineno, Site(module, "compare", kinds[0])))
        elif isinstance(node, ast.Match) and module not in TABLE_MODULES:
            kind = _is_kind(node.subject)
            if kind and any(_matches_a_literal(case.pattern) for case in node.cases):
                found.append((node.lineno, Site(module, "compare", kind)))
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) in ("isinstance", "issubclass")
            and len(node.args) == 2
            and module not in ROOTS
        ):
            classes = node.args[1].elts if isinstance(node.args[1], ast.Tuple) else [node.args[1]]
            for cls in classes:
                name, source_module, bare = _class_reference(cls, imports, aliases)
                if (
                    name is None
                    or (bare and (hasattr(builtins, name) or name in local_classes))
                    or name.endswith(("Error", "Exception"))
                    or source_module in PORT_MODULES
                    or source_module == module
                    or source_module.split(".")[0] not in INTERNAL_TOPS
                ):
                    continue
                found.append((node.lineno, Site(module, "isinstance", name)))
        elif (
            isinstance(node, ast.Call)
            and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == "pick"
        ):
            if module not in ROOTS and _enclosing_function(node, parents) not in BUILD_METHODS:
                found.append((node.lineno, Site(module, "pick", "pick")))
        flag = None
        if isinstance(node, ast.Attribute) and node.attr in BANNED_FLAGS:
            flag = node.attr
        elif isinstance(node, ast.keyword | ast.arg) and node.arg in BANNED_FLAGS:
            flag = node.arg
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in BANNED_FLAGS:
            flag = node.target.id
        if flag:
            found.append((getattr(node, "lineno", 0), Site(module, "flag", flag)))
    return sorted(found, key=lambda pair: (pair[0], pair[1].kind, pair[1].name))


@cache
def walked_sites() -> tuple[tuple[str, int, Site], ...]:
    """Every site of the walked modules, as ``(path, line, Site)``."""
    found = []
    for path in walked_files():
        module = module_name(path)
        for line, site in sites_in(path.read_text(encoding="utf-8"), module, path.name == "__init__.py"):
            found.append((str(path.relative_to(REPO)), line, site))
    return tuple(found)
