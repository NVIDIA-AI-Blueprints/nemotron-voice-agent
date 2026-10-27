# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Who may import whom: the boundary rules between the boxes, checked with ``ast`` on every walked module.

Every import counts, including those inside functions and under ``TYPE_CHECKING``, and so does a module named by a
string or passed to ``importlib.import_module``. A module belongs to the group of the longest prefix of its name in
``TARGET_GROUPS`` and may import its own group and the groups ``MAY_IMPORT`` lists; ``PACKAGE_RULES`` places the
third-party packages. Only ``delegation_core.backend`` may reach a model provider's ``stream()``, unless
``NOT_MODEL_STREAMS`` lists the module. No walked module may import the tests, and the core's own tests may import
only the core, their helpers, the contracts, the standard library, ``openai`` and ``httpx``.
"""

import ast
import re
import sys
import unittest
from dataclasses import dataclass
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
# What the rules cover, relative to src.
WALKED = ("live", "examples/frontend_backend_live", "delegation_core", "server.py")

# ----------------------------------------------------------------------------------------------------- groups
ADAPTER = "adapter"
ROOT = "composition root"
PARKED = "parked"
FBL = "examples.frontend_backend_live"

TARGET_GROUPS = {
    # The gateway. "live" alone is the package's docstring, and any module no other row places.
    "live": "live",
    "live.protocol": "live.protocol",
    "live.events": "live.events",
    "live.ports": "live.ports",
    "live.server_delegation": "live.server_delegation",
    "live.client_delegation": "live.client_delegation",
    "live.session": "live.session",
    "live.gateway": "live.gateway",
    "live.media": "live.media",
    "live.extras": "live.extras",
    "live.mount": "live.mount",
    # The example package is the frontend adapter (adapter/, cascade/, models/, prompts/, config_manager/), except
    # its two composition roots and the parked Realtime adapter.
    FBL: ADAPTER,
    f"{FBL}.factory": ROOT,
    f"{FBL}.pipeline": ROOT,
    f"{FBL}.parked": PARKED,
    # The delegation core; the constructors its tests use for the worker and the thinker are in the tests' helpers.
    "delegation_core": "delegation_core",
    "delegation_core.api": "delegation_core.api",
    "delegation_core.app": "delegation_core.app",
    "delegation_core.im": "delegation_core.im",
    "delegation_core.backend": "delegation_core.backend",
    "delegation_core.backend.port": "delegation_core.backend.port",
    # The three backend adapters, each in its own package: the backend LLM, directly connected,
    # the backend LLM agent and the harness. The rest of the backend (providers, tools, conductors, the
    # control tools and the adapter table) stays in "delegation_core.backend".
    "delegation_core.backend.llm": "delegation_core.backend.llm",
    "delegation_core.backend.agent": "delegation_core.backend.agent",
    "delegation_core.backend.harness": "delegation_core.backend.harness",
    "delegation_core.tables": "delegation_core.tables",
    "delegation_core.ledger": "delegation_core.ledger",
    "delegation_core.config": "delegation_core.config",
    "delegation_core.ids": "delegation_core.ids",
    # The server's composition root, which mounts the gateway.
    "server": ROOT,
}


# ------------------------------------------------------------------------------------------------------ rules
LIVE_GROUPS = frozenset(
    {
        "live",
        "live.protocol",
        "live.events",
        "live.ports",
        "live.server_delegation",
        "live.client_delegation",
        "live.session",
        "live.gateway",
        "live.media",
        "live.extras",
        "live.mount",
    }
)
CORE_GROUPS = frozenset(group for group in TARGET_GROUPS.values() if group.startswith("delegation_core"))

# What each group may import besides itself. Every other internal import breaks a rule.
MAY_IMPORT = {
    "live": frozenset(),
    "live.protocol": frozenset(),
    "live.events": frozenset({"live.protocol"}),
    "live.ports": frozenset({"live.protocol"}),
    "live.server_delegation": frozenset({"live.protocol", "live.ports"}),
    "live.client_delegation": frozenset({"live.protocol", "live.ports"}),
    "live.session": frozenset(
        {"live.protocol", "live.events", "live.ports", "live.server_delegation", "live.client_delegation"}
    ),
    "live.gateway": LIVE_GROUPS,
    "live.media": LIVE_GROUPS,
    "live.extras": LIVE_GROUPS,
    "live.mount": LIVE_GROUPS,
    ADAPTER: frozenset({"live.ports", "live.protocol", "examples.shared", "utils"}),
    "delegation_core": CORE_GROUPS,
    "delegation_core.api": frozenset(
        {"delegation_core.app", "delegation_core.im", "delegation_core.config", "delegation_core.ids"}
    ),
    "delegation_core.app": frozenset(
        {
            "delegation_core.im",
            "delegation_core.ledger",
            "delegation_core.backend",
            "delegation_core.backend.port",
            "delegation_core.config",
            "delegation_core.ids",
            "delegation_core.tables",
        }
    ),
    "delegation_core.im": frozenset(
        {"delegation_core.ledger", "delegation_core.backend.port", "delegation_core.config", "delegation_core.ids"}
    ),
    # The adapter table (backend/adapters.py) loads each adapter's package; each adapter reaches the shared backend.
    "delegation_core.backend": frozenset(
        {
            "delegation_core.backend.port",
            "delegation_core.backend.llm",
            "delegation_core.backend.agent",
            "delegation_core.backend.harness",
            "delegation_core.config",
            "delegation_core.ids",
            "delegation_core.tables",
        }
    ),
    "delegation_core.backend.llm": frozenset(
        {
            "delegation_core.backend",
            "delegation_core.backend.port",
            "delegation_core.config",
            "delegation_core.ids",
            "delegation_core.tables",
        }
    ),
    # The LLM agent's workers are the backend LLM's runs.
    "delegation_core.backend.agent": frozenset(
        {
            "delegation_core.backend",
            "delegation_core.backend.llm",
            "delegation_core.backend.port",
            "delegation_core.config",
            "delegation_core.ids",
            "delegation_core.tables",
        }
    ),
    "delegation_core.backend.harness": frozenset(
        {
            "delegation_core.backend",
            "delegation_core.backend.port",
            "delegation_core.config",
            "delegation_core.ids",
            "delegation_core.tables",
        }
    ),
    "delegation_core.tables": frozenset(),
    # The port (types only) gets the backend's rule without the backend.
    "delegation_core.backend.port": frozenset({"delegation_core.config", "delegation_core.ids"}),
    "delegation_core.ledger": frozenset({"delegation_core.ids"}),
    "delegation_core.config": frozenset(),
    "delegation_core.ids": frozenset(),
}

WEB_EDGES = ("live.gateway", "live.mount", "live.extras", "delegation_core.api.http")
CODEC = ("live.media.audio", "live.gateway.websocket")
# Where each outside package may be imported. Composition roots and the parked code are not checked.
PACKAGE_RULES = {
    # The adapter's Pipecat modules, and the WebRTC peer that the gateway shares with them (a documented exception).
    "pipecat": (
        "examples.frontend_backend_live.adapter",
        "examples.frontend_backend_live.cascade.processor",
        "live.gateway.webrtc",
        "live.media.connection",
    ),
    # The core's providers and the adapter's talker models.
    "openai": ("delegation_core.backend.providers", "examples.frontend_backend_live.models"),
    # The HTTP edges.
    "fastapi": WEB_EDGES,
    "starlette": WEB_EDGES,
    # The wire codec, and the WebSocket edge that builds it.
    "av": CODEC,
    "numpy": CODEC,
    "yaml": ("examples.frontend_backend_live.config_manager", "delegation_core.config"),
    # The harness's protocol packages: only the harness adapter imports them.
    "mcp": ("delegation_core.backend.harness",),
    "websockets": ("delegation_core.backend.harness",),
    "acp": ("delegation_core.backend.harness",),
}

# The backend adapter owns model rounds: only its modules reach a model provider's stream(). The
# example's POST /v1/live/delegate rounds reach the delegation core through its Responses API too, so every
# other stream is one that is no model provider's, listed here by module and receiver with the reason.
ROUND_OWNER = "delegation_core.backend"
NOT_MODEL_STREAMS = {
    ("live.gateway.security", "request"): "Starlette's Request.stream(): the HTTP request's body, read in chunks",
    ("delegation_core.api.service", "admission"): "Admission.stream: the response stream the IM gives a request",
    ("delegation_core.api.service", "self"): "StatelessStream.stream: the response stream of a stateless request",
    ("delegation_core.im.stream", "self"): "ResponseWriter.stream: the response stream the writer writes into",
}

# The tests are no group of the tables. No walked module may import them, roots and parked code included.
TESTS = "tests"
# The delegation core's own tests and their helpers, relative to the repository: they may import only the core, those
# helpers, the standard library, the OpenAI client and the HTTP types its errors carry.
CORE_TESTS = ("tests/unit/delegation_core", "tests/unit/core_helpers.py", "tests/unit/contracts")
# ``tests.unit.delegation_core``: the core's own test modules (the emulator adapter) for the core's other tests.
CORE_TESTS_MAY_IMPORT = (
    "delegation_core",
    "tests.unit.core_helpers",
    "tests.unit.contracts",
    "tests.unit.delegation_core",
)
CORE_TESTS_PACKAGES = ("openai", "httpx")

# The harness's ACP client: each agent class with ways of its own is a package in its folder, and only the agents
# table loads one.
ACP_CLIENT = "delegation_core.backend.harness.acp"
ACP_CLIENT_DIR = SRC / "delegation_core" / "backend" / "harness" / "acp"
AGENTS_TABLE = f"{ACP_CLIENT}.agents"


# The modules that may load a module whose name they compute, and why. None: the mount loads nothing by name; the
# registry (examples_registry.resolve_hook, which the walk does not read) loads the live: hooks and LIVE_ADAPTER_FACTORY
# for the server, as it loads each example's bot.
BY_NAME_LOADERS: dict[str, str] = {}
LOADERS = frozenset({"import_module", "__import__"})
BY_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")


# --------------------------------------------------------------------------------------------------- the walk
@dataclass(frozen=True)
class Import:
    path: str
    line: int
    importer: str
    imported: str

    @property
    def where(self) -> str:
        return f"{self.path}:{self.line}"


@dataclass(frozen=True)
class Violation:
    imp: Import
    rule: str

    def __str__(self) -> str:
        """The failure line: ``<file>:<line> imports <module>: <the rule it breaks>``."""
        return f"{self.imp.where} imports {self.imp.imported}: {self.rule}"


def module_name(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def resolve(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    """The absolute module a ``from`` import names; a relative one is resolved against ``module``."""
    if node.level == 0:
        return node.module or ""
    parts = module.split(".")
    if not is_package:
        parts = parts[:-1]
    if node.level > 1:
        parts = parts[: len(parts) - (node.level - 1)]
    return ".".join([*parts, node.module] if node.module else parts)


def loader_call(node: ast.AST) -> bool:
    """Whether ``node`` calls ``importlib.import_module``, ``import_module`` or ``__import__``."""
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", None)
    return name in LOADERS


def imports_in(source: str, module: str, is_package: bool, known: frozenset[str]) -> list[tuple[int, str]]:
    """Every import in ``source`` as ``(line, module)``, sorted: at the top, in functions, under ``TYPE_CHECKING``.

    ``from package import name`` imports the module ``package.name`` when ``known`` has it, and ``package`` when it
    does not (a class or a function). A ``"module:attribute"`` text imports ``module`` when ``known`` has it, and a
    loader called with a literal name imports that name.
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = resolve(module, is_package, node)
            submodules = {f"{base}.{alias.name}" for alias in node.names} & known
            found.update((node.lineno, name) for name in submodules)
            if len(submodules) < len(node.names):
                found.add((node.lineno, base))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and BY_NAME.fullmatch(node.value):
            name = node.value.split(":")[0]
            if name in known:
                found.add((node.lineno, name))
        elif loader_call(node) and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.add((node.lineno, first.value))
    return sorted(found)


def computed_loads_in(source: str) -> list[int]:
    """The lines of ``source`` that load a module whose name is computed, not written as text."""
    return sorted(
        node.lineno
        for node in ast.walk(ast.parse(source))
        if loader_call(node) and not (node.args and isinstance(node.args[0], ast.Constant))
    )


@dataclass(frozen=True)
class StreamReference:
    path: str
    line: int
    module: str
    receiver: str  # what the attribute is read on: ``provider`` for ``provider.stream(payload)``

    @property
    def where(self) -> str:
        return f"{self.path}:{self.line}"


def names_stream(node: ast.AST) -> bool:
    """Whether ``node`` is ``getattr(a, "stream", ...)``, which reads the attribute by its literal name."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "stream"
    )


def stream_references_in(source: str) -> list[tuple[int, str]]:
    """Every read of an attribute named ``stream`` in ``source``, called or not, as ``(line, receiver)``.

    ``a.stream(...)``, ``f = a.stream`` and ``getattr(a, "stream")`` each give ``a``; setting ``a.stream`` is no read.
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr == "stream" and isinstance(node.ctx, ast.Load):
            found.add((node.lineno, ast.unparse(node.value)))
        elif names_stream(node):
            found.add((node.lineno, ast.unparse(node.args[0])))
    return sorted(found)


@cache
def walked_files() -> tuple[Path, ...]:
    files = []
    for name in WALKED:
        root = SRC / name
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.py")))
    return tuple(files)


@cache
def walked_modules() -> frozenset[str]:
    return frozenset(module_name(path) for path in walked_files())


@cache
def all_imports() -> tuple[Import, ...]:
    known = walked_modules()
    found = []
    for path in walked_files():
        module = module_name(path)
        source = path.read_text(encoding="utf-8")
        for line, imported in imports_in(source, module, path.name == "__init__.py", known):
            found.append(Import(str(path.relative_to(REPO)), line, module, imported))
    return tuple(found)


@cache
def computed_loads() -> tuple[Import, ...]:
    found = []
    for path in walked_files():
        for line in computed_loads_in(path.read_text(encoding="utf-8")):
            found.append(Import(str(path.relative_to(REPO)), line, module_name(path), "<a computed name>"))
    return tuple(found)


@cache
def stream_references() -> tuple[StreamReference, ...]:
    """Every read of an attribute named ``stream`` in the walked modules, roots and parked code included."""
    found = []
    for path in walked_files():
        for line, receiver in stream_references_in(path.read_text(encoding="utf-8")):
            found.append(StreamReference(str(path.relative_to(REPO)), line, module_name(path), receiver))
    return tuple(found)


@cache
def internal_tops() -> frozenset[str]:
    """The repository's own top-level modules and packages under src; every other import is an outside package."""
    names = {path.stem for path in SRC.glob("*.py")}
    names |= {path.name for path in SRC.iterdir() if path.is_dir() and path.name.isidentifier()}
    return frozenset((names - {"__pycache__"}) | {"delegation_core"})


def repo_module_name(path: Path) -> str:
    """A test module's name, relative to the repository (``tests.unit.core_helpers``), as the tests import it."""
    parts = path.relative_to(REPO).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


@cache
def core_test_files() -> tuple[Path, ...]:
    files = []
    for name in CORE_TESTS:
        root = REPO / name
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.py")))
    return tuple(files)


@cache
def core_test_imports() -> tuple[Import, ...]:
    """Every import of the delegation core's tests and their helpers, read as the walk reads ``src``."""
    known = walked_modules() | frozenset(repo_module_name(path) for path in core_test_files())
    found = []
    for path in core_test_files():
        module = repo_module_name(path)
        source = path.read_text(encoding="utf-8")
        for line, imported in imports_in(source, module, path.name == "__init__.py", known):
            found.append(Import(str(path.relative_to(REPO)), line, module, imported))
    return tuple(found)


# -------------------------------------------------------------------------------------------------- the rules
def within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def group_of(module: str) -> str:
    """The group of the longest matching prefix; a module outside the tables is its own top-level group."""
    prefixes = [prefix for prefix in TARGET_GROUPS if within(module, prefix)]
    if prefixes:
        return TARGET_GROUPS[max(prefixes, key=len)]
    parts = module.split(".")
    return ".".join(parts[:2]) if parts[0] == "examples" else parts[0]


def allowed(source: str, target: str) -> bool:
    if source == PARKED:
        return True
    if target == PARKED:
        return False
    return source == ROOT or target == source or target in MAY_IMPORT[source]


def internal_violation(imp: Import) -> Violation | None:
    if imp.imported.split(".")[0] not in internal_tops():
        return None
    source, target = group_of(imp.importer), group_of(imp.imported)
    if allowed(source, target):
        return None
    return Violation(imp, f"group {source} may not import group {target}")


def package_violation(imp: Import) -> Violation | None:
    package = imp.imported.split(".")[0]
    places = PACKAGE_RULES.get(package)
    if places is None or group_of(imp.importer) in (ROOT, PARKED):
        return None
    if any(within(imp.importer, place) for place in places):
        return None
    return Violation(imp, f"{imp.importer} may not import {package} (only {', '.join(places)} may)")


def internal_violations() -> list[Violation]:
    return [violation for imp in all_imports() if (violation := internal_violation(imp))]


def package_violations() -> list[Violation]:
    return [violation for imp in all_imports() if (violation := package_violation(imp))]


def imports_the_tests(imp: Import) -> Violation | None:
    """A walked module that imports the tests. Unlike the group rules, this one exempts no root and no parked code."""
    if not within(imp.imported, TESTS):
        return None
    return Violation(imp, f"{imp.importer} may not import the tests (no walked module may, whatever its group)")


def core_test_violation(imp: Import) -> Violation | None:
    """An import of the core's tests or helpers that is not the core, those helpers, the stdlib, OpenAI or HTTPX."""
    if any(within(imp.imported, allowed) for allowed in CORE_TESTS_MAY_IMPORT):
        return None
    top = imp.imported.split(".")[0]
    if top in sys.stdlib_module_names or top in CORE_TESTS_PACKAGES:
        return None
    return Violation(
        imp,
        f"the delegation core's tests may import only {', '.join(CORE_TESTS_MAY_IMPORT)}, the standard library, "
        f"{' and '.join(CORE_TESTS_PACKAGES)}",
    )


# -------------------------------------------------------------------------------------------------- the tests
class RuleCase(unittest.TestCase):
    def assert_no_problems(self, title: str, problems: list) -> None:
        """Fail with ``title`` and one line per problem, so a change sees everything it broke at once."""
        if problems:
            self.fail("\n".join([title, *map(str, problems)]))


class ImportRuleTests(RuleCase):
    def test_every_internal_import_follows_the_rules(self):
        self.assert_no_problems("Imports that break a rule:", internal_violations())


class ByNameLoadTests(RuleCase):
    def test_a_module_whose_name_is_computed_is_loaded_only_where_a_loader_is_allowed(self):
        problems = [
            f"{imp.where} loads {imp.imported}: only the modules BY_NAME_LOADERS lists may (add it there with the "
            "reason, or name the module as text so the rules see it)"
            for imp in computed_loads()
            if imp.importer not in BY_NAME_LOADERS
        ]
        self.assert_no_problems("Modules loaded by a computed name:", problems)

    def test_each_allowed_loader_still_loads_by_name(self):
        loaders = {imp.importer for imp in computed_loads()}
        self.assertEqual(sorted(set(BY_NAME_LOADERS) - loaders), [], "delete the entries of loaders that are gone")


class PackageRuleTests(RuleCase):
    def test_every_outside_package_import_follows_the_rules(self):
        self.assert_no_problems("Outside packages imported where no rule allows them:", package_violations())


class ModelRoundTests(RuleCase):
    def test_only_the_backend_adapter_reaches_a_model_providers_stream(self):
        problems = [
            f"{ref.where} reads {ref.receiver}.stream: model rounds belong to the backend adapter ({ROUND_OWNER}). "
            "Send the round to the delegation core through its Responses API, or list a stream that is not a model "
            "provider's in NOT_MODEL_STREAMS with the reason"
            for ref in stream_references()
            if not within(ref.module, ROUND_OWNER) and (ref.module, ref.receiver) not in NOT_MODEL_STREAMS
        ]
        self.assert_no_problems("Model rounds outside the backend adapter:", problems)

    def test_the_backend_adapter_runs_the_model_rounds(self):
        # The walk finds modules that run model rounds, so the rule above cannot pass by finding nothing. Any module
        # of the backend may run model rounds: the backend LLM's runs, and the LLM agent's conductor.
        owners = {ref.module for ref in stream_references() if within(ref.module, ROUND_OWNER)}
        self.assertTrue(owners)
        self.assertIn("delegation_core.backend.llm.run", owners)

    def test_each_listed_stream_is_still_there(self):
        found = {(ref.module, ref.receiver) for ref in stream_references()}
        self.assertEqual(sorted(set(NOT_MODEL_STREAMS) - found), [], "delete the entries of streams that are gone")


class TargetLayoutTests(RuleCase):
    """Layout rules beyond the group table."""

    def test_the_cascade_never_imports_the_adapter(self):
        # The cascade runs without Pipecat's pipeline (the UI bot and the tests host it), so it imports no adapter
        # module; the adapter hosts the cascade.
        cascade, adapter = f"{FBL}.cascade", f"{FBL}.adapter"
        problems = [
            f"{imp.where} imports {imp.imported}: the cascade may not import the frontend adapter's Pipecat modules"
            for imp in all_imports()
            if within(imp.importer, cascade) and within(imp.imported, adapter)
        ]
        self.assert_no_problems("The cascade imports the adapter:", problems)

    def test_only_the_agents_table_imports_an_agent_class_package(self):
        # An agent class with ways of its own is a package of its own under the ACP client (acp/hermes/), and only the
        # table that names the agent classes (acp/agents.py) loads one: the generic harness and the ACP client know an
        # agent through the AcpAgent interface alone, so none of them depends on one agent.
        packages = [f"{ACP_CLIENT}.{path.parent.name}" for path in sorted(ACP_CLIENT_DIR.glob("*/__init__.py"))]
        problems = [
            f"{imp.where} imports {imp.imported}: only {AGENTS_TABLE} may import an agent class's package"
            for imp in all_imports()
            for package in packages
            if within(imp.imported, package) and not within(imp.importer, package) and imp.importer != AGENTS_TABLE
        ]
        self.assert_no_problems("Agent class packages imported outside the agents table:", problems)
        # The walk sees the table load each package, so the rule cannot pass by finding nothing.
        loaded = {imp.imported for imp in all_imports() if imp.importer == AGENTS_TABLE}
        self.assertTrue(packages)
        self.assertLessEqual(set(packages), loaded)

    def test_only_the_boxes_and_the_roots_import_the_gateway_or_the_core(self):
        # Another example, a shared module or a tool reaches neither the gateway nor the delegation core.
        problems = []
        for path in sorted(SRC.rglob("*.py")):
            module = module_name(path)
            if any(within(module, top.removesuffix(".py").replace("/", ".")) for top in WALKED):
                continue
            source = path.read_text(encoding="utf-8")
            for line, imported in imports_in(source, module, path.name == "__init__.py", walked_modules()):
                if within(imported, "live") or within(imported, "delegation_core"):
                    problems.append(f"{path.relative_to(REPO)}:{line} imports {imported}")
        self.assert_no_problems("Modules outside the boxes that import the gateway or the core:", problems)


class TestModuleRuleTests(RuleCase):
    def test_no_walked_module_imports_the_tests_roots_and_parked_code_included(self):
        problems = [violation for imp in all_imports() if (violation := imports_the_tests(imp))]
        self.assert_no_problems("Production modules that import the tests:", problems)

    def test_the_core_tests_import_only_the_core_their_helpers_the_standard_library_openai_and_httpx(self):
        problems = [violation for imp in core_test_imports() if (violation := core_test_violation(imp))]
        self.assert_no_problems("The delegation core's tests import outside the core:", problems)

    def test_the_walk_reads_the_core_tests_and_their_helpers(self):
        importers = {imp.importer for imp in core_test_imports()}
        self.assertIn("tests.unit.core_helpers", importers)
        core_tests = {module for module in importers if within(module, "tests.unit.delegation_core")}
        self.assertGreater(len(core_tests), 1, importers)


class WalkTests(RuleCase):
    def test_the_walk_reads_the_gateway_the_example_and_the_server(self):
        importers = {imp.importer for imp in all_imports()}
        for module in ("live.session.session", f"{FBL}.pipeline", "server"):
            self.assertIn(module, importers)

    def test_every_group_has_a_rule(self):
        self.assertEqual(set(TARGET_GROUPS.values()) - set(MAY_IMPORT) - {ROOT, PARKED}, set())


class WalkerTests(unittest.TestCase):
    def test_imports_count_at_the_top_inside_functions_and_under_type_checking(self):
        source = "\n".join(
            [
                "from typing import TYPE_CHECKING",
                "import live.protocol",
                "if TYPE_CHECKING:",
                "    from live.session import LiveProtocolSession",
                "def start():",
                "    import examples_registry",
                "class Host:",
                "    async def run(self):",
                "        from pipecat.frames.frames import EndFrame",
            ]
        )
        self.assertEqual(
            imports_in(source, "live.events.example", False, frozenset()),
            [
                (1, "typing"),
                (2, "live.protocol"),
                (4, "live.session"),
                (6, "examples_registry"),
                (9, "pipecat.frames.frames"),
            ],
        )

    def test_relative_imports_and_submodules_resolve_to_modules(self):
        source = "\n".join(
            [
                "from . import appends",
                "from .session import LiveProtocolSession",
                "from ..protocol import ProtocolError",
                "from live.gateway import LiveGateway, security",
            ]
        )
        known = frozenset({"live.session.appends", "live.session.session", "live.gateway.security"})
        self.assertEqual(
            imports_in(source, "live.session.lifecycle", False, known),
            [
                (1, "live.session.appends"),
                (2, "live.session.session"),
                (3, "live.protocol"),
                (4, "live.gateway"),
                (4, "live.gateway.security"),
            ],
        )
        self.assertEqual(
            imports_in("from . import appends", "live.session", True, known), [(1, "live.session.appends")]
        )

    def test_modules_named_by_text_count_as_imports(self):
        source = "\n".join(
            [
                "import importlib",
                'DEFAULT = "live.session.lifecycle:start"',
                'OTHER = "not.walked:thing"',
                'LABEL = "session:closed"',
                'importlib.import_module("live.gateway")',
                "importlib.import_module(name)",
                "__import__(spec)",
            ]
        )
        known = frozenset({"live.session.lifecycle", "live.gateway"})
        self.assertEqual(
            imports_in(source, "live.mount", False, known),
            [(1, "importlib"), (2, "live.session.lifecycle"), (5, "live.gateway")],
        )
        self.assertEqual(computed_loads_in(source), [6, 7])


class StreamReferenceWalkerTests(unittest.TestCase):
    def test_reads_of_a_stream_attribute_are_found_with_what_they_are_read_on_called_or_not(self):
        source = "\n".join(
            [
                "async def run(provider, loop, request, payload):",
                "    async for event in provider.stream(request):",
                "        pass",
                "    async with aclosing(loop.provider.stream(payload)) as events:",
                "        pass",
                "    stream = open_stream()",
                "    self.stream.put(event)",
                "    await client.create(stream=True)",
                "    run = getattr(provider, 'stream')",
                "    method = loop.provider.stream",
                "    self.stream = stream",
            ]
        )
        found = [(2, "provider"), (4, "loop.provider"), (7, "self"), (9, "provider"), (10, "loop.provider")]
        self.assertEqual(stream_references_in(source), found)


class RuleTests(unittest.TestCase):
    def test_a_module_belongs_to_the_group_of_its_longest_prefix(self):
        cases = {
            "live.server_delegation.passthrough": "live.server_delegation",
            "delegation_core.backend.port": "delegation_core.backend.port",
            "delegation_core.backend.providers.chat_completions": "delegation_core.backend",
            "delegation_core.backend.harness.acp.codec": "delegation_core.backend.harness",
            "delegation_core.backend.control": "delegation_core.backend",
            f"{FBL}.parked.realtime.engine": PARKED,
            f"{FBL}.factory": ROOT,
            f"{FBL}.cascade.turns": ADAPTER,
            "examples.shared.pipeline_utils": "examples.shared",
            "examples_registry": "examples_registry",
            "utils": "utils",
        }
        self.assertEqual({module: group_of(module) for module in cases}, cases)

    def test_the_rules_follow_the_plan_table(self):
        allowed_pairs = {
            (ADAPTER, "live.protocol"),
            ("live.session", "live.server_delegation"),
            ("live.session", "live.client_delegation"),
            ("delegation_core.im", "delegation_core.backend.port"),
            ("delegation_core.backend.agent", "delegation_core.backend.llm"),
            ("delegation_core.backend", "delegation_core.backend.harness"),
            (ROOT, ADAPTER),
            (PARKED, "live.session"),
        }
        refused_pairs = {
            (ADAPTER, "live.session"),
            (ADAPTER, "delegation_core.im"),
            ("live.gateway", ADAPTER),
            ("live.server_delegation", "live.session"),
            ("live.client_delegation", "live.session"),
            ("live.client_delegation", "live.server_delegation"),
            (ADAPTER, "live.client_delegation"),
            ("delegation_core.im", "live.protocol"),
            ("delegation_core.im", "delegation_core.backend"),
            ("delegation_core.im", "delegation_core.backend.llm"),
            ("delegation_core.backend.harness", "delegation_core.backend.llm"),
            ("delegation_core.backend.llm", "delegation_core.backend.harness"),
            ("delegation_core.backend.port", "delegation_core.backend"),
            ("delegation_core.api", "delegation_core.ledger"),
            (ROOT, PARKED),
            ("live.mount", "examples_registry"),  # the server reads the registry and hands the mount its hooks
        }
        for source, target in allowed_pairs:
            self.assertTrue(allowed(source, target), (source, target))
        for source, target in refused_pairs:
            self.assertFalse(allowed(source, target), (source, target))

    def test_the_delegate_route_reaches_the_backend_only_through_the_hook_it_is_handed(self):
        # live/extras.py serves POST /v1/live/delegate; live/mount.py hands it the composition root's hook.
        for target in ("delegation_core.app", ROOT):
            with self.subTest(target=target):
                self.assertFalse(allowed("live.extras", target))

    def test_a_failure_names_the_file_the_line_the_module_and_both_groups(self):
        path = "src/examples/frontend_backend_live/cascade/turns.py"
        imp = Import(path, 7, f"{FBL}.cascade.turns", "live.session.session")
        self.assertEqual(
            str(internal_violation(imp)),
            f"{path}:7 imports live.session.session: group adapter may not import group live.session",
        )
        imp = Import(path, 3, f"{FBL}.cascade.turns", "pipecat.frames.frames")
        self.assertTrue(str(package_violation(imp)).startswith(f"{path}:3 imports pipecat.frames.frames: {FBL}"))
        self.assertIsNone(package_violation(Import(path, 3, f"{FBL}.adapter.bridge", "pipecat.frames.frames")))

    def test_no_group_may_import_the_tests_not_even_a_root_or_the_parked_code(self):
        importers = ("server", f"{FBL}.factory", f"{FBL}.pipeline", f"{FBL}.parked.realtime.engine", "live.mount")
        for importer in importers:
            for imported in ("tests.unit.core_helpers", "tests.unit.live_helpers", "tests"):
                imp = Import("src/x.py", 1, importer, imported)
                self.assertIsNotNone(imports_the_tests(imp), (importer, imported))
        for imported in ("live.protocol", "testsuite", "unittest.mock"):
            self.assertIsNone(imports_the_tests(Import("src/x.py", 1, f"{FBL}.factory", imported)), imported)
        imp = Import("src/server.py", 5, "server", "tests.unit.core_helpers")
        self.assertEqual(
            str(imports_the_tests(imp)),
            "src/server.py:5 imports tests.unit.core_helpers: server may not import the tests (no walked module may, "
            "whatever its group)",
        )

    def test_the_core_tests_may_import_only_the_core_their_helpers_the_standard_library_openai_and_httpx(self):
        importer = "tests.unit.delegation_core.test_delegation_core_im"
        allowed_modules = (
            "delegation_core",
            "delegation_core.im.session",
            "tests.unit.core_helpers",
            "__future__",
            "asyncio",
            "unittest.mock",
            "openai",
            "openai.types.chat",
            "httpx",
        )
        refused_modules = (
            "live.protocol",
            "live.server_delegation",
            f"{FBL}.factory",
            "examples_registry",
            "utils",
            "server",
            "tests.unit.live_helpers",
            "tests.unit.frontend_backend_live.test_frontend_backend_live",
            "pipecat.frames.frames",
            "fastapi",
        )
        path = "tests/unit/delegation_core/test_delegation_core_im.py"
        for module in allowed_modules:
            self.assertIsNone(core_test_violation(Import(path, 1, importer, module)), module)
        for module in refused_modules:
            self.assertIsNotNone(core_test_violation(Import(path, 1, importer, module)), module)
        self.assertEqual(
            str(core_test_violation(Import(path, 9, importer, "live.protocol"))),
            f"{path}:9 imports live.protocol: the delegation core's tests may import only delegation_core, "
            "tests.unit.core_helpers, tests.unit.contracts, tests.unit.delegation_core, the standard library, openai "
            "and httpx",
        )
