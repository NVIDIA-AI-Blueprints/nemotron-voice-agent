# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The example's prompt profiles, one folder each in ``prompts/``, and the lines the voice says word for word.

A profile is a folder that holds ``profile.yaml`` (its settings only: ``default``, ``internal`` and ``tools``) and its
role files, each a ``description`` and a ``content``:
``talker.yaml`` (the talker's prompt), ``thinker.yaml`` (the backend's) and ``channel.yaml`` (the context that follows
both). A listed profile, one the UI offers, needs all three; an internal one may leave the channel out.
``shared/lines.yaml`` holds the lines every profile's voice says.

A UI bot session selects a listed profile, so its talker, channel, backend prompt and tools switch together;
``frontend.cascade.prompt`` names the profile of a Live session's default talker, and the core's ``backend.prompt``
the profile of a Live request's default backend prompt. A new profile is one new folder.

Every read goes to the files, so an edit takes effect at the next session or request.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from utils import load_yaml_file

PROMPTS_DIR = Path(__file__).resolve().parent
LINES_PATH = PROMPTS_DIR / "shared" / "lines.yaml"
# The lines' mapping in LINES_PATH.
LINES_KEY = "lines"
# A profile's settings file; a folder of PROMPTS_DIR that holds one is a profile.
PROFILE_FILE = "profile.yaml"
SETTINGS_KEYS = ("default", "internal", "tools")
ROLES = ("talker", "thinker", "channel")


@dataclass(frozen=True)
class PromptProfile:
    """One profile, as its files read now: its settings and its role files' entries."""

    name: str
    default: bool  # the profile a UI bot session gets when it names none
    internal: bool  # not listed: the UI leaves it out and a UI bot session cannot select it
    roles: Mapping[str, Mapping[str, Any]]  # role -> its file's entry ({description, content}); {} when absent
    tools: Mapping[str, Mapping[str, Any]]  # the backend's server tool sources by name, for the UI bot's sessions

    def content(self, role: str) -> str:
        """The content of ``role`` as its file holds it.

        Raises:
            KeyError: The profile has no ``role`` file, or its content is empty.
        """
        content = self.roles[role].get("content") if role in self.roles else None
        if not str(content or "").strip():
            raise KeyError(f"Prompt {role!r} was not found in the Frontend/Backend Live prompt profile {self.name!r}")
        return content

    def prompt(self, role: str) -> str:
        """The text of ``role``, stripped: what a role starts from when nothing follows it."""
        return str(self.content(role)).strip()

    def instructions(self, role: str) -> str:
        """``role``'s text followed by the channel context, when the profile has one."""
        if "channel" not in self.roles:
            return self.prompt(role)
        return f"{self.prompt(role)}\n\n{self.prompt('channel')}"


def profile_names() -> list[str]:
    """Every profile's name: the folders of ``PROMPTS_DIR`` that hold a ``profile.yaml``, sorted."""
    return sorted(path.parent.name for path in PROMPTS_DIR.glob(f"*/{PROFILE_FILE}"))


def profile_settings(name: str) -> dict[str, Any]:
    """The settings of profile ``name`` (its ``profile.yaml``), checked.

    Raises:
        KeyError: No profile has that name; the message lists the known ones.
        ValueError: The file is malformed, or a setting is unknown or of the wrong kind.
    """
    if name not in profile_names():
        raise KeyError(f"Unknown prompt profile {name!r}; known: {', '.join(profile_names())}")
    path = PROMPTS_DIR / name / PROFILE_FILE
    try:
        settings = load_yaml_file(path, required=True)
    except RuntimeError as exc:
        raise ValueError(f"Prompt profile {name!r}: {exc}") from None
    unknown = set(settings) - set(SETTINGS_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown settings {', '.join(sorted(map(str, unknown)))}")
    for flag in ("default", "internal"):
        if not isinstance(settings.get(flag, False), bool):
            raise ValueError(f"{path}: {flag} must be true or false")
    tools = settings.get("tools", {})
    if not isinstance(tools, Mapping) or not all(isinstance(block, Mapping) for block in tools.values()):
        raise ValueError(f"{path}: tools must map each tool source's name to its block")
    return settings


def get_profile(name: str) -> PromptProfile:
    """Profile ``name``, read from its folder now.

    Raises:
        KeyError: No profile has that name; the message lists the known ones.
        ValueError: Its settings are malformed, or it is listed and has no channel.
    """
    settings = profile_settings(name)
    folder = PROMPTS_DIR / name
    roles = {role: load_yaml_file(folder / f"{role}.yaml") for role in ROLES if (folder / f"{role}.yaml").is_file()}
    internal = settings.get("internal", False)
    if not internal and "channel" not in roles:
        raise ValueError(f"Prompt profile {name!r} is listed, so it needs a channel.yaml")
    return PromptProfile(
        name=name,
        default=settings.get("default", False),
        internal=internal,
        roles=MappingProxyType(roles),
        tools=MappingProxyType(dict(settings.get("tools", {}))),
    )


def default_profile_name() -> str:
    """The name of the one profile with ``default: true``, from the profiles' settings alone.

    Raises:
        ValueError: Not exactly one profile is the default, the default is internal, or a profile's settings are
            malformed.
    """
    defaults = [name for name in profile_names() if profile_settings(name).get("default", False)]
    if len(defaults) != 1:
        raise ValueError(f"Exactly one prompt profile must set default: true; found {defaults or 'none'}")
    (name,) = defaults
    if profile_settings(name).get("internal", False):
        raise ValueError(f"The default prompt profile {name!r} cannot be internal")
    return name


def default_profile() -> PromptProfile:
    """The default profile: the one a UI bot session gets when it names none."""
    return get_profile(default_profile_name())


def listed_profile_names() -> list[str]:
    """The profiles the UI lists and a UI bot session can select: every one that is not internal, sorted."""
    return [name for name in profile_names() if not profile_settings(name).get("internal", False)]


def ui_catalog() -> dict[str, dict[str, Any]]:
    """The UI's prompt list for this example (the registry's ``prompt_catalog`` hook).

    One entry per listed profile, in the shape of an example's ``prompts.yaml`` entry: its talker's ``description`` and
    ``content`` as the talker file holds them, and ``default``. Internal profiles are left out, and their role files
    are never read here.
    """
    default = default_profile_name()
    catalog = {}
    for name in listed_profile_names():
        talker = get_profile(name).roles.get("talker", {})
        catalog[name] = {
            "description": talker.get("description", ""),
            "content": talker.get("content", ""),
            "default": name == default,
        }
    return catalog


def catalog_line(name: str) -> str:
    """Return the line ``name`` that the voice says word for word; an empty line means the voice says none.

    Raises:
        KeyError: The shared lines have no line of that name.
    """
    lines = load_yaml_file(LINES_PATH).get(LINES_KEY)
    if not isinstance(lines, dict) or name not in lines:
        raise KeyError(f"Line {name!r} was not found in the Frontend/Backend Live {LINES_PATH.name}")
    return str(lines[name] or "")
