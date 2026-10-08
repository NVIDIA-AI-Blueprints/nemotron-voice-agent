# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name -> implementation registries for the frontend and backend roles."""

from __future__ import annotations


class Registry[T]:
    """Implementations of one role, by name."""

    def __init__(self, kind: str):
        """Create an empty registry for the role called ``kind``."""
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str):
        """Return a class decorator that registers the class under ``name``."""

        def decorate(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            cls.name = name
            self._items[name] = cls
            return cls

        return decorate

    def get(self, name: str) -> type[T]:
        """Return the class registered as ``name``."""
        try:
            return self._items[name]
        except KeyError:
            raise ValueError(f"Unknown {self.kind} {name!r}; registered: {', '.join(self.names()) or 'none'}") from None

    def names(self) -> list[str]:
        """Return the registered names, sorted."""
        return sorted(self._items)


FRONTENDS: Registry = Registry("frontend")
BACKENDS: Registry = Registry("backend")
