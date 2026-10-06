# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""OpenShell/Fabric backend adapter."""

from voiceclaw.adapters.openshell_fabric.committed_turn import OpenShellFabricAdapter
from voiceclaw.adapters.openshell_fabric.factory import build_openshell_fabric_backend

__all__ = ["OpenShellFabricAdapter", "build_openshell_fabric_backend"]
