// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useCallback, useEffect, useState } from "react";
import { readLSString, writeLSString } from "../utils";

const STORAGE_PREFIX = "nvidia-voice-agent-collapsed:";

export function useCollapsible(storageKey: string, defaultCollapsed = false) {
  const key = `${STORAGE_PREFIX}${storageKey}`;
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    const stored = readLSString(key);
    return stored ? stored === "1" : defaultCollapsed;
  });

  const toggle = useCallback(() => {
    setCollapsed((prev) => !prev);
  }, []);

  useEffect(() => {
    writeLSString(key, collapsed ? "1" : "0");
  }, [collapsed, key]);

  return { collapsed, toggle };
}
