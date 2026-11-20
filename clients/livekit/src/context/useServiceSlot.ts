// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useMemo, useState } from "react";
import type { CustomFieldSpec, CustomService, ServiceEntry, ServiceSettingValues, Slot } from "../api";
import { CUSTOM_ID_PREFIX, effectiveServiceId, parseStoredServices, settingsKey, storedServices } from "../lib/services";
import type { SlotSelection } from "../lib/session";
import { readLSJson, writeLSJson } from "../utils";

const EMPTY_ENTRIES: ServiceEntry[] = [];
const NO_CUSTOM: CustomService[] = [];

export interface SlotState {
  entries: ServiceEntry[];
  /** The fields of a custom endpoint. Undefined when the server does not accept custom endpoints. */
  customSpec: CustomFieldSpec | undefined;
  custom: CustomService[];
  selectedId: string;
  selection: SlotSelection | undefined;
  select: (id: string) => void;
  /** Add a custom endpoint. Returns it, or undefined when custom endpoints cannot change now. */
  addCustom: (name: string, fields: Record<string, string>) => CustomService | undefined;
  updateCustom: (id: string, name: string, fields: Record<string, string>) => void;
  removeCustom: (id: string) => void;
}

function storageKey(slot: Slot): string {
  return `nvidia-voice-agent-livekit-${slot}-custom`;
}

/**
 * The services of one slot: the offered entries, the custom endpoints stored in this browser, and the choice.
 *
 * Custom endpoints are offered only when the server accepts them (`customSpec`). Nothing changes while `locked`.
 */
export function useServiceSlot(
  slot: Slot,
  offered: ServiceEntry[] | undefined,
  settings: Record<string, ServiceSettingValues>,
  customSpec: CustomFieldSpec | undefined,
  locked: boolean,
): SlotState {
  const entries = offered ?? EMPTY_ENTRIES;
  const [stored, setStored] = useState<CustomService[]>(() => parseStoredServices(readLSJson(storageKey(slot)), slot));
  const [chosenId, setChosenId] = useState("");
  const custom = customSpec ? stored : NO_CUSTOM;
  const selectedId = effectiveServiceId(chosenId, entries, custom);
  const editable = Boolean(customSpec) && !locked;

  const persist = useCallback((next: CustomService[]) => {
    setStored(next);
    writeLSJson(storageKey(slot), storedServices(next));
  }, [slot]);

  const select = useCallback((id: string) => {
    if (!locked) setChosenId(id);
  }, [locked]);

  const addCustom = useCallback((name: string, fields: Record<string, string>) => {
    if (!editable) return undefined;
    const service: CustomService = { id: `${CUSTOM_ID_PREFIX}${slot}-${crypto.randomUUID()}`, slot, name, fields };
    persist([...stored, service]);
    return service;
  }, [editable, persist, slot, stored]);

  const updateCustom = useCallback((id: string, name: string, fields: Record<string, string>) => {
    if (editable) persist(stored.map((service) => (service.id === id ? { ...service, name, fields } : service)));
  }, [editable, persist, stored]);

  const removeCustom = useCallback((id: string) => {
    if (editable) persist(stored.filter((service) => service.id !== id));
  }, [editable, persist, stored]);

  const selection = useMemo<SlotSelection | undefined>(() => {
    const service = custom.find((item) => item.id === selectedId);
    if (service && customSpec) return { kind: "custom", service, spec: customSpec };
    const entry = entries.find((item) => item.id === selectedId);
    return entry && { kind: "builtin", entry, settings: settings[settingsKey(slot, entry.id)] ?? {} };
  }, [custom, customSpec, entries, selectedId, settings, slot]);

  return useMemo(
    () => ({ entries, customSpec, custom, selectedId, selection, select, addCustom, updateCustom, removeCustom }),
    [entries, customSpec, custom, selectedId, selection, select, addCustom, updateCustom, removeCustom],
  );
}
