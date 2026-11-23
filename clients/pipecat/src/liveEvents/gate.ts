// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// When the EVENT LOG tab applies. The shared app shows it only for an example whose registry entry lists the capability
// (`examples_registry.yaml`, `capabilities`), the way the webcam panel follows `webcam`.

/** The EVENT LOG tab. */
export const LOG_CAPABILITY = "event_log";

/** Whether the Event log applies: the example lists `event_log`. */
export function logEnabled(example: { capabilities?: string[] } | null | undefined): boolean {
  return example?.capabilities?.includes(LOG_CAPABILITY) ?? false;
}
