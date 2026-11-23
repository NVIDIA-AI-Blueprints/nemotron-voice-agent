// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const LIVEKIT_PORT = 7880;

/**
 * Return the LiveKit signaling address for this page.
 *
 * The default is the page's own host on port 7880, with `wss://` when the page is served over HTTPS. A configured
 * public address (for a page behind an HTTPS proxy) is used only when it names the page's own host, so a page opened
 * as localhost through an SSH tunnel keeps its local address.
 */
export function resolveServerUrl(page: { protocol: string; hostname: string }, configured: string): string {
  const fallback = `${page.protocol === "https:" ? "wss" : "ws"}://${page.hostname}:${LIVEKIT_PORT}`;
  if (!configured) return fallback;
  try {
    return new URL(configured).hostname === page.hostname ? configured : fallback;
  } catch {
    return fallback;
  }
}
