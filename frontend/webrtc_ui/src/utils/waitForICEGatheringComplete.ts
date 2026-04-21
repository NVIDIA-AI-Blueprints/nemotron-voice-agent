// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

export default async function waitForICEGatheringComplete(
  pc: RTCPeerConnection,
  timeoutMs = 2000
): Promise<void> {
  if (pc.iceGatheringState === "complete") return;
  console.log(
    "Waiting for ICE gathering to complete. Current state:",
    pc.iceGatheringState
  );
  return new Promise((resolve) => {
    const checkState = () => {
      console.log("icegatheringstatechange:", pc.iceGatheringState);
      if (pc.iceGatheringState === "complete") {
        cleanup();
        resolve();
      }
    };
    const onTimeout = () => {
      console.warn(`ICE gathering timed out after ${timeoutMs} ms.`);
      cleanup();
      resolve();
    };
    const cleanup = () => {
      pc.removeEventListener("icegatheringstatechange", checkState);
      clearTimeout(timeoutId);
    };
    pc.addEventListener("icegatheringstatechange", checkState);
    const timeoutId = setTimeout(onTimeout, timeoutMs);
    // Checking the state again to avoid any eventual race condition
    checkState();
  });
}
