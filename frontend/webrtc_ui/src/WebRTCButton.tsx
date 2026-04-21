// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { twMerge } from "tailwind-merge";
import type usePipecatWebRTC from "./hooks/use-pipecat-webrtc";

type Props = ReturnType<typeof usePipecatWebRTC>;

export default function WebRTCButton(props: Props) {
  const className = "bg-nvidia px-4 py-2 rounded-lg text-white";
  switch (props.status) {
    case "init":
      return (
        <button className={className} onClick={props.start}>
          Start
        </button>
      );
    case "connecting":
      return (
        <button className={twMerge(className, "opacity-40")} disabled>
          Connecting...
        </button>
      );
    case "connected":
      return (
        <button className={className} onClick={props.stop}>
          Stop
        </button>
      );
    case "error":
      return (
        <button
          className={className}
          onClick={props.start}
          title={props.error.message}
        >
          Error
        </button>
      );
  }
}
