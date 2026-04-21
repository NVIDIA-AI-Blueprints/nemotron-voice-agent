// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useEffect, useRef } from "react";

interface Props {
  streamOrTrack: MediaStream | MediaStreamTrack | null;
}
export function AudioStream(props: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);
  useEffect(() => {
    if (audioRef.current && props.streamOrTrack) {
      audioRef.current.srcObject =
        props.streamOrTrack instanceof MediaStream
          ? props.streamOrTrack
          : new MediaStream([props.streamOrTrack]);
    }
  }, [props.streamOrTrack]);

  if (!props.streamOrTrack) {
    return;
  }

  return <audio ref={audioRef} autoPlay />;
}
