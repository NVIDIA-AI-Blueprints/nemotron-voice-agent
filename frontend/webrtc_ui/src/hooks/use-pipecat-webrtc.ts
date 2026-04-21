// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useCallback, useRef, useState } from "react";
import waitForICEGatheringComplete from "../utils/waitForICEGatheringComplete";
import { useMutation } from "@tanstack/react-query";

interface Params {
  url: string;
  rtcConfig: ConstructorParameters<typeof RTCPeerConnection>[0];
  onError: (error: Error) => void;
}

interface ReturnInit {
  status: "init";
  start: () => void;
}

interface ReturnConnecting {
  status: "connecting";
  stop: () => void;
}

interface ReturnConnected {
  status: "connected";
  stop: () => void;
  micStream: MediaStream;
  stream: MediaStream;
  dataChannel: RTCDataChannel;
}

interface ReturnError {
  status: "error";
  start: () => void;
  error: Error;
}

type Return = ReturnInit | ReturnConnecting | ReturnConnected | ReturnError;

export default function usePipecatWebRTC(params: Params): Return {
  const [status, setStatus] = useState<Return["status"]>("init");
  const [error, setError] = useState<Error | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const outputStreamRef = useRef<MediaStream | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const dataChannelRef = useRef<RTCDataChannel | null>(null);

  const offerMutation = useMutation({
    mutationKey: ["offer"],
    mutationFn: async (offer: RTCSessionDescriptionInit) => {
      const res = await fetch(params.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },
  });

  const stop = useCallback(() => {
    for (const stream of [micStreamRef, outputStreamRef]) {
      if (stream.current) {
        for (const track of stream.current.getTracks()) {
          track.stop();
        }
        stream.current = null;
      }
    }
    if (pcRef.current) {
      for (const transceiver of pcRef.current.getTransceivers()) {
        transceiver.stop();
      }
      pcRef.current.close();
      pcRef.current = null;
    }
    setError(null);
    setStatus("init");
  }, []);

  const connect = useCallback(async () => {
    try {
      const pc = new RTCPeerConnection(params.rtcConfig);
      pcRef.current = pc;

      const dataChannel = pc.createDataChannel("messaging", { ordered: true });
      dataChannelRef.current = dataChannel;

      dataChannel.onopen = () => {
        console.log("Data channel opened");
        dataChannel.send(JSON.stringify({
          id: "client-ready-msg",
          label: "rtvi-ai",
          type: "client-ready",
          data: {
            "version": "1.1.0",
          }
        }));
      };

      dataChannel.onclose = () => {
        dataChannelRef.current = null;
      };

      pc.oniceconnectionstatechange = () => {
        console.log("oniceconnectionstatechange", pc?.iceConnectionState);
      };
      pc.onconnectionstatechange = () => {
        console.log("onconnectionstatechange", pc?.connectionState);
        const connectionState = pc?.connectionState;
        if (connectionState === "connected") {
          setStatus("connected");
        } else if (connectionState === "disconnected") {
          stop();
        } else if (connectionState === "failed") {
          stop();
          const err = new Error("WebRTC connection failed");
          setError(err);
          params.onError(err);
          setStatus("error");
        }
      };
      pc.onicecandidate = (event) => {
        if (event.candidate) {
          console.log("New ICE candidate:", event.candidate);
        } else {
          console.log("All ICE candidates have been sent.");
        }
      };
      pc.ontrack = (e) => (outputStreamRef.current = e.streams[0]);
      micStreamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
      const micTrack = micStreamRef.current.getAudioTracks()[0];
      // Mute microphone by default until Start
      micTrack.enabled = false;

      // SmallWebRTCTransport expects to receive both transceivers
      pc.addTransceiver(micTrack, { direction: "sendrecv" });
      pc.addTransceiver("video", { direction: "sendrecv" });
      await pc.setLocalDescription(await pc.createOffer());
      await waitForICEGatheringComplete(pc);
      const offer = pc.localDescription;


      const answer = await offerMutation.mutateAsync(offer as RTCSessionDescriptionInit);
      await pc.setRemoteDescription(answer as RTCSessionDescriptionInit);
    } catch (e) {
      stop();
      const err =
        e instanceof Error
          ? e
          : new Error("Unknown error during connection setup");
      setError(err);
      params.onError(err);
      setStatus("error");
    }
  }, [stop, params.url, params.rtcConfig, params.onError]);

  const start = useCallback(() => {
    setStatus("connecting");
    connect();
  }, [connect]);

  switch (status) {
    case "init":
      return { status, start };
    case "connecting":
      return { status, stop };
    case "connected":
      return {
        status,
        stop,
        micStream: micStreamRef.current!,
        stream: outputStreamRef.current!,
        dataChannel: dataChannelRef.current!,
      };
    case "error":
      return { status, start, error: error! };
  }
}
