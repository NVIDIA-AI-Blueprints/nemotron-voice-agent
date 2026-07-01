// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useMemo } from "react";
import { PipecatClient } from "@pipecat-ai/client-js";
import { PipecatClientProvider, PipecatClientAudio } from "@pipecat-ai/client-react";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import { WebSocketTransport, ProtobufFrameSerializer } from "@pipecat-ai/websocket-transport";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient, useIceServers } from "./api";
import { AppProvider } from "./context/AppContext";
import { useApp } from "./context/useApp";
import { Header } from "./components/Header";
import { StatusPanel } from "./components/status-panel";
import { Sidebar } from "./components/Sidebar";
import { CenterPanel } from "./components/content";

const EMPTY_ICE_SERVERS: RTCIceServer[] = [];

function AppInner() {
  const { selectedTransport } = useApp();
  const { data: iceConfig, isFetched: iceServersLoaded } = useIceServers();
  const iceServers = iceConfig?.iceServers ?? EMPTY_ICE_SERVERS;

  const client = useMemo(() => {
    if (selectedTransport === "websocket") {
      return new PipecatClient({
        transport: new WebSocketTransport({
          serializer: new ProtobufFrameSerializer(),
          recorderSampleRate: 16000,
          playerSampleRate: 16000,
        }),
        enableMic: true,
        enableCam: false,
        enableScreenShare: false,
      });
    }
    if (!iceServersLoaded) return null;
    return new PipecatClient({
      transport: new SmallWebRTCTransport({ iceServers }),
      enableMic: true,
    });
  }, [iceServers, iceServersLoaded, selectedTransport]);

  if (!client) {
    return <div className="h-screen d-flex items-center justify-center">Loading connection...</div>;
  }

  return (
    <PipecatClientProvider client={client}>
      <div className="h-screen d-flex flex-col overflow-hidden">
        <Header />
        <div className="flex-1 d-flex overflow-hidden">
          <StatusPanel />
          <CenterPanel />
          <Sidebar />
        </div>
        <PipecatClientAudio />
      </div>
    </PipecatClientProvider>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppProvider>
        <AppInner />
      </AppProvider>
    </QueryClientProvider>
  );
}

export default App;
