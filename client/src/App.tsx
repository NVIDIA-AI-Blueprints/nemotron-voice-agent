// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { Component, useCallback, useMemo, useState, type ErrorInfo, type ReactNode } from "react";
import { PipecatClient } from "@pipecat-ai/client-js";
import { PipecatClientProvider, PipecatClientAudio } from "@pipecat-ai/client-react";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import { WebSocketTransport, ProtobufFrameSerializer } from "@pipecat-ai/websocket-transport";
import { QueryClientProvider } from "@tanstack/react-query";
import { createStore } from "jotai";
import { queryClient, useIceServers } from "./api";
import { AppProvider } from "./context/AppContext";
import { useApp } from "./context/useApp";
import { Header } from "./components/Header";
import { StatusPanel } from "./components/status-panel";
import { Sidebar } from "./components/Sidebar";
import { CenterPanel } from "./components/content";

const EMPTY_ICE_SERVERS: RTCIceServer[] = [];

class AppErrorBoundary extends Component<
  Readonly<{ children: ReactNode; resetKey: number }>,
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Nemotron Voice Agent UI error:", error, errorInfo);
  }

  componentDidUpdate(prevProps: Readonly<{ resetKey: number }>) {
    if (prevProps.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="h-screen d-flex items-center justify-center p-4">
        <div className="panel-section max-w-md">
          <p className="panel-label">UI ERROR</p>
          <p className="text-sm text-secondary mb-3">
            The session UI hit an unexpected browser-side error.
          </p>
          <button className="btn-primary" type="button" onClick={() => globalThis.location.reload()}>
            Reload
          </button>
        </div>
      </div>
    );
  }
}

function AppInner() {
  const { selectedTransport } = useApp();
  const { data: iceConfig, isFetched: iceServersLoaded } = useIceServers();
  const iceServers = iceConfig?.iceServers ?? EMPTY_ICE_SERVERS;
  const [clientGeneration, setClientGeneration] = useState(0);
  const resetClient = useCallback(() => {
    globalThis.setTimeout(() => {
      setClientGeneration((generation) => generation + 1);
    }, 0);
  }, []);
  const jotaiStore = useMemo(() => {
    void clientGeneration;
    return createStore();
  }, [clientGeneration]);

  const client = useMemo(() => {
    void clientGeneration;
    const callbacks = { onDisconnected: resetClient };
    if (selectedTransport === "websocket") {
      return new PipecatClient({
        transport: new WebSocketTransport({
          serializer: new ProtobufFrameSerializer(),
          recorderSampleRate: 16000,
          playerSampleRate: 16000,
        }),
        callbacks,
        enableMic: true,
        enableCam: false,
        enableScreenShare: false,
      });
    }
    if (!iceServersLoaded) return null;
    return new PipecatClient({
      transport: new SmallWebRTCTransport({ iceServers }),
      callbacks,
      enableMic: true,
    });
  }, [clientGeneration, iceServers, iceServersLoaded, resetClient, selectedTransport]);

  if (!client) {
    return <div className="h-screen d-flex items-center justify-center">Loading connection...</div>;
  }

  return (
    <PipecatClientProvider key={clientGeneration} client={client} jotaiStore={jotaiStore}>
      <AppErrorBoundary resetKey={clientGeneration}>
        <div className="h-screen d-flex flex-col overflow-hidden">
          <Header />
          <div className="flex-1 d-flex overflow-hidden">
            <StatusPanel />
            <CenterPanel />
            <Sidebar />
          </div>
          <PipecatClientAudio />
        </div>
      </AppErrorBoundary>
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
