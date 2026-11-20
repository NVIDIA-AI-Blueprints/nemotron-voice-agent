// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient, useAuthStatus } from "./api";
import { AppProvider } from "./context/AppContext";
import { CenterPanel } from "./components/content/CenterPanel";
import { Header } from "./components/Header";
import { LoginScreen } from "./components/LoginScreen";
import { Sidebar } from "./components/Sidebar";
import { StatusPanel } from "./components/status-panel/StatusPanel";
import { SessionProvider } from "./session/SessionProvider";

function VoiceAgent() {
  return (
    <AppProvider>
      <SessionProvider>
        <div className="h-screen d-flex flex-col overflow-hidden">
          <Header />
          <div className="flex-1 d-flex overflow-hidden">
            <StatusPanel />
            <CenterPanel />
            <Sidebar />
          </div>
        </div>
      </SessionProvider>
    </AppProvider>
  );
}

/** The page, after the sign-in when the UI server asks for one. Without the auth endpoint the page opens directly. */
function AuthGate() {
  const { data: auth, isLoading } = useAuthStatus();
  if (isLoading) return <div className="h-screen d-flex items-center justify-center">Loading...</div>;
  if (auth?.required && !auth.authenticated) return <LoginScreen askUsername={auth.username} />;
  return <VoiceAgent />;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthGate />
    </QueryClientProvider>
  );
}

export default App;
