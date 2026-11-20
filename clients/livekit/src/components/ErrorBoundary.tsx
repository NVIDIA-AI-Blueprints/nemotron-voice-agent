// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Component, type ReactNode } from "react";

/** Show what failed, with a reload button, instead of a blank page when rendering throws. */
export class ErrorBoundary extends Component<Readonly<{ children: ReactNode }>, { error: string }> {
  state = { error: "" };

  static getDerivedStateFromError(error: unknown) {
    return { error: (error instanceof Error ? error.message : String(error)) || "Unknown error" };
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="h-screen d-flex items-center justify-center">
        <div className="login-card d-flex flex-col gap-3" role="alert">
          <h1 className="text-lg font-semibold">The page stopped working</h1>
          <p className="error-text">{this.state.error}</p>
          <button type="button" className="btn-primary" onClick={() => globalThis.location.reload()}>Reload</button>
        </div>
      </div>
    );
  }
}
