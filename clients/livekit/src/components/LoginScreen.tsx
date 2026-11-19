// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, type FormEvent } from "react";
import { login, queryClient } from "../api";

const TOO_MANY_ATTEMPTS = 429;

/** Sign in when the UI server sets a password. The server checks the typed values and never sends them back. */
export function LoginScreen({ askUsername }: Readonly<{ askUsername: boolean }>) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const status = await login(username, password);
      if (status >= 200 && status < 300) {
        setPassword("");
        await queryClient.invalidateQueries();
        return;
      }
      setError(status === TOO_MANY_ATTEMPTS ? "Too many attempts. Wait a minute." : "Wrong username or password.");
    } catch {
      setError("The UI server could not be reached.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="h-screen d-flex items-center justify-center">
      <form className="login-card d-flex flex-col gap-3" onSubmit={handleSubmit}>
        <h1 className="text-lg font-semibold">
          <span className="brand-accent">Nemotron</span> Voice Agent
        </h1>
        <p className="settings-hint">Sign in to use the voice agent.</p>
        {askUsername && (
          <input
            className="svc-input"
            type="text"
            placeholder="Username"
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoFocus
          />
        )}
        <input
          className="svc-input"
          type="password"
          placeholder="Password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoFocus={!askUsername}
          required
        />
        <button className="btn-primary" type="submit" disabled={busy || !password}>Sign in</button>
        {error && <p className="error-text">{error}</p>}
      </form>
    </div>
  );
}
