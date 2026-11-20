// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// `npm run dev` serves the client with hot reload and sends the data endpoints to a running UI server.
const UI_SERVER = process.env.LIVEKIT_UI_SERVER ?? 'http://127.0.0.1:3000'
const API_PATHS = [
  '/auth/status', '/client-config', '/login', '/token', '/services', '/voices', '/prompts', '/tools', '/deployment',
  '/phone',
]

export default defineConfig({
  plugins: [react()],
  base: '/',
  server: {
    proxy: Object.fromEntries(API_PATHS.map((path) => [path, UI_SERVER])),
  },
})
