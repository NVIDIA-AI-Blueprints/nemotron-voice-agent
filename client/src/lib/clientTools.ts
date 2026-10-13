// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

// Single source of truth for the tools this client can execute.
//
// Each entry both *declares* the tool (name/description/parameters, sent to
// the bot as requestData.tools at connect time so the server can hand it to
// the LLM) and *implements* it (the handler, wired up in App.tsx via
// client.registerFunctionCallHandler). The bot never runs this logic itself
// — see src/examples/multilingual/tools.py and tool_handlers.py for how the
// server forwards a call for one of these names over RTVI and waits for the
// browser's answer.

import type { FunctionCallCallback } from "@pipecat-ai/client-js";
import { evaluateMathExpression, resolveExpressionArgument } from "./mathWorkerClient";

export type ClientToolDeclaration = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  /**
   * True when running the tool is itself something the user sees or hears (a
   * robot moving, a sound playing). Recorded sessions then count the moment the
   * call is sent as the turn's first response. Metadata only: the server never
   * forwards it to the LLM. Absent means false.
   */
  perceivable?: boolean;
};

export type ClientTool = ClientToolDeclaration & {
  handler: FunctionCallCallback;
};

export const CLIENT_TOOLS: ClientTool[] = [
  {
    name: "get_client_local_time",
    description:
      "Get the user's current local date, time, and timezone directly from their own device. " +
      "Use this whenever the user asks what time, date, or day it is for them, since the server " +
      "may run in a different timezone.",
    parameters: {
      type: "object",
      properties: {},
      required: [],
      additionalProperties: false,
    },
    handler: async () => {
      const now = new Date();
      return {
        iso8601: now.toISOString(),
        local_date: now.toLocaleDateString(),
        local_time: now.toLocaleTimeString(),
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      };
    },
  },
  {
    name: "evaluate_math_expression",
    description:
      "Evaluate a math expression (arithmetic, powers, roots, trigonometry, logarithms) using a " +
      "sandboxed calculator in the user's browser. Use this for any calculation instead of " +
      "computing or guessing the answer yourself.",
    parameters: {
      type: "object",
      properties: {
        expression: {
          type: "string",
          description:
            "A plain arithmetic expression using +, -, *, /, %, ^ (power), parentheses, numeric " +
            "literals, and Math function names such as sqrt, pow, sin, cos, log, abs " +
            "(for example 'sqrt(16) + 2^3').",
        },
      },
      required: ["expression"],
      additionalProperties: false,
    },
    handler: async (fn) => {
      const expression = resolveExpressionArgument(fn.arguments);
      try {
        const result = await evaluateMathExpression(expression);
        return { expression, result };
      } catch (err) {
        return { expression, error: err instanceof Error ? err.message : String(err) };
      }
    },
  },
];

/** Declarations only (no handlers) — what gets sent to the bot at connect time. */
export function clientToolDeclarations(): ClientToolDeclaration[] {
  return CLIENT_TOOLS.map(({ name, description, parameters, perceivable }) => ({
    name,
    description,
    parameters,
    ...(perceivable ? { perceivable: true } : {}),
  }));
}
