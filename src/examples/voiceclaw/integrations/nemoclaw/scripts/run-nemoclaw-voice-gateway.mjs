#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

import { verifyReceipt } from "./nemoclaw-build-receipt.mjs";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const integrationDirectory = path.resolve(scriptDirectory, "..");
const exampleDirectory = path.resolve(integrationDirectory, "../..");
const runtimeRoot = process.env.VOICECLAW_RUNTIME_ROOT || path.join(exampleDirectory, ".runtime");
const nemoclawRoot = process.env.NEMOCLAW_SOURCE_DIR || path.join(runtimeRoot, "nemoclaw");
const launchModule = path.join(nemoclawRoot, "dist/lib/actions/voice-gateway/launch.js");
const versionFile = path.join(integrationDirectory, "compat/version.env");
const compatibilityPatch = path.join(
  integrationDirectory,
  "compat/0001-response-only-gateway-compatibility.patch",
);

function required(name) {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Set ${name} before starting the NemoClaw voice gateway.`);
  return value;
}

function positivePort(value) {
  const port = Number(value);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    throw new Error("NEMOCLAW_VOICE_GATEWAY_PORT must be an integer from 1024 through 65535.");
  }
  return port;
}

function pinnedRevision() {
  const version = fs.readFileSync(versionFile, "utf8");
  const match = /^NEMOCLAW_REVISION=([0-9a-f]{40})$/mu.exec(version);
  if (!match) throw new Error("VoiceClaw's pinned NemoClaw revision is invalid.");
  return match[1];
}

function pinnedDescribe() {
  const version = fs.readFileSync(versionFile, "utf8");
  const match = /^NEMOCLAW_DESCRIBE=(v[^\s]+)$/mu.exec(version);
  if (!match) throw new Error("VoiceClaw's pinned NemoClaw description is invalid.");
  return match[1];
}

function gitOutput(args) {
  const result = spawnSync("git", ["-C", nemoclawRoot, ...args], {
    encoding: "utf8",
    maxBuffer: 2 * 1024 * 1024,
  });
  if (result.error || result.status !== 0) {
    const operation = typeof args[0] === "string" ? args[0] : "operation";
    const reason = result.error?.code || `exit ${result.status ?? "unknown"}`;
    const detail = typeof result.stderr === "string" ? result.stderr.trim().slice(0, 300) : "";
    throw new Error(
      `The prepared NemoClaw checkout could not run git ${operation} (${reason})${detail ? `: ${detail}` : "."}`,
    );
  }
  return result.stdout;
}

function verifyPreparedCheckout() {
  const expectedRevision = pinnedRevision();
  const revision = gitOutput(["rev-parse", "HEAD"]).trim();
  if (revision !== expectedRevision) {
    throw new Error("The prepared NemoClaw checkout does not match VoiceClaw's pinned revision.");
  }
  const actualPatch = gitOutput([
    "diff",
    "HEAD",
    "--binary",
    "--no-ext-diff",
    "--unified=0",
    "--src-prefix=a/",
    "--dst-prefix=b/",
  ]);
  if (actualPatch !== fs.readFileSync(compatibilityPatch, "utf8")) {
    throw new Error("The prepared NemoClaw checkout does not contain exactly the reviewed compatibility patch.");
  }
  if (gitOutput(["ls-files", "--others", "--exclude-standard"]).trim()) {
    throw new Error("The prepared NemoClaw checkout contains unexpected untracked source files.");
  }
  verifyReceipt(nemoclawRoot, expectedRevision, pinnedDescribe());
}

function waitForExit(child, timeoutMilliseconds) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return Promise.resolve(true);
  return new Promise((resolve) => {
    let settled = false;
    const finish = (observed) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      child.off("exit", exited);
      child.off("error", failed);
      resolve(observed);
    };
    const exited = () => finish(true);
    const failed = () => finish(false);
    const timer = setTimeout(() => finish(false), timeoutMilliseconds);
    timer.unref();
    child.once("exit", exited);
    child.once("error", failed);
    if (child.exitCode !== null || child.signalCode !== null) finish(true);
  });
}

async function stop(child, signal = "SIGTERM") {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  if (!child.kill(signal)) throw new Error("The NemoClaw voice gateway could not be signalled for shutdown.");
  if (await waitForExit(child, 5000)) return;
  if (!child.kill("SIGKILL")) throw new Error("The NemoClaw voice gateway could not be force-terminated.");
  if (!(await waitForExit(child, 5000))) {
    throw new Error("The NemoClaw voice gateway did not confirm exit after force termination.");
  }
}

function signalExitCode(signal) {
  return { SIGINT: 130, SIGKILL: 137, SIGTERM: 143 }[signal] || 1;
}

async function main() {
  verifyPreparedCheckout();
  if (!fs.existsSync(launchModule)) {
    throw new Error(`Prepared NemoClaw launcher not found at ${launchModule}. Run prepare-nemoclaw.sh first.`);
  }
  const require = createRequire(import.meta.url);
  const { runVoiceGatewayLaunch } = require(launchModule);
  if (typeof runVoiceGatewayLaunch !== "function") {
    throw new Error("The pinned NemoClaw checkout does not export its trusted voice-gateway launcher.");
  }

  const options = {
    deploymentCredentialPath: required("NEMOCLAW_DEPLOYMENT_CREDENTIAL_FILE"),
    openClawCredentialPath: required("NEMOCLAW_AGENT_CREDENTIAL_FILE"),
    gatewayUrl: required("NEMOCLAW_AGENT_GATEWAY_URL"),
    runtimeIdentity: process.env.NEMOCLAW_RUNTIME_IDENTITY?.trim() || "voiceclaw-local",
    runtimeProfile: process.env.NEMOCLAW_RUNTIME_PROFILE?.trim() || "voiceclaw-committed-turn-v1",
    sandbox: required("NEMOCLAW_SANDBOX"),
    agent: process.env.NEMOCLAW_AGENT?.trim() || "main",
    listenPort: positivePort(process.env.NEMOCLAW_VOICE_GATEWAY_PORT || "18800"),
  };

  let child;
  try {
    child = await runVoiceGatewayLaunch(options);
    child.stdout?.pipe(process.stdout);
    child.stderr?.pipe(process.stderr);
  } catch (error) {
    if (error?.child) await stop(error.child);
    throw error;
  }

  let rejectTermination;
  const terminationFailure = new Promise((_resolve, reject) => {
    rejectTermination = reject;
  });
  let stopping = false;
  const forward = (signal) => {
    if (stopping || child.exitCode !== null || child.signalCode !== null) return;
    stopping = true;
    void stop(child, signal).catch(rejectTermination);
  };
  process.once("SIGINT", () => forward("SIGINT"));
  process.once("SIGTERM", () => forward("SIGTERM"));

  const childExit = new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      process.exitCode = signal ? signalExitCode(signal) : (code ?? 1);
      resolve();
    });
  });
  await Promise.race([childExit, terminationFailure]);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : "NemoClaw voice gateway failed.");
  process.exitCode = 1;
});
