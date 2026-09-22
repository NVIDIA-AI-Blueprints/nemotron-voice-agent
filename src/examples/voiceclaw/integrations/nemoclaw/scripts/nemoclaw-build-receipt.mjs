#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const RECEIPT_SCHEMA = "voiceclaw.nemoclaw-build.v1";
const REVISION = /^[0-9a-f]{40}$/u;

function receiptPath(root) {
  return `${path.resolve(root)}.voiceclaw-build-receipt.json`;
}

function hashTree(root) {
  const dist = path.join(root, "dist");
  const digest = crypto.createHash("sha256");
  const visit = (directory, relativeDirectory = "") => {
    const entries = fs.readdirSync(directory, { withFileTypes: true }).sort((left, right) =>
      left.name.localeCompare(right.name, "en"),
    );
    for (const entry of entries) {
      const relative = path.posix.join(relativeDirectory, entry.name);
      const absolute = path.join(directory, entry.name);
      if (entry.isSymbolicLink()) throw new Error("NemoClaw dist must not contain symbolic links.");
      if (entry.isDirectory()) {
        visit(absolute, relative);
        continue;
      }
      if (!entry.isFile()) throw new Error("NemoClaw dist contains an unsupported filesystem entry.");
      const metadata = fs.statSync(absolute);
      const name = Buffer.from(relative, "utf8");
      const content = fs.readFileSync(absolute);
      const header = Buffer.alloc(17);
      header.writeBigUInt64BE(BigInt(name.length), 0);
      header.writeBigUInt64BE(BigInt(content.length), 8);
      header.writeUInt8(metadata.mode & 0o111 ? 1 : 0, 16);
      digest.update(header);
      digest.update(name);
      digest.update(content);
    }
  };
  visit(dist);
  return digest.digest("hex");
}

function validatedIdentity(root, revision, describe) {
  if (!REVISION.test(revision)) throw new Error("The NemoClaw receipt revision is invalid.");
  const identity = JSON.parse(fs.readFileSync(path.join(root, "dist/build-identity.json"), "utf8"));
  const expectedVersion = describe.replace(/^v/u, "");
  if (identity?.sourceRevision !== revision || identity?.nemoclawVersion !== expectedVersion) {
    throw new Error("The built NemoClaw identity does not match VoiceClaw's exact pin.");
  }
  return expectedVersion;
}

export function writeReceipt(root, revision, describe) {
  const absoluteRoot = path.resolve(root);
  const version = validatedIdentity(absoluteRoot, revision, describe);
  const receipt = {
    schema: RECEIPT_SCHEMA,
    revision,
    version,
    distSha256: hashTree(absoluteRoot),
  };
  const target = receiptPath(absoluteRoot);
  if (fs.existsSync(target)) {
    const existing = fs.lstatSync(target);
    if (!existing.isFile() || existing.isSymbolicLink() || existing.uid !== process.getuid()) {
      throw new Error("Refusing to replace an unsafe NemoClaw build receipt path.");
    }
  }
  const temporary = `${target}.${process.pid}.${crypto.randomBytes(8).toString("hex")}.tmp`;
  let descriptor;
  try {
    descriptor = fs.openSync(temporary, "wx", 0o600);
    fs.writeFileSync(descriptor, `${JSON.stringify(receipt, null, 2)}\n`, { encoding: "utf8" });
    fs.fsyncSync(descriptor);
    fs.closeSync(descriptor);
    descriptor = undefined;
    fs.renameSync(temporary, target);
  } finally {
    if (descriptor !== undefined) fs.closeSync(descriptor);
    try {
      fs.unlinkSync(temporary);
    } catch (error) {
      if (!(error instanceof Error) || error.code !== "ENOENT") throw error;
    }
  }
  return target;
}

export function verifyReceipt(root, revision, describe) {
  const absoluteRoot = path.resolve(root);
  const target = receiptPath(absoluteRoot);
  const metadata = fs.lstatSync(target);
  if (!metadata.isFile() || metadata.isSymbolicLink() || metadata.uid !== process.getuid() || metadata.mode & 0o077) {
    throw new Error("The NemoClaw build receipt is not an owner-only regular file.");
  }
  const receipt = JSON.parse(fs.readFileSync(target, "utf8"));
  const version = validatedIdentity(absoluteRoot, revision, describe);
  if (
    receipt?.schema !== RECEIPT_SCHEMA ||
    receipt?.revision !== revision ||
    receipt?.version !== version ||
    receipt?.distSha256 !== hashTree(absoluteRoot)
  ) {
    throw new Error("The prepared NemoClaw build does not match its reviewed receipt.");
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [mode, root, revision, describe] = process.argv.slice(2);
  if (mode !== "write" || !root || !revision || !describe) {
    console.error("Usage: nemoclaw-build-receipt.mjs write CHECKOUT REVISION DESCRIBE");
    process.exitCode = 2;
  } else {
    try {
      writeReceipt(root, revision, describe);
    } catch (error) {
      console.error(error instanceof Error ? error.message : "Could not write the NemoClaw build receipt.");
      process.exitCode = 1;
    }
  }
}
