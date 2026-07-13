// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useCallback, useState } from "react";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";
import { useApp } from "../context/useApp";
import { useSubagents, type Subagent } from "../api";
import { isRecord, stringField } from "../utils";
import { CollapsibleCard } from "./CollapsibleCard";

export function SubagentsPanel() {
  const { selectedExample } = useApp();
  const { data: subagents = [], isLoading } = useSubagents(selectedExample?.key ?? "");

  if (isLoading || subagents.length === 0) return null;

  return <SubagentList subagents={subagents} />;
}

function SubagentList({ subagents }: Readonly<{ subagents: Subagent[] }>) {
  const [activeTasks, setActiveTasks] = useState<Map<string, Set<string>>>(new Map());

  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    useCallback((message: unknown) => {
      if (!isRecord(message) || stringField(message, "type") !== "agent-task-update") return;
      const agent = stringField(message, "agent");
      const taskId = stringField(message, "task_id");
      if (!agent || !taskId) return;
      const done = ["done", "error", "complete"].includes(stringField(message, "status"));
      setActiveTasks((prev) => {
        const next = new Map(prev);
        const tasks = new Set(next.get(agent) ?? []);
        if (done) tasks.delete(taskId);
        else tasks.add(taskId);
        if (tasks.size > 0) next.set(agent, tasks);
        else next.delete(agent);
        return next;
      });
    }, []),
  );

  return (
    <CollapsibleCard label="SUBAGENTS" storageKey="subagents">
      <ul className="subagents-list">
        {subagents.map((subagent) => {
          const working = activeTasks.has(subagent.key);
          return (
            <li
              key={subagent.key}
              className={`subagent-row is-on${working ? " is-working" : ""}`}
              title={subagent.capability}
            >
              <span className="subagent-info">
                <span className="subagent-name">{subagent.label}</span>
                <span className="subagent-meta">
                  <span className={`subagent-badge subagent-badge-reasoning-${subagent.reasoning}`}>
                    reasoning: {subagent.reasoning}
                  </span>
                  {!subagent.delegatable && <span className="subagent-badge subagent-badge-ambient">ambient</span>}
                </span>
              </span>
            </li>
          );
        })}
      </ul>
    </CollapsibleCard>
  );
}
