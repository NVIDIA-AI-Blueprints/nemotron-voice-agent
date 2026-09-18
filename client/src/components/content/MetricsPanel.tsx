// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useCallback, useMemo, useRef } from "react";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";
import { isRecord } from "../../utils";
import { TTFBChart } from "./TTFBChart";

interface TokenMetrics {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

interface TTFBMetric {
  id: string;
  processor: string;
  value: number;
  timestamp: string;
}

interface LatencyPoint {
  id: string;
  value: number;
  timestamp: string;
  first: boolean;
}

interface MetricBoardRow {
  id: string;
  groupId?: string;
  groupLabel?: string;
  timestamp: string;
  category: string;
  metric: string;
  processor: string;
  model: string;
  value: number;
  unit: string;
  rawValue: number;
}

interface MetricGroupValue {
  key: string;
  label: string;
  value: number;
  unit: string;
}

interface MetricGroup {
  id: string;
  ordinal: number;
  label: string;
  timestamp: string;
  category: string;
  source: string;
  metrics: MetricGroupValue[];
}

type MetricsView = "charts" | "board";
type BoardFilter = string;
type DetailColumnKey = "timestamp" | "group" | "category" | "metric" | "processor" | "model" | "value" | "unit";

interface DetailColumn {
  key: DetailColumnKey;
  label: string;
  render: (row: MetricBoardRow) => string;
}

const ALL_BOARD_FILTER = "all";

const TURN_COLORS = ["#23331b", "#1f3038", "#332b1b", "#2f2438", "#1f352f", "#3a2626", "#292f3a", "#34341f"];

const SUMMARY_BASE_COLUMNS = [
  { id: "summary:ordinal", label: "#" },
  { id: "summary:timestamp", label: "Time" },
  { id: "summary:group", label: "Group" },
  { id: "summary:category", label: "Category" },
  { id: "summary:source", label: "Source" },
];

const DETAIL_COLUMNS: DetailColumn[] = [
  { key: "timestamp", label: "Time", render: (row) => row.timestamp },
  { key: "group", label: "Group", render: (row) => row.groupLabel ?? row.groupId ?? "-" },
  { key: "category", label: "Category", render: (row) => row.category },
  { key: "metric", label: "Metric", render: (row) => row.metric },
  { key: "processor", label: "Processor", render: (row) => row.processor || "-" },
  { key: "model", label: "Model", render: (row) => row.model || "-" },
  { key: "value", label: "Value", render: (row) => row.value.toFixed(3) },
  { key: "unit", label: "Unit", render: (row) => row.unit || "-" },
];

let metricPointId = 0;

function nextMetricPointId(prefix: string) {
  metricPointId += 1;
  return `${prefix}-${metricPointId}`;
}

function groupColor(groupId?: string) {
  if (!groupId) return "";
  const index = [...groupId].reduce((sum, char) => sum + char.charCodeAt(0), 0) % TURN_COLORS.length;
  return TURN_COLORS[index];
}

function keepLastPerProcessor(entries: TTFBMetric[], limit: number) {
  const counts = new Map<string, number>();
  const kept: TTFBMetric[] = [];

  for (const entry of [...entries].reverse()) {
    const count = counts.get(entry.processor) ?? 0;
    if (count < limit) {
      kept.push(entry);
      counts.set(entry.processor, count + 1);
    }
  }

  return kept.reverse();
}

function flattenNumericFields(value: unknown, prefix = ""): Array<[string, number]> {
  if (typeof value === "number" && Number.isFinite(value)) {
    return [[prefix || "value", value]];
  }
  if (!isRecord(value)) return [];

  return Object.entries(value).flatMap(([key, nested]) => {
    if (key === "processor" || key === "model" || key === "type") return [];
    const field = prefix ? `${prefix}.${key}` : key;
    return flattenNumericFields(nested, field);
  });
}

function normalizeMetricValue(field: string, value: number) {
  if (field.endsWith("_ms") || field.includes("time_ms")) {
    return { value, unit: "ms" };
  }
  return { value, unit: "" };
}

function metricRowsFromMetricsPayload(metrics: unknown, timestamp: string): MetricBoardRow[] {
  if (!isRecord(metrics)) return [];

  return Object.entries(metrics).flatMap(([category, entries]) => {
    const items = Array.isArray(entries) ? entries : [entries];
    return items.flatMap((entry) => {
      if (!isRecord(entry)) return [];
      const processor = typeof entry.processor === "string" ? entry.processor : "";
      const model = typeof entry.model === "string" ? entry.model : "";
      return flattenNumericFields(entry).map(([field, rawValue]) => {
        const normalized = normalizeMetricValue(field, rawValue);
        return {
          id: nextMetricPointId("board"),
          timestamp,
          category,
          metric: field,
          processor,
          model,
          value: normalized.value,
          unit: normalized.unit,
          rawValue,
        };
      });
    });
  });
}

function metricRowsFromServerMessage(message: unknown, timestamp: string): MetricBoardRow[] {
  if (!isRecord(message) || typeof message.type !== "string") return [];
  if (message.type === "metric-group") {
    const groupId = typeof message.group_id === "string" ? message.group_id : undefined;
    const groupLabel = typeof message.group_label === "string" ? message.group_label : groupId;
    const category = typeof message.category === "string" ? message.category : "grouped";
    const source = typeof message.source === "string" ? message.source : "";
    const metrics = Array.isArray(message.metrics) ? message.metrics : [];
    return metrics.flatMap((metric) => {
      if (!isRecord(metric) || typeof metric.value !== "number" || !Number.isFinite(metric.value)) return [];
      const key = typeof metric.key === "string" ? metric.key : "value";
      const label = typeof metric.label === "string" ? metric.label : key;
      const unit = typeof metric.unit === "string" ? metric.unit : "";
      return {
        id: nextMetricPointId("board"),
        groupId,
        groupLabel,
        timestamp,
        category,
        metric: label,
        processor: source,
        model: "",
        value: metric.value,
        unit,
        rawValue: metric.value,
      };
    });
  }
  const groupId = typeof message.group_id === "string" ? message.group_id : undefined;
  const groupLabel = typeof message.group_label === "string" ? message.group_label : groupId;
  return flattenNumericFields(message).map(([field, rawValue]) => {
    const normalized = normalizeMetricValue(field, rawValue);
    return {
      id: nextMetricPointId("board"),
      groupId,
      groupLabel,
      timestamp,
      category: "server_message",
      metric: `${message.type}.${field}`,
      processor: "",
      model: "",
      value: normalized.unit ? normalized.value : rawValue,
      unit: normalized.unit,
      rawValue,
    };
  });
}

function metricGroupFromServerMessage(message: unknown, timestamp: string, ordinal: number): MetricGroup | null {
  if (!isRecord(message) || message.type !== "metric-group") return null;
  const groupId = typeof message.group_id === "string" ? message.group_id : "";
  if (!groupId || !Array.isArray(message.metrics)) return null;
  const metrics = message.metrics.flatMap((metric): MetricGroupValue[] => {
    if (!isRecord(metric) || typeof metric.value !== "number" || !Number.isFinite(metric.value)) return [];
    const key = typeof metric.key === "string" ? metric.key : "value";
    return [{
      key,
      label: typeof metric.label === "string" ? metric.label : key,
      value: metric.value,
      unit: typeof metric.unit === "string" ? metric.unit : "",
    }];
  });
  if (metrics.length === 0) return null;
  return {
    id: groupId,
    ordinal,
    label: typeof message.group_label === "string" ? message.group_label : groupId,
    timestamp,
    category: typeof message.category === "string" ? message.category : "grouped",
    source: typeof message.source === "string" ? message.source : "",
    metrics,
  };
}

function escapeHtml(value: string | number) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function detailColumnId(key: DetailColumnKey) {
  return `detail:${key}`;
}

function summaryMetricColumnId(key: string) {
  return `summary:metric:${key}`;
}

function downloadExcel(headers: string[], bodyRows: Array<Array<string | number>>) {
  if (headers.length === 0 || bodyRows.length === 0) return;
  const html = `
    <html>
      <head><meta charset="utf-8" /></head>
      <body>
        <table>
          <thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>
          <tbody>
            ${bodyRows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}
          </tbody>
        </table>
      </body>
    </html>
  `;
  const blob = new Blob([html], { type: "application/vnd.ms-excel;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `metrics-board-${new Date().toISOString().replace(/[:.]/g, "-")}.xls`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function downloadBoardAsExcel(rows: MetricBoardRow[], columns: DetailColumn[]) {
  downloadExcel(
    columns.map((column) => column.label),
    rows.map((row) => columns.map((column) => column.render(row))),
  );
}

function downloadGroupedSummaryAsExcel(
  groups: MetricGroup[],
  columns: Array<{ key: string; label: string }>,
  isColumnVisible: (columnId: string) => boolean,
) {
  const baseColumns = [
    { id: "summary:ordinal", label: "#", render: (group: MetricGroup) => String(group.ordinal) },
    { id: "summary:timestamp", label: "Timestamp", render: (group: MetricGroup) => group.timestamp },
    { id: "summary:group", label: "Group", render: (group: MetricGroup) => group.label },
    { id: "summary:category", label: "Category", render: (group: MetricGroup) => group.category },
    { id: "summary:source", label: "Source", render: (group: MetricGroup) => group.source },
  ].filter((column) => isColumnVisible(column.id));
  const visibleMetricColumns = columns.filter((column) => isColumnVisible(summaryMetricColumnId(column.key)));
  const headers = [...baseColumns.map((column) => column.label), ...visibleMetricColumns.map((column) => column.label)];
  const bodyRows = groups.map((group) => [
    ...baseColumns.map((column) => column.render(group)),
    ...visibleMetricColumns.map((column) => formatMetricValue(group.metrics.find((metric) => metric.key === column.key))),
  ]);
  downloadExcel(headers, bodyRows);
}

function metricMatchesFilter(row: MetricBoardRow, filter: BoardFilter) {
  return filter === ALL_BOARD_FILTER || row.category === filter;
}

function formatCategoryLabel(category: string) {
  if (category === ALL_BOARD_FILTER) return "All";
  return category
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatMetricValue(metric?: MetricGroupValue) {
  if (!metric) return "-";
  const value = metric.unit === "ms" ? metric.value.toFixed(1) : metric.value.toFixed(3);
  return metric.unit ? `${value} ${metric.unit}` : value;
}

function groupRowStyle(groupId?: string) {
  const color = groupColor(groupId);
  return color ? { backgroundColor: color } : undefined;
}

export function MetricsPanel() {
  const [activeView, setActiveView] = useState<MetricsView>("charts");
  const [boardFilter, setBoardFilter] = useState<BoardFilter>(ALL_BOARD_FILTER);
  const [hiddenBoardColumns, setHiddenBoardColumns] = useState<string[]>([]);
  const [showDetailRows, setShowDetailRows] = useState(false);
  const [tokens, setTokens] = useState<TokenMetrics>({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 });
  const [ttfbHistory, setTtfbHistory] = useState<TTFBMetric[]>([]);
  const [latencyHistory, setLatencyHistory] = useState<LatencyPoint[]>([]);
  const [metricRows, setMetricRows] = useState<MetricBoardRow[]>([]);
  const [metricGroups, setMetricGroups] = useState<MetricGroup[]>([]);
  const nextMetricGroupOrdinal = useRef(1);
  const metricGroupOrdinals = useRef(new Map<string, number>());
  const ttfbByProcessor = useMemo(() => {
    const grouped = new Map<string, TTFBMetric[]>();
    for (const entry of ttfbHistory) {
      grouped.set(entry.processor, [...(grouped.get(entry.processor) ?? []), entry]);
    }
    return [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [ttfbHistory]);
  const filteredMetricRows = useMemo(
    () => metricRows.filter((row) => metricMatchesFilter(row, boardFilter)),
    [boardFilter, metricRows],
  );
  const filteredMetricGroups = useMemo(
    () => metricGroups.filter((group) => boardFilter === ALL_BOARD_FILTER || group.category === boardFilter),
    [boardFilter, metricGroups],
  );
  const boardFilterOptions = useMemo(() => {
    const categories = new Set<string>();
    for (const group of metricGroups) categories.add(group.category);
    for (const row of metricRows) categories.add(row.category);
    return [
      { id: ALL_BOARD_FILTER, label: "All" },
      ...[...categories].sort((a, b) => a.localeCompare(b)).map((category) => ({
        id: category,
        label: formatCategoryLabel(category),
      })),
    ];
  }, [metricGroups, metricRows]);
  const groupMetricColumns = useMemo(() => {
    const byKey = new Map<string, { key: string; label: string }>();
    for (const group of filteredMetricGroups) {
      for (const metric of group.metrics) {
        if (!byKey.has(metric.key)) byKey.set(metric.key, { key: metric.key, label: metric.label });
      }
    }
    return [...byKey.values()];
  }, [filteredMetricGroups]);
  const visibleDetailColumns = useMemo(
    () => DETAIL_COLUMNS.filter((column) => !hiddenBoardColumns.includes(detailColumnId(column.key))),
    [hiddenBoardColumns],
  );
  const visibleGroupMetricColumns = useMemo(
    () => groupMetricColumns.filter((column) => !hiddenBoardColumns.includes(summaryMetricColumnId(column.key))),
    [groupMetricColumns, hiddenBoardColumns],
  );
  const isColumnVisible = useCallback(
    (columnId: string) => !hiddenBoardColumns.includes(columnId),
    [hiddenBoardColumns],
  );
  const toggleBoardColumn = useCallback((columnId: string) => {
    setHiddenBoardColumns((prev) => (
      prev.includes(columnId)
        ? prev.filter((id) => id !== columnId)
        : [...prev, columnId]
    ));
  }, []);
  const hasVisibleSummaryColumns = useMemo(
    () => SUMMARY_BASE_COLUMNS.some((column) => isColumnVisible(column.id)) || visibleGroupMetricColumns.length > 0,
    [isColumnVisible, visibleGroupMetricColumns.length],
  );
  const canDownloadBoard = showDetailRows
    ? filteredMetricRows.length > 0 && visibleDetailColumns.length > 0
    : filteredMetricGroups.length > 0 && hasVisibleSummaryColumns;
  const handleDownloadBoard = useCallback(() => {
    if (showDetailRows) {
      downloadBoardAsExcel(filteredMetricRows, visibleDetailColumns);
      return;
    }
    downloadGroupedSummaryAsExcel(filteredMetricGroups, groupMetricColumns, isColumnVisible);
  }, [filteredMetricGroups, filteredMetricRows, groupMetricColumns, isColumnVisible, showDetailRows, visibleDetailColumns]);

  useRTVIClientEvent(
    RTVIEvent.Metrics,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    useCallback((metrics: any) => {
      const timestamp = new Date().toLocaleTimeString();
      if (metrics.tokens && metrics.tokens.length > 0) {
        const t = metrics.tokens[0];
        setTokens({
          prompt_tokens: t.prompt_tokens || 0,
          completion_tokens: t.completion_tokens || 0,
          total_tokens: t.total_tokens || 0,
        });
      }

      if (metrics.ttfb && metrics.ttfb.length > 0) {
        const newEntries = metrics.ttfb.map((ttfb: { processor: string; value?: number }) => ({
          id: nextMetricPointId("ttfb"),
          processor: ttfb.processor,
          value: (ttfb.value ?? 0) * 1000,
          timestamp,
        }));
        setTtfbHistory((prev) => keepLastPerProcessor([...prev, ...newEntries], 10));
      }

      const boardRows = metricRowsFromMetricsPayload(metrics, timestamp);
      if (boardRows.length > 0) {
        setMetricRows((prev) => [...prev, ...boardRows].slice(-500));
      }
    }, [])
  );

  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    useCallback((message: any) => {
      const timestamp = new Date().toLocaleTimeString();
      if (message?.type === "user-bot-latency") {
        setLatencyHistory((prev) => [
          ...prev,
          {
            id: nextMetricPointId("latency"),
            value: (message.latency ?? 0) * 1000,
            timestamp,
            first: message.first ?? false,
          },
        ].slice(-10));
      }
      const boardRows = metricRowsFromServerMessage(message, timestamp);
      if (boardRows.length > 0) {
        setMetricRows((prev) => [...prev, ...boardRows].slice(-500));
      }
      if (isRecord(message) && message.type === "metric-group") {
        const groupId = typeof message.group_id === "string" ? message.group_id : "";
        if (groupId && !metricGroupOrdinals.current.has(groupId)) {
          metricGroupOrdinals.current.set(groupId, nextMetricGroupOrdinal.current);
          nextMetricGroupOrdinal.current += 1;
        }
        const ordinal = metricGroupOrdinals.current.get(groupId) ?? nextMetricGroupOrdinal.current;
        const metricGroup = metricGroupFromServerMessage(message, timestamp, ordinal);
        if (metricGroup) {
          setMetricGroups((prev) => [...prev, metricGroup].slice(-100));
        }
      }
    }, [])
  );

  return (
    <div className="metrics-panel p-4">
      <div className="metrics-subtabs">
        <button className={`tab-btn ${activeView === "charts" ? "active" : ""}`} onClick={() => setActiveView("charts")}>
          Charts
        </button>
        <button className={`tab-btn ${activeView === "board" ? "active" : ""}`} onClick={() => setActiveView("board")}>
          Board
        </button>
      </div>

      {activeView === "charts" ? (
        <>
          {/* Token Usage */}
          <div className="metrics-section">
            <h3 className="metrics-title">Token Usage</h3>
            <div className="metrics-grid">
              <div className="metric-card">
                <span className="metric-label">Prompt Tokens</span>
                <span className="metric-value">{tokens.prompt_tokens}</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Completion Tokens</span>
                <span className="metric-value">{tokens.completion_tokens}</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Total Tokens</span>
                <span className="metric-value">{tokens.total_tokens}</span>
              </div>
            </div>
          </div>

          {/* User→Bot Latency */}
          <div className="metrics-section">
            <h3 className="metrics-title">User→Bot Latency</h3>
            {latencyHistory.length >= 2 ? (
              <TTFBChart data={latencyHistory} title="Response Latency" label="Latency" />
            ) : (
              <p className="text-secondary">No latency data yet. Start a conversation.</p>
            )}
          </div>

          {/* TTFB Metrics */}
          <div className="metrics-section">
            <h3 className="metrics-title">TTFB Metrics</h3>
            {ttfbByProcessor.length > 0 ? (
              ttfbByProcessor.map(([processor, points]) => (
                <TTFBChart key={processor} data={points} title={processor} />
              ))
            ) : (
              <p className="text-secondary">No TTFB data yet. Start a conversation.</p>
            )}
          </div>
        </>
      ) : (
        <div className="metrics-section">
          <div className="metrics-board-header">
            <div>
              <h3 className="metrics-title">Metrics Board</h3>
              <p className="text-secondary">
                Showing {filteredMetricGroups.length} grouped summaries.
                {showDetailRows ? ` Details: ${filteredMetricRows.length} of ${metricRows.length} samples.` : ""}
              </p>
            </div>
            <div className="metrics-board-actions">
              <button className="metrics-secondary-btn" onClick={() => setShowDetailRows((value) => !value)}>
                {showDetailRows ? "Hide Details" : "Show Details"}
              </button>
              <button
                className="metrics-download-btn"
                onClick={handleDownloadBoard}
                disabled={!canDownloadBoard}
                title={!canDownloadBoard ? "No metrics to download" : "Download Excel"}
                aria-label="Download Excel"
              >
                ↓
              </button>
            </div>
          </div>

          <div className="metrics-board-filters">
            {boardFilterOptions.map((filter) => (
              <button
                key={filter.id}
                className={`metrics-filter-btn ${boardFilter === filter.id ? "active" : ""}`}
                onClick={() => setBoardFilter(filter.id)}
              >
                {filter.label}
              </button>
            ))}
          </div>

          <details className="metrics-column-controls">
            <summary>Columns</summary>
            <div className="metrics-column-groups">
              <div className="metrics-column-group">
                <span className="metrics-column-group-title">Summary</span>
                {[...SUMMARY_BASE_COLUMNS, ...groupMetricColumns.map((column) => ({
                  id: summaryMetricColumnId(column.key),
                  label: column.label,
                }))].map((column) => (
                  <label key={column.id} className="metrics-column-toggle">
                    <input
                      type="checkbox"
                      checked={isColumnVisible(column.id)}
                      onChange={() => toggleBoardColumn(column.id)}
                    />
                    {column.label}
                  </label>
                ))}
              </div>
              <div className="metrics-column-group">
                <span className="metrics-column-group-title">Details</span>
                {DETAIL_COLUMNS.map((column) => (
                  <label key={column.key} className="metrics-column-toggle">
                    <input
                      type="checkbox"
                      checked={isColumnVisible(detailColumnId(column.key))}
                      onChange={() => toggleBoardColumn(detailColumnId(column.key))}
                    />
                    {column.label}
                  </label>
                ))}
              </div>
            </div>
          </details>

          {filteredMetricGroups.length > 0 && groupMetricColumns.length > 0 && (
            <div className="metrics-turn-summary">
              <p className="chart-title">Grouped Metrics Summary</p>
              <div className="metrics-board-scroll metrics-board-scroll--compact">
                <table className="metrics-board-table">
                  <thead>
                    <tr>
                      {isColumnVisible("summary:ordinal") && <th>#</th>}
                      {isColumnVisible("summary:timestamp") && <th>Time</th>}
                      {isColumnVisible("summary:group") && <th>Group</th>}
                      {isColumnVisible("summary:category") && <th>Category</th>}
                      {isColumnVisible("summary:source") && <th>Source</th>}
                      {groupMetricColumns.map((column) => (
                        isColumnVisible(summaryMetricColumnId(column.key)) && <th key={column.key}>{column.label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {filteredMetricGroups.map((group) => (
                      <tr key={`${group.ordinal}-${group.id}`} style={groupRowStyle(group.id)}>
                        {isColumnVisible("summary:ordinal") && <td>{group.ordinal}</td>}
                        {isColumnVisible("summary:timestamp") && <td>{group.timestamp}</td>}
                        {isColumnVisible("summary:group") && <td>{group.label}</td>}
                        {isColumnVisible("summary:category") && <td>{group.category}</td>}
                        {isColumnVisible("summary:source") && <td>{group.source || "-"}</td>}
                        {groupMetricColumns.map((column) => (
                          isColumnVisible(summaryMetricColumnId(column.key)) && (
                            <td key={column.key}>
                              {formatMetricValue(group.metrics.find((metric) => metric.key === column.key))}
                            </td>
                          )
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {showDetailRows && filteredMetricRows.length > 0 ? (
            <div className="metrics-board-scroll">
              <table className="metrics-board-table">
                <thead>
                  <tr>
                    {visibleDetailColumns.map((column) => (
                      <th key={column.key}>{column.label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredMetricRows.map((row) => (
                    <tr key={row.id} style={groupRowStyle(row.groupId)}>
                      {visibleDetailColumns.map((column) => (
                        <td key={column.key}>{column.render(row)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : showDetailRows ? (
            <p className="text-secondary">No metric rows yet. Start a conversation.</p>
          ) : null}
        </div>
      )}
    </div>
  );
}
