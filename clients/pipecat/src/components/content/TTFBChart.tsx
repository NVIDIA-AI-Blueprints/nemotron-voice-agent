// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface DataPoint {
  id: string;
  timestamp: string;
  value: number;
}

interface TTFBChartProps {
  data: DataPoint[];
  title: string;
  label?: string;
}

export function TTFBChart({ data, title, label = "TTFB" }: Readonly<TTFBChartProps>) {
  if (data.length < 2) return null;

  return (
    <div className="chart-container">
      <p className="chart-title">{title}</p>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="id"
            stroke="#666666"
            fontSize={10}
            tickLine={false}
            tickFormatter={(_, index) => data[index]?.timestamp ?? ""}
          />
          <YAxis
            stroke="#666666"
            fontSize={10}
            tickLine={false}
            tickFormatter={(v) => `${v}ms`}
          />
          <Tooltip
            contentStyle={{
              background: "#1a1a1a",
              border: "1px solid #2a2a2a",
              borderRadius: "4px",
              color: "#e0e0e0",
              fontSize: "12px",
            }}
            labelFormatter={(_, payload) => payload?.[0]?.payload?.timestamp ?? ""}
            formatter={(value) => [`${Number(value).toFixed(0)}ms`, label]}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#76b900"
            strokeWidth={2}
            dot={{ fill: "#76b900", r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
