// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import React, { useState, useEffect } from "react";

interface PromptInputProps {
  defaultValue?: string;
  onChange?: (prompt: string) => void;
  disabled?: boolean;
}

export const PromptInput: React.FC<PromptInputProps> = ({
  defaultValue = " ",
  onChange,
  disabled = false
}) => {
  const [prompt, setPrompt] = useState(defaultValue);

  // Update prompt state when defaultValue prop changes
  useEffect(() => {
    setPrompt(defaultValue);
  }, [defaultValue]);

  // Notify parent component when prompt changes
  const handlePromptChange = (newPrompt: string) => {
    setPrompt(newPrompt);
    onChange?.(newPrompt);
  };

  return (
    <div className="h-full flex flex-col bg-gray-50 p-3">
      <textarea
        value={prompt}
        onChange={(e) => handlePromptChange(e.target.value)}
        disabled={disabled}
        className={`flex-1 p-6 border-2 rounded-xl focus:outline-none resize-none text-lg leading-relaxed shadow-sm min-h-0 ${
          disabled
            ? 'border-gray-300 bg-gray-100 text-gray-500 cursor-not-allowed'
            : 'border-gray-400 bg-white focus:ring-2 focus:ring-blue-500 focus:border-transparent'
        }`}
        placeholder={disabled ? "Prompt locked during conversation..." : "Start typing your prompt here..."}
      />
    </div>
  );
};
