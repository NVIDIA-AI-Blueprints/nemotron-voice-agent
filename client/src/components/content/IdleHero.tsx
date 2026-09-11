// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export function IdleHero({ connecting, fadingOut }: Readonly<{ connecting: boolean; fadingOut: boolean }>) {
  const showSpinner = connecting || fadingOut;

  return (
    <div className="idle-hero">
      <div className="idle-hero__icon">
        <svg viewBox="0 0 120 120" width="120" height="120" fill="none">
          {showSpinner ? (
            <>
              <circle
                cx="60" cy="60" r="58"
                stroke="var(--nvidia-green)" strokeWidth="2.5" opacity="0.5"
                strokeDasharray="91 274" strokeLinecap="round"
              >
                <animateTransform attributeName="transform" type="rotate"
                  from="0 60 60" to="360 60 60" dur="1.4s" repeatCount="indefinite" />
              </circle>
              <circle
                cx="60" cy="60" r="44"
                stroke="var(--nvidia-green)" strokeWidth="2" opacity="0.35"
                strokeDasharray="69 207" strokeLinecap="round"
              >
                <animateTransform attributeName="transform" type="rotate"
                  from="360 60 60" to="0 60 60" dur="1.8s" repeatCount="indefinite" />
              </circle>
            </>
          ) : (
            <>
              <circle cx="60" cy="60" r="58" stroke="var(--nvidia-green)" strokeWidth="2" opacity="0.15" />
              <circle cx="60" cy="60" r="44" stroke="var(--nvidia-green)" strokeWidth="1.5" opacity="0.25" />
            </>
          )}
          <g>
            {[26, 38, 50, 62, 74, 86, 98].map((x, i) => (
              <rect
                key={x}
                x={x - 3}
                y={48}
                width="6"
                rx="3"
                height="24"
                fill="var(--nvidia-green)"
                opacity="0.6"
              >
                <animate
                  attributeName="height"
                  values={`24;${12 + ((i * 7 + 3) % 36)};24`}
                  dur={`${1.2 + i * 0.15}s`}
                  repeatCount="indefinite"
                />
                <animate
                  attributeName="y"
                  values={`48;${60 - (12 + ((i * 7 + 3) % 36)) / 2};48`}
                  dur={`${1.2 + i * 0.15}s`}
                  repeatCount="indefinite"
                />
              </rect>
            ))}
          </g>
        </svg>
      </div>

      <div className="idle-hero__text">
        {showSpinner ? (
          <>
            <h2 className="idle-hero__title">Connecting&hellip;</h2>
            <p className="idle-hero__subtitle">Setting up your voice session</p>
          </>
        ) : (
          <>
            <h2 className="idle-hero__title">Ready to Chat</h2>
            <p className="idle-hero__subtitle">
              Press <span className="idle-hero__accent">Connect</span> to start a voice conversation
            </p>
          </>
        )}
      </div>

      <div className={`idle-hero__hints ${showSpinner ? "idle-hero__hints--hidden" : ""}`}>
        <span className="idle-hero__hint">Try: &ldquo;I'd like to order some flowers&rdquo;</span>
        <span className="idle-hero__hint">Try: &ldquo;What do you recommend for a birthday?&rdquo;</span>
      </div>
    </div>
  );
}
