// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

import { useRef, useEffect } from "react";

interface AudioWaveFormProps {
  streamOrTrack?: MediaStream | MediaStreamTrack | null;
  width?: number;
  height?: number;
  lineColor?: string;
  backgroundColor?: string;
}

export function AudioWaveForm({
  streamOrTrack,
  width = 300,
  height = 80,
  lineColor = "#76B900", // NVIDIA green color
  backgroundColor = "transparent",
}: AudioWaveFormProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const canvasCtx = canvas.getContext("2d");
    if (!canvasCtx) return;

    // Clear canvas and apply background color
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    if (backgroundColor !== "transparent") {
      canvasCtx.fillStyle = backgroundColor;
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    }

    // If no stream is provided, we just leave the canvas empty
    if (!streamOrTrack) return;

    let stream: MediaStream;
    if (streamOrTrack instanceof MediaStreamTrack) {
      // If a track is provided, create a MediaStream from it
      stream = new MediaStream([streamOrTrack]);
    } else {
      stream = streamOrTrack;
    }

    const audioContext = new AudioContext();
    const analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);

    // Connect source to analyzer
    source.connect(analyser);
    analyser.fftSize = 256;

    // Store references for cleanup
    audioContextRef.current = audioContext;
    analyserRef.current = analyser;
    sourceRef.current = source;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);

      const width = canvas.width;
      const height = canvas.height;

      analyser.getByteTimeDomainData(dataArray);

      // Clear canvas
      canvasCtx.clearRect(0, 0, width, height);

      if (backgroundColor !== "transparent") {
        canvasCtx.fillStyle = backgroundColor;
        canvasCtx.fillRect(0, 0, width, height);
      }

      // Draw histogram
      const barCount = 12;
      const barWidth = width / barCount;
      const barSpacing = barWidth * 0.1; // 10% of barWidth for spacing
      const adjustedBarWidth = barWidth - barSpacing;

      canvasCtx.fillStyle = lineColor;

      // Process audio data for each bar
      for (let i = 0; i < barCount; i++) {
        // Calculate which portion of the data to use for this bar
        const dataStart = Math.floor((i / barCount) * bufferLength);
        const dataEnd = Math.floor(((i + 1) / barCount) * bufferLength);

        // Calculate average value for this section of data
        let sum = 0;
        for (let j = dataStart; j < dataEnd; j++) {
          // Convert from 0-255 scale to amplitude (-1 to 1)
          const amplitude = (dataArray[j] - 128) / 128;
          sum += Math.abs(amplitude); // Use absolute value for energy level
        }

        const avgAmplitude = sum / (dataEnd - dataStart) || 0;

        const minBarHeight = height * 0.02;
        const scalingFactor = 8;
        const barHeight = Math.max(
          minBarHeight,
          height * Math.min(1, Math.pow(avgAmplitude * scalingFactor, 1.2))
        );

        // Draw the bar (from bottom of canvas)
        canvasCtx.fillRect(
          i * barWidth + barSpacing / 2,
          height - barHeight,
          adjustedBarWidth,
          barHeight
        );
      }
    };

    // Start animation
    draw();

    // Cleanup function
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }

      if (sourceRef.current) {
        sourceRef.current.disconnect();
      }

      if (
        audioContextRef.current &&
        audioContextRef.current.state !== "closed"
      ) {
        audioContextRef.current.close();
      }
    };
  }, [streamOrTrack, lineColor, backgroundColor]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        width: "100%",
        maxWidth: `${width}px`,
        height: `${height}px`,
        borderRadius: "4px",
      }}
    />
  );
}
