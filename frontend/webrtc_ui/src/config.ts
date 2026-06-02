// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

// export const RTC_CONFIG = {};

// const host = window.location.hostname;

// export const RTC_OFFER_URL = `http://${host}:7860/offer`;


export const RTC_CONFIG: ConstructorParameters<typeof RTCPeerConnection>[0] = {
  iceServers: [
    {
      urls: "turn:216.86.161.41:3479",
      username: "admin",
      credential: "admin",
    },
  ],
};

const host = "216.86.161.41";

export const RTC_OFFER_URL = `http://${host}:7860/offer`;
