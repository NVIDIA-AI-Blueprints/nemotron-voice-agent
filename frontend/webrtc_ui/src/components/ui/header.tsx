// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

export const Header = () => {
    return (
        <header className="border-b-1 border-gray-200 p-6 flex items-center">
            <img src="logo.png" alt="NVIDIA Logo" className="h-16 mr-8" />
            <h1 className="text-2xl">Voice Agent Demo</h1>
        </header>
    );
};
