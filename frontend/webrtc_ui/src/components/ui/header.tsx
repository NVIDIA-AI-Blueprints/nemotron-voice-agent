// // SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// // SPDX-License-Identifier: BSD-2-Clause

import { useState } from "react";
import { ArrowUpRight, Menu, X } from "lucide-react";

export const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const navLinks = [
    "Live demo",
    "Architecture",
    "Outcomes",
    "Compliance",
  ];

  return (
    <>
      <header className="sticky top-0 z-40 bg-white border-b-2 border-[#FB4E0B]">
        <div className="max-w-350 mx-auto px-8 h-16 flex items-center justify-between">
          {/* Left Side */}
          <div className="flex items-center gap-6">
            {/* Menu Icon */}
            <button
              onClick={() => setIsMenuOpen(true)}
              className="p-2 rounded-md hover:bg-gray-100 transition-colors"
              aria-label="Open Menu"
            >
              <Menu className="w-6 h-6 text-black" />
            </button>

            {/* Logos + Title */}
            <div className="flex items-center gap-4">
              <div className="flex items-center h-8">
                <img
                  src="logo2.png"
                  alt="NVIDIA Logo"
                  className="h-12 w-14 object-contain mr-2"
                />

                <img
                  src="logo.png"
                  alt="EXL Logo"
                  className="h-12 w-24 object-contain mr-2"
                />

                <img
                  src="crusoe_logo.jpeg"
                  alt="Crusoe Logo"
                  className="h-12 w-18 object-contain"
                />
              </div>

              <div className="font-bold text-[17px] tracking-[0.01em] text-black leading-[1.1] font-['Yantramanav']">
               Collections Agent
              </div>
            </div>
          </div>

          {/* Right Side */}
          <button className="flex items-center gap-2 px-4 py-2 text-[12px] rounded-xl font-bold tracking-[0.01em] text-white bg-[#FB4E0B] hover:bg-[#e03a00] transition-colors duration-200">
            Book a deep-dive
            <ArrowUpRight className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* Overlay */}
      <div
        onClick={() => setIsMenuOpen(false)}
        className={`fixed inset-0 bg-black/40 z-50 transition-opacity duration-300 ${
          isMenuOpen
            ? "opacity-100 pointer-events-auto"
            : "opacity-0 pointer-events-none"
        }`}
      />

      {/* Right Drawer */}
      <aside
        className={`fixed top-0 right-0 h-screen w-[320px] bg-white shadow-2xl z-50 transform transition-transform duration-300 ease-in-out ${
          isMenuOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        {/* Drawer Header */}
        <div className="flex items-center justify-between px-6 h-16 border-b">
          <h2 className="text-lg font-semibold">Menu</h2>

          <button
            onClick={() => setIsMenuOpen(false)}
            className="p-2 rounded-md hover:bg-gray-100"
            aria-label="Close Menu"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Drawer Links */}
        <nav className="flex flex-col p-6">
          {navLinks.map((link) => (
            <a
              key={link}
              href="#"
              onClick={() => setIsMenuOpen(false)}
              className="py-4 border-b border-gray-200 text-[15px] text-[#414141] hover:text-[#FB4E0B] transition-colors"
            >
              {link}
            </a>
          ))}
        </nav>

        {/* Drawer Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-6 border-t bg-white">
          <button className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-semibold text-white bg-[#FB4E0B] hover:bg-[#e03a00] transition-colors">
            Book a deep-dive
            <ArrowUpRight className="w-5 h-5" />
          </button>
        </div>
      </aside>
    </>
  );
};
