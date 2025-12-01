"use client"

import React, { useEffect, useState } from "react";

export default function ThemeToggle() {
  const [isDark, setIsDark] = useState<boolean>(false);

  useEffect(() => {
    try {
      const stored = localStorage.getItem("theme");
      if (stored === "dark") {
        document.documentElement.classList.add("dark");
        setIsDark(true);
      } else if (stored === "light") {
        document.documentElement.classList.remove("dark");
        setIsDark(false);
      } else {
        // respect system preference
        const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
        if (prefersDark) {
          document.documentElement.classList.add("dark");
          setIsDark(true);
        }
      }
    } catch (e) {
      // ignore
    }
  }, []);

  const toggle = () => {
    try {
      if (document.documentElement.classList.contains("dark")) {
        document.documentElement.classList.remove("dark");
        localStorage.setItem("theme", "light");
        setIsDark(false);
      } else {
        document.documentElement.classList.add("dark");
        localStorage.setItem("theme", "dark");
        setIsDark(true);
      }
    } catch (e) {
      // ignore
    }
  };

  return (
    <label className="inline-flex items-center cursor-pointer select-none" aria-label="Toggle theme">
      <input
        type="checkbox"
        checked={isDark}
        onChange={toggle}
        className="sr-only"
      />
      <span
        className={`relative inline-block w-12 h-6 rounded-full border transition-colors ${
          isDark
            ? 'bg-gray-700 border-gray-600'
            : 'bg-gray-200 border-gray-300'
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full transition-all shadow ${
            isDark
              ? 'translate-x-6 bg-gray-100'
              : 'translate-x-0 bg-white'
          }`}
        />
      </span>
      <span className={`ml-3 text-sm font-medium ${isDark ? 'text-gray-100' : 'text-gray-900'}`}>
        {isDark ? 'Dark' : 'Light'}
      </span>
    </label>
  );
}
