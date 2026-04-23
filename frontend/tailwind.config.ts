import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        gray: {
          950: "#030712",
        },
      },
      animation: {
        "wave-1": "wave1 8s ease-in-out infinite",
        "wave-2": "wave2 11s ease-in-out infinite",
        "wave-3": "wave3 14s ease-in-out infinite",
      },
      keyframes: {
        wave1: {
          "0%, 100%": { transform: "translateX(0px)" },
          "50%": { transform: "translateX(-60px)" },
        },
        wave2: {
          "0%, 100%": { transform: "translateX(0px)" },
          "50%": { transform: "translateX(80px)" },
        },
        wave3: {
          "0%, 100%": { transform: "translateX(0px)" },
          "50%": { transform: "translateX(-40px)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
