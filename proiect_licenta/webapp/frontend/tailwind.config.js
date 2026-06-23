/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          900: "#0b1220",
          800: "#0f1729",
          700: "#16203a",
          600: "#1e2b4a",
        },
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        grow: {
          "0%": { width: "0%" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.4s ease-out both",
        grow: "grow 0.7s ease-out both",
      },
    },
  },
  plugins: [],
};
