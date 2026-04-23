import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MindMap — EEG Emotion Classification",
  description:
    "End-to-end EEG emotion classification using machine learning. Detect positive, neutral, and negative emotional states from brainwave data in real time.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className="bg-gray-950 text-white antialiased">{children}</body>
    </html>
  );
}
