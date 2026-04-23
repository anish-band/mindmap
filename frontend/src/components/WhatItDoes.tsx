"use client";
import { motion } from "framer-motion";
import { Zap, BarChart2, Eye } from "lucide-react";

const FEATURES = [
  {
    icon: Zap,
    color: "text-indigo-400",
    bg: "bg-indigo-950/50 border-indigo-800/40",
    title: "Real-Time Classification",
    desc: "Streams EEG signals through a trained ML model and predicts emotional state live with per-class confidence scores, updating every 50ms.",
  },
  {
    icon: BarChart2,
    color: "text-purple-400",
    bg: "bg-purple-950/50 border-purple-800/40",
    title: "Multi-Model Comparison",
    desc: "Benchmarks SVM, Random Forest, 1D CNN, and LSTM architectures to find the most accurate and generalizable approach across different evaluation strategies.",
  },
  {
    icon: Eye,
    color: "text-cyan-400",
    bg: "bg-cyan-950/50 border-cyan-800/40",
    title: "Explainable AI",
    desc: "Uses SHAP values to reveal which brain regions and frequency features drive each prediction, making the black-box model transparent and interpretable.",
  },
];

const STATS = [
  { value: "2,132", label: "EEG samples" },
  { value: "2,548", label: "features per sample" },
  { value: "3", label: "emotion classes" },
  { value: "98.82%", label: "best accuracy (1D CNN)" },
];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.15 } },
};
const item = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.25, 0.1, 0.25, 1] as const } },
};

export default function WhatItDoes() {
  return (
    <section id="what" className="py-28 px-4 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
        className="text-center mb-16"
      >
        <p className="text-indigo-400 text-sm font-semibold uppercase tracking-widest mb-3">
          Capabilities
        </p>
        <h2 className="text-4xl sm:text-5xl font-bold">What It Does</h2>
        <p className="mt-4 text-gray-400 max-w-xl mx-auto">
          A complete pipeline from raw EEG signals to explainable, real-time
          emotion predictions.
        </p>
      </motion.div>

      {/* Feature cards */}
      <motion.div
        variants={container}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true }}
        className="grid md:grid-cols-3 gap-6 mb-16"
      >
        {FEATURES.map((f) => (
          <motion.div
            key={f.title}
            variants={item}
            className={`rounded-2xl border p-8 ${f.bg} hover:scale-[1.02] transition-transform cursor-default`}
          >
            <div className={`w-12 h-12 rounded-xl bg-gray-900 flex items-center justify-center mb-5 ${f.color}`}>
              <f.icon size={22} />
            </div>
            <h3 className="text-lg font-semibold mb-3">{f.title}</h3>
            <p className="text-gray-400 text-sm leading-relaxed">{f.desc}</p>
          </motion.div>
        ))}
      </motion.div>

      {/* Dataset stats */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
        className="rounded-2xl border border-gray-800 bg-gray-900/50 p-8"
      >
        <p className="text-center text-sm text-gray-500 mb-6 uppercase tracking-widest">
          Dataset — Muse EEG Headband · AF7, AF8, TP9, TP10 electrodes
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
          {STATS.map((s) => (
            <div key={s.label} className="text-center">
              <div className="text-3xl font-bold text-white mb-1">{s.value}</div>
              <div className="text-sm text-gray-500">{s.label}</div>
            </div>
          ))}
        </div>
      </motion.div>
    </section>
  );
}
