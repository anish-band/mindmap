"use client";
import { motion } from "framer-motion";

const STEPS = [
  {
    n: "01",
    title: "Data Collection",
    desc: "EEG signals recorded from subjects watching emotional video clips (positive, neutral, negative stimuli) via a Muse EEG headband across 4 electrode placements: AF7, AF8, TP9, TP10.",
    color: "bg-indigo-600",
  },
  {
    n: "02",
    title: "Feature Engineering",
    desc: "Raw signals transformed into 2,548 statistical features: band power ratios (theta/alpha, alpha/beta), left-right hemisphere asymmetry indices, and FFT spectral statistics up to 750 frequency bins.",
    color: "bg-purple-600",
  },
  {
    n: "03",
    title: "Model Training",
    desc: "Four ML architectures trained and evaluated: SVM (RBF kernel), Random Forest (200 trees), 1D CNN, and LSTM. Best performer: 1D CNN at 98.82% accuracy on a random 80/20 split.",
    color: "bg-violet-600",
  },
  {
    n: "04",
    title: "Explainability",
    desc: "SHAP TreeExplainer identifies which EEG features and brain regions most influence each prediction — connecting model outputs back to neuroscience concepts like frontal asymmetry and theta/alpha coherence.",
    color: "bg-cyan-600",
  },
  {
    n: "05",
    title: "Real-Time Inference",
    desc: "Trained Random Forest streams predictions at 50ms intervals simulating live EEG classification. Each sample returns a label, per-class confidence, and rolling accuracy.",
    color: "bg-teal-600",
  },
];

export default function HowItWorks() {
  return (
    <section id="how" className="py-28 px-4 bg-gray-900/30">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <p className="text-indigo-400 text-sm font-semibold uppercase tracking-widest mb-3">
            Pipeline
          </p>
          <h2 className="text-4xl sm:text-5xl font-bold">How It Works</h2>
        </motion.div>

        {/* Timeline */}
        <div className="relative">
          {/* Vertical line */}
          <div className="absolute left-[22px] top-0 bottom-0 w-px bg-gray-800 hidden sm:block" />

          <div className="space-y-10">
            {STEPS.map((s, i) => (
              <motion.div
                key={s.n}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                className="flex gap-6"
              >
                {/* Circle */}
                <div className={`flex-shrink-0 w-11 h-11 rounded-full ${s.color} flex items-center justify-center text-xs font-bold z-10`}>
                  {s.n}
                </div>

                {/* Card */}
                <div className="flex-1 bg-gray-900 border border-gray-800 rounded-2xl p-6 hover:border-gray-700 transition-colors mb-2">
                  <h3 className="font-semibold text-lg mb-2">{s.title}</h3>
                  <p className="text-gray-400 text-sm leading-relaxed">{s.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Callout */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mt-12 rounded-2xl border border-amber-800/40 bg-amber-950/20 p-6"
        >
          <p className="text-amber-400 text-xs font-semibold uppercase tracking-widest mb-2">
            Subject-Independent Evaluation
          </p>
          <p className="text-gray-300 text-sm leading-relaxed">
            The LSTM was tested on a held-out subject it{" "}
            <span className="font-semibold text-white">never saw during training</span>,
            achieving 76.49% accuracy — demonstrating real-world generalizability.
            The other models were evaluated on random splits from the same subjects,
            which is an easier (but less realistic) test.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
