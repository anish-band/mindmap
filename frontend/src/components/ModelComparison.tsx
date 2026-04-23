"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Cell,
} from "recharts";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Model = {
  name: string;
  accuracy: number;
  f1: number;
  split: string;
};

const COLORS = ["#6366f1", "#8b5cf6", "#06b6d4", "#f59e0b"];
const SPLIT_LABEL: Record<string, string> = {
  random_80_20: "Random 80/20",
  subject_independent: "Subject-independent",
};

export default function ModelComparison() {
  const [models, setModels] = useState<Model[]>([]);

  useEffect(() => {
    fetch(`${API}/compare`)
      .then((r) => r.json())
      .then((d) => setModels(d.models))
      .catch(() => {});
  }, []);

  const chartData = models.map((m) => ({
    name: m.name,
    "Accuracy (%)": m.accuracy,
    "F1 (%)": m.f1,
  }));

  return (
    <section id="compare" className="py-28 px-4 bg-gray-900/30">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <p className="text-indigo-400 text-sm font-semibold uppercase tracking-widest mb-3">
            Benchmarks
          </p>
          <h2 className="text-4xl sm:text-5xl font-bold mb-4">
            Four Models, One Winner
          </h2>
          <p className="text-gray-400 max-w-xl mx-auto">
            Trained, benchmarked, and compared across accuracy, F1, and
            evaluation strategy.
          </p>
        </motion.div>

        {models.length > 0 && (
          <>
            {/* Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="bg-gray-900 border border-gray-800 rounded-2xl p-6 mb-6"
            >
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={chartData} barGap={4}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 12 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[60, 100]} tick={{ fill: "#9ca3af", fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
                  <Tooltip
                    contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
                    formatter={(v) => [`${Number(v).toFixed(2)}%`]}
                  />
                  <Legend wrapperStyle={{ color: "#9ca3af", fontSize: 12 }} />
                  <Bar dataKey="Accuracy (%)" radius={[6, 6, 0, 0]}>
                    {chartData.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} opacity={0.9} />
                    ))}
                  </Bar>
                  <Bar dataKey="F1 (%)" radius={[6, 6, 0, 0]} fill="#374151" opacity={0.7} />
                </BarChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Table */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden mb-6"
            >
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase tracking-widest">
                    <th className="text-left px-6 py-3">Model</th>
                    <th className="text-right px-6 py-3">Accuracy</th>
                    <th className="text-right px-6 py-3">F1</th>
                    <th className="text-right px-6 py-3">Eval Split</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {models.map((m, i) => (
                    <tr key={m.name} className="hover:bg-gray-800/40 transition-colors">
                      <td className="px-6 py-3.5 font-medium flex items-center gap-2">
                        <span
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ background: COLORS[i % COLORS.length] }}
                        />
                        {m.name}
                      </td>
                      <td className="px-6 py-3.5 text-right font-mono">
                        {m.accuracy.toFixed(2)}%
                      </td>
                      <td className="px-6 py-3.5 text-right font-mono text-gray-400">
                        {m.f1.toFixed(2)}%
                      </td>
                      <td className="px-6 py-3.5 text-right">
                        <span className={`text-xs px-2 py-0.5 rounded-full ${
                          m.split === "subject_independent"
                            ? "bg-amber-950/60 text-amber-400"
                            : "bg-indigo-950/60 text-indigo-400"
                        }`}>
                          {SPLIT_LABEL[m.split] ?? m.split}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </motion.div>
          </>
        )}

        {/* Callout */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl border border-amber-800/40 bg-amber-950/20 p-6"
        >
          <p className="text-amber-400 text-xs font-semibold uppercase tracking-widest mb-2">
            Why does LSTM drop to 76%?
          </p>
          <p className="text-gray-300 text-sm leading-relaxed">
            The LSTM was tested on a subject it{" "}
            <span className="font-semibold text-white">never saw during training</span> —
            a harder, more realistic evaluation called{" "}
            <span className="font-semibold text-white">subject-independent testing</span>.
            The other models were tested on random splits from the same subjects they
            trained on, which is an easier but less realistic scenario. The LSTM&apos;s 76%
            reflects genuine cross-subject generalization.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
