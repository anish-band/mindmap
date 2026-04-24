"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { Loader2, Brain } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Feature = {
  feature: string;
  importance: number;
  shap_positive: number;
  brain_region: string;
};

type ExplainResult = {
  prediction: string;
  confidence: Record<string, number>;
  top_features: Feature[];
};

const LABEL_COLOR: Record<string, string> = {
  POSITIVE: "text-green-400",
  NEUTRAL: "text-yellow-400",
  NEGATIVE: "text-red-400",
};

const REGION_META: Record<string, { label: string; color: string; bg: string }> = {
  frontal_left:              { label: "Left Frontal (AF7)",          color: "#6366f1", bg: "bg-indigo-950/40 border-indigo-800/40" },
  frontal_right:             { label: "Right Frontal (AF8)",         color: "#8b5cf6", bg: "bg-purple-950/40 border-purple-800/40" },
  temporal_parietal_left:    { label: "Left Temporal-Parietal (TP9)",color: "#06b6d4", bg: "bg-cyan-950/40 border-cyan-800/40"    },
  temporal_parietal_right:   { label: "Right Temporal-Parietal (TP10)", color: "#14b8a6", bg: "bg-teal-950/40 border-teal-800/40" },
  band_ratios:               { label: "Band Ratios",                 color: "#f59e0b", bg: "bg-amber-950/40 border-amber-800/40"  },
  interhemispheric:          { label: "Interhemispheric Coherence",  color: "#22c55e", bg: "bg-green-950/40 border-green-800/40"  },
};

function regionScore(features: Feature[], region: string) {
  const feats = features.filter((f) => f.brain_region === region);
  if (!feats.length) return 0;
  return feats.reduce((s, f) => s + f.importance, 0) / feats.length;
}

export default function Explainability() {
  const [sample, setSample]     = useState<number[] | null>(null);
  const [trueLabel, setTrueLabel] = useState<string>("");
  const [loading, setLoading]   = useState(false);
  const [explaining, setExplaining] = useState(false);
  const [result, setResult]     = useState<ExplainResult | null>(null);

  // Fetch a sample on mount
  useEffect(() => {
    setLoading(true);
    fetch(`${API}/sample`)
      .then((r) => r.json())
      .then((d) => {
        setSample(d.features);
        setTrueLabel(d.true_label);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const explain = async () => {
    if (!sample) return;
    setExplaining(true);
    setResult(null);
    try {
      const r = await fetch(`${API}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: sample }),
      });
      setResult(await r.json());
    } catch {}
    setExplaining(false);
  };

  const loadNew = () => {
    setLoading(true);
    setResult(null);
    fetch(`${API}/sample`)
      .then((r) => r.json())
      .then((d) => { setSample(d.features); setTrueLabel(d.true_label); })
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  const chartData = result?.top_features.slice(0, 20).map((f) => ({
    name: f.feature.length > 14 ? f.feature.slice(0, 14) + "…" : f.feature,
    fullName: f.feature,
    importance: parseFloat(f.importance.toFixed(5)),
    region: f.brain_region,
  })) ?? [];

  const regions = Object.keys(REGION_META);

  return (
    <section id="explain" className="py-28 px-4">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <p className="text-indigo-400 text-sm font-semibold uppercase tracking-widest mb-3">
            Explainability
          </p>
          <h2 className="text-4xl sm:text-5xl font-bold mb-4">
            What Did the Model Learn?
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto text-sm leading-relaxed">
            SHAP (SHapley Additive exPlanations) values reveal which EEG features
            contributed most to each prediction — connecting model outputs back to
            neuroscience and making the black-box interpretable.
          </p>
        </motion.div>

        {/* Sample card + explain button */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="bg-gray-900 border border-gray-800 rounded-2xl p-6 mb-6"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
            <div>
              <p className="text-sm text-gray-400">Pre-loaded sample from held-out test set</p>
              {trueLabel && (
                <p className="text-xs text-gray-600 mt-0.5">
                  True label:{" "}
                  <span className={`font-semibold ${LABEL_COLOR[trueLabel] ?? "text-white"}`}>
                    {trueLabel}
                  </span>
                </p>
              )}
            </div>
            <div className="flex gap-2">
              <button
                onClick={loadNew}
                disabled={loading}
                className="text-sm bg-gray-800 hover:bg-gray-700 px-4 py-2 rounded-lg transition-colors"
              >
                {loading ? "Loading…" : "Load New Sample"}
              </button>
              <button
                onClick={explain}
                disabled={!sample || explaining}
                className="inline-flex items-center gap-2 text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2 rounded-lg transition-colors font-medium"
              >
                {explaining ? <Loader2 size={14} className="animate-spin" /> : <Brain size={14} />}
                {explaining ? "Running SHAP…" : "Explain Prediction"}
              </button>
            </div>
          </div>
          {sample && (
            <div className="bg-gray-950 rounded-xl p-4 font-mono text-xs text-gray-500 overflow-hidden">
              [{sample.slice(0, 12).map((v) => v.toFixed(2)).join(", ")}, …{" "}
              <span className="text-gray-600">+{sample.length - 12} more</span>]
            </div>
          )}
        </motion.div>

        {/* Loading skeleton */}
        {explaining && !result && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6">
              <div className="h-3 w-20 bg-gray-800 rounded animate-pulse mb-4" />
              <div className="h-10 w-40 bg-gray-800 rounded-lg animate-pulse" />
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6">
              <div className="h-3 w-52 bg-gray-800 rounded animate-pulse mb-6" />
              <div className="space-y-3">
                {[...Array(8)].map((_, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <div className="h-3 rounded bg-gray-800 animate-pulse" style={{ width: `${90 + i * 10}px` }} />
                    <div className="h-4 flex-1 rounded bg-gray-800 animate-pulse" style={{ opacity: 1 - i * 0.09 }} />
                  </div>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="rounded-xl border border-gray-800 p-4">
                  <div className="h-3 w-28 bg-gray-800 rounded animate-pulse mb-3" />
                  <div className="h-7 w-16 bg-gray-800 rounded animate-pulse mb-3" />
                  <div className="h-1 bg-gray-800 rounded-full animate-pulse" />
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Results */}
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Prediction + confidence */}
            <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6">
              <div className="flex flex-wrap items-center gap-6">
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">
                    Prediction
                  </p>
                  <p className={`text-3xl font-bold ${LABEL_COLOR[result.prediction] ?? "text-white"}`}>
                    {result.prediction}
                  </p>
                </div>
                <div className="flex gap-4 flex-wrap">
                  {Object.entries(result.confidence).map(([lbl, conf]) => (
                    <div key={lbl} className="text-center">
                      <div className={`text-lg font-bold ${LABEL_COLOR[lbl] ?? "text-gray-300"}`}>
                        {(conf * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-gray-600">{lbl}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* SHAP bar chart */}
            <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6">
              <p className="text-xs text-gray-500 uppercase tracking-widest mb-4">
                Top 20 Features by SHAP Importance
              </p>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fill: "#9ca3af", fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    dataKey="name"
                    type="category"
                    tick={{ fill: "#9ca3af", fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                    width={110}
                  />
                  <Tooltip
                    contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
                    formatter={(v, _, p) => [Number(v).toExponential(3), (p as { payload?: { fullName?: string } })?.payload?.fullName ?? ""]}
                  />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {chartData.map((d, i) => {
                      const meta = REGION_META[d.region];
                      return <Cell key={i} fill={meta?.color ?? "#6366f1"} />;
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Brain region cards */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              {regions.map((r) => {
                const meta = REGION_META[r];
                const score = regionScore(result.top_features, r);
                const maxScore = Math.max(...regions.map((rr) => regionScore(result.top_features, rr)));
                const pct = maxScore > 0 ? score / maxScore : 0;
                return (
                  <div key={r} className={`rounded-xl border p-4 ${meta.bg}`}>
                    <p className="text-xs text-gray-400 mb-2">{meta.label}</p>
                    <div className="text-xl font-bold mb-2" style={{ color: meta.color }}>
                      {score > 0 ? score.toExponential(2) : "—"}
                    </div>
                    <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{ width: `${pct * 100}%`, background: meta.color }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </div>
    </section>
  );
}
