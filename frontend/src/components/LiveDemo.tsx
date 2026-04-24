"use client";
import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, ResponsiveContainer,
  Tooltip, ReferenceLine,
} from "recharts";
import { Play, Square, CheckCircle, XCircle } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const WS_URL = API.replace(/^http/, "ws");

const LABEL_STYLE: Record<string, { color: string; bg: string; text: string }> = {
  POSITIVE: { color: "text-green-400",  bg: "bg-green-400",  text: "POSITIVE" },
  NEUTRAL:  { color: "text-yellow-400", bg: "bg-yellow-400", text: "NEUTRAL"  },
  NEGATIVE: { color: "text-red-400",    bg: "bg-red-400",    text: "NEGATIVE" },
};

type Sample = {
  sample_index: number;
  true_label: string;
  predicted_label: string;
  confidence: number;
  rolling_accuracy: number;
  eeg_snapshot: number[];
};

type ClassCount = Record<string, { correct: number; total: number }>;

export default function LiveDemo() {
  const [status, setStatus]             = useState<"idle" | "running" | "done">("idle");
  const [current, setCurrent]           = useState<Sample | null>(null);
  const [eegData, setEegData]           = useState<{ i: number; v: number }[]>([]);
  const [accData, setAccData]           = useState<{ i: number; acc: number }[]>([]);
  const [feed, setFeed]                 = useState<Sample[]>([]);
  const [classCounts, setClassCounts]   = useState<ClassCount>({});
  const wsRef = useRef<WebSocket | null>(null);

  const start = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    setStatus("running");
    setCurrent(null);
    setEegData([]);
    setAccData([]);
    setFeed([]);
    setClassCounts({});

    const ws = new WebSocket(`${WS_URL}/simulate`);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.done) { setStatus("done"); return; }

      const s = msg as Sample;
      setCurrent(s);

      setEegData(s.eeg_snapshot.map((v, i) => ({ i, v })));
      setAccData((prev) => [...prev, { i: s.sample_index, acc: s.rolling_accuracy * 100 }]);
      setFeed((prev) => [s, ...prev].slice(0, 10));
      setClassCounts((prev) => {
        const lbl = s.true_label;
        const was = prev[lbl] ?? { correct: 0, total: 0 };
        return {
          ...prev,
          [lbl]: {
            total: was.total + 1,
            correct: was.correct + (s.predicted_label === lbl ? 1 : 0),
          },
        };
      });
    };

    ws.onerror = () => setStatus("idle");
    ws.onclose = () => { if (status === "running") setStatus("done"); };
  }, [status]);

  const stop = () => {
    wsRef.current?.close();
    setStatus("idle");
  };


  return (
    <section id="demo" className="py-28 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-4"
        >
          <p className="text-indigo-400 text-sm font-semibold uppercase tracking-widest mb-3">
            Interactive
          </p>
          <h2 className="text-4xl sm:text-5xl font-bold mb-4">Live Demo</h2>
          <p className="text-gray-400 max-w-xl mx-auto mb-8">
            200 held-out test samples streamed at 50ms intervals through the
            trained Random Forest model. Watch the predictions build in real time.
          </p>
          <div className="flex justify-center gap-3">
            {status !== "running" ? (
              <button
                onClick={start}
                className="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-6 py-3 rounded-xl font-semibold transition-all hover:scale-105"
              >
                <Play size={16} fill="currentColor" />
                {status === "done" ? "Restart Demo" : "Start Demo"}
              </button>
            ) : (
              <button
                onClick={stop}
                className="inline-flex items-center gap-2 bg-gray-700 hover:bg-gray-600 px-6 py-3 rounded-xl font-semibold transition-all"
              >
                <Square size={16} fill="currentColor" />
                Stop
              </button>
            )}
          </div>
        </motion.div>

        {/* Main panels */}
        <AnimatePresence>
          {(status === "running" || status === "done") && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid lg:grid-cols-3 gap-4 mt-10"
            >
              {/* EEG Waveform */}
              <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-4">
                  EEG Snapshot — 50 features
                </p>
                {current ? (
                  <>
                    <ResponsiveContainer width="100%" height={160}>
                      <LineChart data={eegData}>
                        <XAxis dataKey="i" hide />
                        <YAxis hide domain={["auto", "auto"]} />
                        <Line
                          type="monotone"
                          dataKey="v"
                          stroke="#6366f1"
                          strokeWidth={1.5}
                          dot={false}
                          isAnimationActive={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                    <p className="text-xs text-gray-600 mt-2 text-center">
                      Sample #{current.sample_index}
                    </p>
                  </>
                ) : (
                  <div className="h-[160px] flex flex-col gap-2 justify-center px-2">
                    {[0.6, 1, 0.4, 0.8, 0.5].map((w, i) => (
                      <div key={i} className="h-1.5 bg-gray-800 rounded-full animate-pulse" style={{ width: `${w * 100}%` }} />
                    ))}
                  </div>
                )}
              </div>

              {/* Emotion label + confidence */}
              <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 flex flex-col">
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-4">
                  Prediction
                </p>
                <div className="flex-1 flex items-center justify-center">
                  {current ? (
                    <span
                      className={`text-5xl font-bold tracking-tight ${
                        LABEL_STYLE[current.predicted_label]?.color ?? "text-white"
                      }`}
                    >
                      {current.predicted_label}
                    </span>
                  ) : (
                    <div className="h-12 w-36 bg-gray-800 rounded-xl animate-pulse" />
                  )}
                </div>
                <div className="space-y-2 mt-4">
                  {(["POSITIVE", "NEUTRAL", "NEGATIVE"] as const).map((lbl) => {
                    const isActive = current ? lbl === current.predicted_label : false;
                    return (
                      <div key={lbl} className="flex items-center gap-3">
                        <span className="text-xs text-gray-500 w-16">{lbl}</span>
                        <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                          <motion.div
                            className={`h-full rounded-full ${LABEL_STYLE[lbl].bg}`}
                            animate={{ width: isActive ? `${current!.confidence * 100}%` : "0%" }}
                            transition={{ duration: 0.2 }}
                          />
                        </div>
                        <span className="text-xs text-gray-400 w-10 text-right">
                          {isActive ? `${(current!.confidence * 100).toFixed(0)}%` : "—"}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Rolling accuracy */}
              <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
                <div className="flex items-center justify-between mb-4">
                  <p className="text-xs text-gray-500 uppercase tracking-widest">
                    Rolling Accuracy
                  </p>
                  <span className="text-lg font-bold text-white">
                    {current ? `${(current.rolling_accuracy * 100).toFixed(1)}%` : "—"}
                  </span>
                </div>
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={accData}>
                    <XAxis dataKey="i" hide />
                    <YAxis domain={[0, 100]} hide />
                    <Tooltip
                      contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
                      formatter={(v) => [`${Number(v).toFixed(1)}%`, "Accuracy"]}
                      labelFormatter={(l) => `Sample ${l}`}
                    />
                    <ReferenceLine y={98.58} stroke="#374151" strokeDasharray="4 2" />
                    <Line
                      type="monotone"
                      dataKey="acc"
                      stroke="#22c55e"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Prediction feed */}
        {feed.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-6 bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden"
          >
            <div className="px-5 py-3 border-b border-gray-800">
              <p className="text-xs text-gray-500 uppercase tracking-widest">
                Live Prediction Feed — last 10
              </p>
            </div>
            <div className="divide-y divide-gray-800">
              {feed.map((s) => {
                const correct = s.predicted_label === s.true_label;
                return (
                  <div key={s.sample_index} className="flex items-center gap-4 px-5 py-2.5 text-sm hover:bg-gray-800/40 transition-colors">
                    <span className="text-gray-600 w-12">#{s.sample_index}</span>
                    <span className="w-5">
                      {correct
                        ? <CheckCircle size={14} className="text-green-400" />
                        : <XCircle size={14} className="text-red-400" />}
                    </span>
                    <span className={`font-medium w-20 ${LABEL_STYLE[s.true_label]?.color}`}>
                      {s.true_label}
                    </span>
                    <span className="text-gray-500 text-xs">→</span>
                    <span className={`font-medium w-20 ${LABEL_STYLE[s.predicted_label]?.color}`}>
                      {s.predicted_label}
                    </span>
                    <span className="text-gray-500 ml-auto text-xs">
                      {(s.confidence * 100).toFixed(0)}% conf
                    </span>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}

        {/* Final summary */}
        {status === "done" && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 bg-green-950/30 border border-green-800/40 rounded-2xl p-6"
          >
            <h3 className="font-bold text-lg mb-4 text-green-400">
              Simulation Complete — Final Summary
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {Object.entries(classCounts).map(([lbl, { correct, total }]) => (
                <div key={lbl} className="bg-gray-900 rounded-xl p-4 text-center">
                  <div className={`text-xl font-bold mb-1 ${LABEL_STYLE[lbl]?.color}`}>
                    {((correct / total) * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-gray-500">{lbl}</div>
                  <div className="text-xs text-gray-600">{correct}/{total}</div>
                </div>
              ))}
              {accData.length > 0 && (
                <div className="bg-gray-900 rounded-xl p-4 text-center">
                  <div className="text-xl font-bold mb-1 text-white">
                    {accData[accData.length - 1].acc.toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">Overall Accuracy</div>
                  <div className="text-xs text-gray-600">200 samples</div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </section>
  );
}
