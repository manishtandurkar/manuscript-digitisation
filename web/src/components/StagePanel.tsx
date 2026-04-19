import { useState } from "react";
import type { StageName } from "../types";

const ALL_STAGES: { id: StageName; label: string }[] = [
  { id: "preprocess", label: "Preprocess" },
  { id: "enhance",    label: "Enhance (Super-res)" },
  { id: "binarise",   label: "Binarise" },
  { id: "ocr",        label: "OCR / Transcribe" },
  { id: "translate",  label: "Translate" },
];

interface Props {
  selectedCount: number;
  onRun: (stages: StageName[]) => void;
  isRunning: boolean;
}

export default function StagePanel({ selectedCount, onRun, isRunning }: Props) {
  const [checkedStages, setCheckedStages] = useState<Set<StageName>>(
    new Set(["preprocess"])
  );

  if (selectedCount === 0) return null;

  function toggleStage(id: StageName) {
    const next = new Set(checkedStages);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setCheckedStages(next);
  }

  function runSelected() {
    if (checkedStages.size > 0) onRun(Array.from(checkedStages));
  }

  function runFull() {
    onRun(ALL_STAGES.map((s) => s.id));
  }

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-5 mb-6">
      <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
        Pipeline stages
        <span className="ml-2 text-indigo-400 normal-case font-normal">
          — {selectedCount} image{selectedCount > 1 ? "s" : ""} selected
        </span>
      </h2>

      <div className="flex flex-wrap gap-3 mb-5">
        {ALL_STAGES.map((stage) => (
          <label key={stage.id} className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={checkedStages.has(stage.id)}
              onChange={() => toggleStage(stage.id)}
              className="w-4 h-4 accent-indigo-500"
            />
            <span className="text-sm text-gray-300">{stage.label}</span>
          </label>
        ))}
      </div>

      <div className="flex gap-3">
        <button
          onClick={runSelected}
          disabled={isRunning || checkedStages.size === 0}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium rounded-lg transition-colors"
        >
          {isRunning ? "Processing…" : "Run selected stages"}
        </button>
        <button
          onClick={runFull}
          disabled={isRunning}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white text-sm font-medium rounded-lg transition-colors"
        >
          Run full pipeline
        </button>
      </div>
    </div>
  );
}
