import { useState } from "react";
import type { StageName } from "../types";

const ALL_STAGES: { id: StageName; label: string }[] = [
  { id: "preprocess", label: "Preprocess" },
  { id: "enhance",    label: "Enhance (Super-res)" },
  { id: "binarise",   label: "Binarise" },
  { id: "ocr",        label: "OCR / Transcribe" },
  { id: "translate",  label: "Translate" },
];

const ENHANCE_MODES: { value: string; label: string; description: string }[] = [
  { value: "dstretch",  label: "DStretch",  description: "Fast · decorrelation stretch, reveals faded pigment" },
  { value: "superres",  label: "Super-res", description: "Slow · Real-ESRGAN 4× upscale (CPU: ~1–3 min/image)" },
];

const BINARISE_METHODS: { value: string; label: string; description: string }[] = [
  { value: "sauvola", label: "Sauvola",  description: "Best for uneven backgrounds" },
  { value: "otsu",    label: "Otsu",     description: "Fast, good for clean paper" },
  { value: "adaptive",label: "Adaptive", description: "Mixed quality fallback" },
];

interface Props {
  selectedCount: number;
  onRun: (stages: StageName[], stageOptions: Record<string, Record<string, string>>) => void;
  isRunning: boolean;
}

export default function StagePanel({ selectedCount, onRun, isRunning }: Props) {
  const [checkedStages, setCheckedStages] = useState<Set<StageName>>(
    new Set(["preprocess"])
  );
  const [binariseMethod, setBinariseMethod] = useState("sauvola");
  const [enhanceMode, setEnhanceMode] = useState("dstretch");

  if (selectedCount === 0) return null;

  function toggleStage(id: StageName) {
    const next = new Set(checkedStages);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setCheckedStages(next);
  }

  function buildOptions(): Record<string, Record<string, string>> {
    const opts: Record<string, Record<string, string>> = {};
    if (checkedStages.has("enhance"))   opts.enhance   = { mode: enhanceMode };
    if (checkedStages.has("binarise"))  opts.binarise  = { method: binariseMethod };
    return opts;
  }

  function runSelected() {
    if (checkedStages.size > 0) onRun(Array.from(checkedStages), buildOptions());
  }

  function runFull() {
    onRun(ALL_STAGES.map((s) => s.id), {
      enhance:  { mode: enhanceMode },
      binarise: { method: binariseMethod },
    });
  }

  const enhanceChecked  = checkedStages.has("enhance");
  const binariseChecked = checkedStages.has("binarise");

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-5 mb-6">
      <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
        Pipeline stages
        <span className="ml-2 text-indigo-400 normal-case font-normal">
          — {selectedCount} image{selectedCount > 1 ? "s" : ""} selected
        </span>
      </h2>

      <div className="flex flex-wrap gap-3 mb-4">
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

      {enhanceChecked && (
        <div className="mb-4 flex items-center gap-3 pl-1 flex-wrap">
          <span className="text-xs text-gray-500 uppercase tracking-wider">Enhance mode</span>
          <div className="flex flex-wrap gap-2">
            {ENHANCE_MODES.map((m) => (
              <button
                key={m.value}
                onClick={() => setEnhanceMode(m.value)}
                title={m.description}
                className={`text-xs px-3 py-1.5 rounded-lg border font-medium transition-colors ${
                  enhanceMode === m.value
                    ? "bg-indigo-600 border-indigo-500 text-white"
                    : "bg-gray-800 border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-600"
                }`}
              >
                {m.label}
                {m.value === "superres" && (
                  <span className="ml-1.5 text-[10px] bg-amber-900/60 text-amber-300 px-1 py-0.5 rounded">
                    SLOW
                  </span>
                )}
              </button>
            ))}
          </div>
          <span className="text-xs text-gray-600 italic">
            {ENHANCE_MODES.find((m) => m.value === enhanceMode)?.description}
          </span>
        </div>
      )}

      {binariseChecked && (
        <div className="mb-5 flex items-center gap-3 pl-1 flex-wrap">
          <span className="text-xs text-gray-500 uppercase tracking-wider">Binarise method</span>
          <div className="flex flex-wrap gap-2">
            {BINARISE_METHODS.map((m) => (
              <button
                key={m.value}
                onClick={() => setBinariseMethod(m.value)}
                title={m.description}
                className={`text-xs px-3 py-1.5 rounded-lg border font-medium transition-colors ${
                  binariseMethod === m.value
                    ? "bg-indigo-600 border-indigo-500 text-white"
                    : "bg-gray-800 border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-600"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
          <span className="text-xs text-gray-600 italic">
            {BINARISE_METHODS.find((m) => m.value === binariseMethod)?.description}
          </span>
        </div>
      )}

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
