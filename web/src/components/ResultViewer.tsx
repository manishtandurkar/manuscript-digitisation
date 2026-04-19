import type { Job, StageResult } from "../types";
import ProgressBar from "./ProgressBar";

interface Props {
  job: Job;
}

const STAGE_LABELS: Record<string, string> = {
  preprocess: "Preprocessed",
  enhance: "Enhanced",
  binarise: "Binarised",
  ocr: "OCR Text",
  translate: "Translation",
};

function StageOutput({ stage, result }: { stage: string; result: StageResult }) {
  const label = STAGE_LABELS[stage] ?? stage;

  if (result.status === "pending" || result.status === "running") {
    return (
      <div className="flex flex-col items-center gap-2">
        <div className="w-full h-36 bg-gray-800 rounded-lg animate-pulse" />
        <span className="text-xs text-gray-500">{label} — waiting…</span>
      </div>
    );
  }

  if (result.status === "skipped") {
    return (
      <div className="flex flex-col items-center gap-2">
        <div className="w-full h-36 bg-gray-900 border border-dashed border-gray-700 rounded-lg flex items-center justify-center">
          <span className="text-xs text-gray-600">Not implemented</span>
        </div>
        <span className="text-xs text-gray-600">{label}</span>
      </div>
    );
  }

  if (result.status === "failed") {
    return (
      <div className="flex flex-col items-center gap-2">
        <div className="w-full h-36 bg-red-950 border border-red-800 rounded-lg flex items-center justify-center px-2">
          <span className="text-xs text-red-400 text-center">{result.error ?? "Error"}</span>
        </div>
        <span className="text-xs text-red-500">{label} — failed</span>
      </div>
    );
  }

  if (result.url) {
    return (
      <div className="flex flex-col items-center gap-2">
        <img src={result.url} alt={label} className="w-full h-36 object-cover rounded-lg bg-gray-800" />
        <span className="text-xs text-gray-400">{label}</span>
      </div>
    );
  }

  if (result.text) {
    return (
      <div className="flex flex-col gap-2">
        <div className="w-full min-h-36 bg-gray-800 rounded-lg p-3 text-xs text-gray-300 whitespace-pre-wrap">
          {result.text}
        </div>
        <span className="text-xs text-gray-400">{label}</span>
      </div>
    );
  }

  return null;
}

export default function ResultViewer({ job }: Props) {
  return (
    <div>
      {job.status === "running" && (
        <ProgressBar completed={job.completed} total={job.total} />
      )}

      <div className="space-y-6">
        {Object.entries(job.results).map(([imageId, stageResults]) => (
          <div key={imageId} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-xs text-gray-500 mb-3 font-mono">{imageId}</p>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
              <div className="flex flex-col items-center gap-2">
                <img
                  src={`/data/raw/tamil_stone/${imageId}.jpg`}
                  alt="Original"
                  className="w-full h-36 object-cover rounded-lg bg-gray-800"
                />
                <span className="text-xs text-gray-400">Original</span>
              </div>

              {Object.entries(stageResults).map(([stage, result]) => (
                <StageOutput key={stage} stage={stage} result={result} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
