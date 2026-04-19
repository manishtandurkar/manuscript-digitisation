import { useState } from "react";
import type { Job, StageResult } from "../types";
import ProgressBar from "./ProgressBar";
import ComparisonSlider from "./ComparisonSlider";

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

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    done:    "bg-emerald-900/60 text-emerald-400 border-emerald-700",
    failed:  "bg-red-900/60 text-red-400 border-red-700",
    skipped: "bg-gray-800 text-gray-500 border-gray-700",
    pending: "bg-gray-800 text-gray-600 border-gray-700",
    running: "bg-indigo-900/60 text-indigo-400 border-indigo-700 animate-pulse",
  };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${styles[status] ?? styles.pending}`}>
      {status}
    </span>
  );
}

function Lightbox({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <img
        src={src}
        alt="Full size"
        className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      />
      <button
        onClick={onClose}
        className="absolute top-4 right-4 w-9 h-9 bg-gray-800 hover:bg-gray-700 rounded-full flex items-center justify-center text-gray-400 hover:text-white transition-colors"
      >
        ✕
      </button>
    </div>
  );
}

function StageCard({
  stage,
  result,
  originalUrl,
  onExpand,
}: {
  stage: string;
  result: StageResult;
  originalUrl: string;
  onExpand: (url: string) => void;
}) {
  const label = STAGE_LABELS[stage] ?? stage;

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-200">{label}</span>
        <StatusBadge status={result.status} />
      </div>

      {(result.status === "pending" || result.status === "running") && (
        <div className="h-56 bg-gray-800 rounded-lg animate-pulse" />
      )}

      {result.status === "skipped" && (
        <div className="h-56 bg-gray-900 border border-dashed border-gray-700 rounded-lg flex flex-col items-center justify-center gap-2">
          <span className="text-2xl opacity-30">⏭</span>
          <span className="text-xs text-gray-600">Not yet implemented</span>
        </div>
      )}

      {result.status === "failed" && (
        <div className="h-56 bg-red-950/40 border border-red-800 rounded-lg flex flex-col items-center justify-center gap-2 px-4">
          <span className="text-2xl opacity-50">⚠</span>
          <span className="text-xs text-red-400 text-center">{result.error ?? "Processing failed"}</span>
        </div>
      )}

      {result.status === "done" && result.url && (
        <div className="space-y-2">
          <ComparisonSlider
            before={originalUrl}
            after={result.url}
            afterLabel={label}
            height="h-56"
          />
          <p className="text-xs text-gray-500 text-center">← drag to compare →</p>
          <button
            onClick={() => onExpand(result.url!)}
            className="w-full text-xs text-gray-500 hover:text-indigo-400 transition-colors py-1"
          >
            Click to expand full size
          </button>
        </div>
      )}

      {result.status === "done" && result.text && (
        <div className="h-56 overflow-auto bg-gray-900 rounded-lg p-3 text-xs text-gray-300 whitespace-pre-wrap font-mono">
          {result.text}
        </div>
      )}
    </div>
  );
}

export default function ResultViewer({ job }: Props) {
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);

  return (
    <div className="space-y-8">
      {lightboxUrl && (
        <Lightbox src={lightboxUrl} onClose={() => setLightboxUrl(null)} />
      )}

      {job.status === "running" && (
        <ProgressBar completed={job.completed} total={job.total} />
      )}

      {job.status === "done" && (
        <div className="flex items-center gap-2 text-sm text-emerald-400">
          <span className="w-2 h-2 bg-emerald-400 rounded-full" />
          Processing complete — {job.total} image{job.total > 1 ? "s" : ""}
        </div>
      )}

      {Object.entries(job.results).map(([imageId, stageResults]) => {
        const originalUrl = `/data/raw/tamil_stone/${imageId}.jpg`;
        return (
          <div key={imageId} className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
            {/* Image header */}
            <div className="flex items-center gap-4 px-5 py-4 border-b border-gray-800">
              <img
                src={originalUrl}
                alt={imageId}
                className="w-14 h-14 object-cover rounded-lg bg-gray-800 flex-shrink-0"
              />
              <div>
                <p className="font-medium text-gray-100">{imageId}</p>
                <p className="text-xs text-gray-500 mt-0.5">
                  {Object.values(stageResults).filter(r => r.status === "done").length} of{" "}
                  {Object.keys(stageResults).length} stages complete
                </p>
              </div>
              <button
                onClick={() => setLightboxUrl(originalUrl)}
                className="ml-auto text-xs text-gray-500 hover:text-indigo-400 transition-colors"
              >
                View original
              </button>
            </div>

            {/* Stage cards grid */}
            <div className="p-5 grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
              {Object.entries(stageResults).map(([stage, result]) => (
                <StageCard
                  key={stage}
                  stage={stage}
                  result={result}
                  originalUrl={originalUrl}
                  onExpand={setLightboxUrl}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
