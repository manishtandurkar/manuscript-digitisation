import { useState } from "react";
import { createPortal } from "react-dom";
import type { Job, StageResult, ImageMeta } from "../types";
import ProgressBar from "./ProgressBar";
import ComparisonSlider from "./ComparisonSlider";
import { useImages } from "../hooks/useImages";

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
  return createPortal(
    <div
      className="fixed inset-0 bg-black/95 z-[9999] flex items-center justify-center p-6"
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
    </div>,
    document.body
  );
}

const METHOD_LABELS: Record<string, string> = {
  sauvola: "Sauvola",
  otsu: "Otsu",
  adaptive: "Adaptive",
};

const ENHANCE_MODE_LABELS: Record<string, string> = {
  dstretch: "DStretch",
  superres: "Super-res",
};

function StageResult({
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
    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
      {/* Stage header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 bg-gray-900/80">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-gray-200">{label}</span>
          {stage === "binarise" && result.method && (
            <span className="text-xs px-2 py-0.5 rounded-full border font-medium bg-gray-800 text-gray-500 border-gray-700">
              {METHOD_LABELS[result.method] ?? result.method}
            </span>
          )}
          {stage === "enhance" && result.mode && (
            <span className="text-xs px-2 py-0.5 rounded-full border font-medium bg-gray-800 text-gray-500 border-gray-700">
              {ENHANCE_MODE_LABELS[result.mode] ?? result.mode}
            </span>
          )}
        </div>
        <StatusBadge status={result.status} />
      </div>

      {/* Stage content */}
      <div className="p-4">
        {(result.status === "pending" || result.status === "running") && (
          <div className="h-64 bg-gray-800 rounded-lg animate-pulse flex items-center justify-center">
            <span className="text-gray-600 text-sm">Processing…</span>
          </div>
        )}

        {result.status === "skipped" && (
          <div className="h-64 bg-gray-900 border border-dashed border-gray-700 rounded-lg flex flex-col items-center justify-center gap-2">
            <span className="text-2xl opacity-30">⏭</span>
            <span className="text-xs text-gray-600">Not yet implemented</span>
          </div>
        )}

        {result.status === "failed" && (
          <div className="h-64 bg-red-950/40 border border-red-800 rounded-lg flex flex-col items-center justify-center gap-2 px-4">
            <span className="text-2xl opacity-50">⚠</span>
            <span className="text-xs text-red-400 text-center">{result.error ?? "Processing failed"}</span>
          </div>
        )}

        {result.status === "done" && result.url && (
          <div className="space-y-3">
            {/* Comparison slider — full width, taller */}
            <ComparisonSlider
              before={originalUrl}
              after={result.url}
              afterLabel={label}
              height="h-72"
            />
            <p className="text-xs text-gray-500 text-center">← drag slider to compare original vs {label.toLowerCase()} →</p>
            <div className="flex gap-2">
              <button
                onClick={() => onExpand(result.url!)}
                className="flex-1 text-xs text-indigo-400 hover:text-indigo-300 transition-colors py-1.5 px-3 bg-indigo-950/50 hover:bg-indigo-900/50 rounded-lg border border-indigo-800"
              >
                View {label.toLowerCase()}
              </button>
              <button
                onClick={() => onExpand(originalUrl)}
                className="flex-1 text-xs text-gray-500 hover:text-gray-200 transition-colors py-1.5 px-3 bg-gray-800 hover:bg-gray-700 rounded-lg"
              >
                View original
              </button>
            </div>
          </div>
        )}

        {result.status === "done" && result.text && (
          <div className="h-64 overflow-auto bg-gray-950 rounded-lg p-3 text-xs text-gray-300 whitespace-pre-wrap font-mono">
            {result.text}
          </div>
        )}
      </div>
    </div>
  );
}

function ImageResultCard({
  imageId,
  stageResults,
  imageMeta,
  originalUrl,
  thumbnailUrl,
  onExpand,
}: {
  imageId: string;
  stageResults: Record<string, StageResult>;
  imageMeta?: ImageMeta;
  originalUrl: string;
  thumbnailUrl: string;
  onExpand: (url: string) => void;
}) {
  const doneCount = Object.values(stageResults).filter(r => r.status === "done").length;
  const totalCount = Object.keys(stageResults).length;

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
      {/* Image header: original preview + metadata */}
      <div className="flex items-stretch border-b border-gray-800">
        {/* Original image — prominent */}
        <button
          onClick={() => onExpand(originalUrl)}
          className="relative w-40 h-36 flex-shrink-0 bg-gray-800 overflow-hidden group"
          title="View original full size"
        >
          {thumbnailUrl ? (
            <img
              src={thumbnailUrl}
              alt={imageId}
              className="absolute inset-0 w-full h-full object-cover group-hover:opacity-80 transition-opacity"
            />
          ) : (
            <div className="absolute inset-0 bg-gray-700 animate-pulse" />
          )}
          <div className="absolute inset-0 flex items-end p-2 bg-gradient-to-t from-black/70 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
            <span className="text-xs text-white font-medium">View original</span>
          </div>
          <span className="absolute top-2 left-2 text-xs bg-black/70 text-gray-300 px-1.5 py-0.5 rounded font-medium">
            Original
          </span>
        </button>

        {/* Metadata */}
        <div className="flex-1 px-5 py-4 flex flex-col justify-center gap-1.5">
          <p className="font-semibold text-gray-100 text-base">{imageMeta?.filename ?? imageId}</p>
          {imageMeta && (
            <p className="text-xs text-indigo-300 font-medium">{imageMeta.language}</p>
          )}
          <p className="text-xs text-gray-500">
            {doneCount} of {totalCount} stage{totalCount !== 1 ? "s" : ""} complete
          </p>
          {/* Stage status pills */}
          <div className="flex flex-wrap gap-1.5 mt-1">
            {Object.entries(stageResults).map(([stage, result]) => (
              <span
                key={stage}
                className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border font-medium ${
                  result.status === "done"
                    ? "bg-emerald-900/60 text-emerald-400 border-emerald-700"
                    : result.status === "failed"
                    ? "bg-red-900/60 text-red-400 border-red-700"
                    : result.status === "running"
                    ? "bg-indigo-900/60 text-indigo-400 border-indigo-700 animate-pulse"
                    : "bg-gray-800 text-gray-600 border-gray-700"
                }`}
              >
                {STAGE_LABELS[stage] ?? stage}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Stage results */}
      <div className="p-5 grid grid-cols-1 lg:grid-cols-2 gap-4">
        {Object.entries(stageResults).map(([stage, result]) => (
          <StageResult
            key={stage}
            stage={stage}
            result={result}
            originalUrl={originalUrl}
            onExpand={onExpand}
          />
        ))}
      </div>
    </div>
  );
}

export default function ResultViewer({ job }: Props) {
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);
  const { data: images } = useImages();

  // Build a lookup map from image id → ImageMeta for correct URLs
  const imageMap: Record<string, ImageMeta> = {};
  if (images) {
    for (const img of images) {
      imageMap[img.id] = img;
    }
  }

  return (
    <div className="space-y-6">
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
        const meta = imageMap[imageId];
        const originalUrl = meta?.url ?? "";
        const thumbnailUrl = meta?.thumbnail_url ?? meta?.url ?? "";

        return (
          <ImageResultCard
            key={imageId}
            imageId={imageId}
            stageResults={stageResults}
            imageMeta={meta}
            originalUrl={originalUrl}
            thumbnailUrl={thumbnailUrl}
            onExpand={setLightboxUrl}
          />
        );
      })}
    </div>
  );
}
