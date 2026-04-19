import { useState, useCallback } from "react";
import { useJob } from "./hooks/useJob";
import { processImages } from "./api/client";
import ImageGrid from "./components/ImageGrid";
import StagePanel from "./components/StagePanel";
import ResultViewer from "./components/ResultViewer";
import type { StageName } from "./types";

export default function App() {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<"gallery" | "results">("gallery");

  const { data: job } = useJob(jobId);

  const isRunning = job?.status === "running";

  const handleRun = useCallback(async (stages: StageName[]) => {
    setError(null);
    try {
      const { job_id } = await processImages(Array.from(selected), stages);
      setJobId(job_id);
      setView("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  }, [selected]);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Inscription Digitisation</h1>
          <p className="text-xs text-gray-500 mt-0.5">IDP Processing Pipeline</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setView("gallery")}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              view === "gallery"
                ? "bg-gray-800 text-white"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            Gallery
          </button>
          <button
            onClick={() => setView("results")}
            disabled={!jobId}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              view === "results"
                ? "bg-gray-800 text-white"
                : "text-gray-500 hover:text-gray-300 disabled:opacity-30"
            }`}
          >
            Results
            {job?.status === "running" && (
              <span className="ml-1.5 w-2 h-2 bg-indigo-400 rounded-full inline-block animate-pulse" />
            )}
          </button>
        </div>
      </header>

      {error && (
        <div className="mx-6 mt-4 px-4 py-3 bg-red-950 border border-red-800 rounded-lg text-sm text-red-300 flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-500 hover:text-red-300 ml-4">✕</button>
        </div>
      )}

      <main className="max-w-screen-xl mx-auto px-6 py-6">
        {view === "gallery" && (
          <>
            <StagePanel
              selectedCount={selected.size}
              onRun={handleRun}
              isRunning={isRunning}
            />
            <ImageGrid selected={selected} onSelectionChange={setSelected} />
          </>
        )}

        {view === "results" && job && (
          <ResultViewer job={job} />
        )}

        {view === "results" && !job && (
          <p className="text-gray-500 text-center py-20">
            No results yet. Select images and run the pipeline.
          </p>
        )}
      </main>
    </div>
  );
}
