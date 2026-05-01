import { useState, useCallback } from "react";
import { useJob } from "./hooks/useJob";
import { processImages } from "./api/client";
import ImageGrid from "./components/ImageGrid";
import StagePanel from "./components/StagePanel";
import ResultViewer from "./components/ResultViewer";
import type { StageName } from "./types";

type View = "gallery" | "results";

function NavItem({
  active,
  onClick,
  disabled,
  children,
}: {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors text-left ${
        active
          ? "bg-indigo-600 text-white"
          : "text-gray-400 hover:text-gray-200 hover:bg-gray-800 disabled:opacity-30 disabled:cursor-not-allowed"
      }`}
    >
      {children}
    </button>
  );
}

export default function App() {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("gallery");

  const { data: job } = useJob(jobId);
  const isRunning = job?.status === "running";

  const handleRun = useCallback(async (
    stages: StageName[],
    stageOptions: Record<string, Record<string, string>> = {}
  ) => {
    setError(null);
    try {
      const { job_id } = await processImages(Array.from(selected), stages, stageOptions);
      setJobId(job_id);
      setView("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  }, [selected]);

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100 overflow-hidden">

      {/* Sidebar */}
      <aside className="w-56 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
        {/* Branding */}
        <div className="px-4 py-5 border-b border-gray-800">
          <h1 className="text-sm font-bold text-white leading-tight">Inscription Digitisation</h1>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          <p className="text-xs font-semibold text-gray-600 uppercase tracking-wider px-3 mb-2">Views</p>
          <NavItem active={view === "gallery"} onClick={() => setView("gallery")}>
            <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth={1.75} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
            </svg>
            Gallery
          </NavItem>
          <NavItem active={view === "results"} onClick={() => setView("results")} disabled={!jobId}>
            <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth={1.75} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Results
            {isRunning && (
              <span className="ml-auto w-2 h-2 bg-indigo-400 rounded-full animate-pulse" />
            )}
          </NavItem>
        </nav>

        {/* Selection summary */}
        {selected.size > 0 && (
          <div className="px-4 py-4 border-t border-gray-800">
            <p className="text-xs text-gray-400">
              <span className="text-indigo-400 font-semibold">{selected.size}</span> image{selected.size > 1 ? "s" : ""} selected
            </p>
          </div>
        )}
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">

        {/* Top bar */}
        <header className="h-14 border-b border-gray-800 px-6 flex items-center justify-between flex-shrink-0">
          <h2 className="text-sm font-semibold text-gray-200">
            {view === "gallery" ? "Image Gallery" : "Processing Results"}
          </h2>
          {job && (
            <span className={`text-xs px-2.5 py-1 rounded-full border font-medium ${
              job.status === "running"
                ? "bg-indigo-900/50 text-indigo-400 border-indigo-700"
                : job.status === "done"
                ? "bg-emerald-900/50 text-emerald-400 border-emerald-700"
                : "bg-red-900/50 text-red-400 border-red-700"
            }`}>
              {job.status === "running" ? `Processing ${job.completed}/${job.total}…` : job.status}
            </span>
          )}
        </header>

        {/* Error banner */}
        {error && (
          <div className="mx-6 mt-4 px-4 py-3 bg-red-950 border border-red-800 rounded-lg text-sm text-red-300 flex items-center justify-between flex-shrink-0">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="text-red-500 hover:text-red-300 ml-4">✕</button>
          </div>
        )}

        {/* Scrollable content */}
        <main className="flex-1 overflow-y-auto">
          {view === "gallery" && (
            <div className="p-6 space-y-4">
              {selected.size > 0 && (
                <StagePanel
                  selectedCount={selected.size}
                  onRun={handleRun}
                  isRunning={isRunning}
                />
              )}
              <ImageGrid selected={selected} onSelectionChange={setSelected} />
            </div>
          )}

          {view === "results" && job && (
            <div className="p-6">
              <ResultViewer job={job} />
            </div>
          )}

          {view === "results" && !job && (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
              <span className="text-4xl opacity-20">📊</span>
              <p className="text-gray-500 text-sm">No results yet.</p>
              <button
                onClick={() => setView("gallery")}
                className="text-indigo-400 hover:text-indigo-300 text-sm"
              >
                Go to Gallery →
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
