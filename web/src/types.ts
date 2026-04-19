export interface ImageMeta {
  id: string;
  filename: string;
  url: string;
  thumbnail_url: string;
}

export type StageStatus = "pending" | "running" | "done" | "failed" | "skipped";

export interface StageResult {
  status: StageStatus;
  url?: string;
  text?: string;
  error?: string;
  reason?: string;
}

export type StageName = "preprocess" | "enhance" | "binarise" | "ocr" | "translate";

export interface Job {
  job_id: string;
  status: "running" | "done" | "failed";
  total: number;
  completed: number;
  results: Record<string, Record<StageName, StageResult>>;
}
