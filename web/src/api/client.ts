import type { ImageMeta, Job } from "../types";

export async function listImages(): Promise<ImageMeta[]> {
  const res = await fetch("/api/images");
  if (!res.ok) throw new Error(`Failed to load images: ${res.status}`);
  return res.json();
}

export async function processImages(
  imageIds: string[],
  stages: string[]
): Promise<{ job_id: string }> {
  const res = await fetch("/api/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_ids: imageIds, stages }),
  });
  if (!res.ok) throw new Error(`Failed to start job: ${res.status}`);
  return res.json();
}

export async function getJob(jobId: string): Promise<Job> {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Failed to get job: ${res.status}`);
  return res.json();
}
