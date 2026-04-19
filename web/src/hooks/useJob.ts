import { useQuery } from "@tanstack/react-query";
import { getJob } from "../api/client";
import type { Job } from "../types";

export function useJob(jobId: string | null) {
  return useQuery<Job>({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId!),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "running" ? 2000 : false;
    },
  });
}
