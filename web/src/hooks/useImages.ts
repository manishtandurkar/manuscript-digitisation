import { useQuery } from "@tanstack/react-query";
import { listImages } from "../api/client";

export function useImages() {
  return useQuery({
    queryKey: ["images"],
    queryFn: listImages,
    staleTime: Infinity,
  });
}
