import { useState } from "react";
import { useImages } from "../hooks/useImages";
import ImageCard from "./ImageCard";

interface Props {
  selected: Set<string>;
  onSelectionChange: (selected: Set<string>) => void;
}

export default function ImageGrid({ selected, onSelectionChange }: Props) {
  const { data: images, isLoading, error } = useImages();
  const [search, setSearch] = useState("");

  if (isLoading) {
    return <p className="text-gray-400 text-center py-20">Loading images…</p>;
  }
  if (error || !images) {
    return <p className="text-red-400 text-center py-20">Failed to load images.</p>;
  }

  const filtered = images.filter((img) =>
    img.filename.toLowerCase().includes(search.toLowerCase())
  );

  function toggle(id: string) {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    onSelectionChange(next);
  }

  function selectAll() {
    onSelectionChange(new Set(filtered.map((img) => img.id)));
  }

  function deselectAll() {
    onSelectionChange(new Set());
  }

  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <input
          type="text"
          placeholder="Search images…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500"
        />
        <button
          onClick={selectAll}
          className="text-sm text-indigo-400 hover:text-indigo-300 px-2"
        >
          Select all
        </button>
        <button
          onClick={deselectAll}
          className="text-sm text-gray-500 hover:text-gray-400 px-2"
        >
          Deselect all
        </button>
        {selected.size > 0 && (
          <span className="text-sm text-indigo-300 font-medium">
            {selected.size} selected
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
        {filtered.map((img) => (
          <ImageCard
            key={img.id}
            image={img}
            selected={selected.has(img.id)}
            onToggle={toggle}
          />
        ))}
      </div>

      {filtered.length === 0 && (
        <p className="text-gray-500 text-center py-12">No images match "{search}".</p>
      )}
    </div>
  );
}
