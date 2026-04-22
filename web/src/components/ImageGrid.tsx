import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { useImages } from "../hooks/useImages";
import ImageCard from "./ImageCard";

interface Props {
  selected: Set<string>;
  onSelectionChange: (selected: Set<string>) => void;
}

export default function ImageGrid({ selected, onSelectionChange }: Props) {
  const { data: images, isLoading, error } = useImages();
  const [search, setSearch] = useState("");
  const [language, setLanguage] = useState("All");
  const [visibleCount, setVisibleCount] = useState(300);
  const deferredSearch = useDeferredValue(search);

  const allImages = images ?? [];

  const languages = useMemo(() => {
    const counts = new Map<string, number>();
    for (const img of allImages) {
      counts.set(img.language, (counts.get(img.language) ?? 0) + 1);
    }
    return Array.from(counts.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([name, count]) => ({ name, count }));
  }, [allImages]);

  const filtered = useMemo(
    () => {
      const query = deferredSearch.toLowerCase();
      return allImages.filter((img) => {
        const matchesLanguage = language === "All" || img.language === language;
        const matchesSearch =
          img.filename.toLowerCase().includes(query) ||
          img.language.toLowerCase().includes(query) ||
          img.collection.toLowerCase().includes(query);
        return matchesLanguage && matchesSearch;
      });
    },
    [allImages, deferredSearch, language]
  );

  useEffect(() => {
    setVisibleCount(300);
  }, [deferredSearch, language]);

  const visibleImages = useMemo(
    () => filtered.slice(0, visibleCount),
    [filtered, visibleCount]
  );

  const groupedImages = useMemo(() => {
    const groups = new Map<string, typeof visibleImages>();
    for (const img of visibleImages) {
      const group = groups.get(img.language) ?? [];
      group.push(img);
      groups.set(img.language, group);
    }
    return Array.from(groups.entries()).sort(([a], [b]) => a.localeCompare(b));
  }, [visibleImages]);

  if (isLoading) {
    return <p className="text-gray-400 text-center py-20">Loading images…</p>;
  }
  if (error || !images) {
    return <p className="text-red-400 text-center py-20">Failed to load images.</p>;
  }

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
          placeholder="Search by image or language…"
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

      <div className="flex flex-wrap gap-2 mb-5" aria-label="Language filters">
        <button
          onClick={() => setLanguage("All")}
          className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
            language === "All"
              ? "bg-indigo-600 text-white border-indigo-500"
              : "bg-gray-900 text-gray-400 border-gray-700 hover:text-gray-200 hover:border-gray-500"
          }`}
        >
          All <span className="opacity-70">{allImages.length}</span>
        </button>
        {languages.map((item) => (
          <button
            key={item.name}
            onClick={() => setLanguage(item.name)}
            className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
              language === item.name
                ? "bg-indigo-600 text-white border-indigo-500"
                : "bg-gray-900 text-gray-400 border-gray-700 hover:text-gray-200 hover:border-gray-500"
            }`}
          >
            {item.name} <span className="opacity-70">{item.count}</span>
          </button>
        ))}
      </div>

      <div className="space-y-7">
        {groupedImages.map(([groupLanguage, groupImages]) => (
          <section key={groupLanguage}>
            <div className="mb-3 flex items-center gap-3">
              <h3 className="text-sm font-semibold text-gray-200">{groupLanguage}</h3>
              <span className="text-xs text-gray-500">
                {groupImages.length} shown
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {groupImages.map((img) => (
                <ImageCard
                  key={img.id}
                  image={img}
                  selected={selected.has(img.id)}
                  onToggle={toggle}
                />
              ))}
            </div>
          </section>
        ))}
      </div>

      {visibleImages.length < filtered.length && (
        <div className="flex justify-center mt-5">
          <button
            onClick={() => setVisibleCount((count) => count + 300)}
            className="text-sm px-4 py-2 rounded-lg border border-gray-700 bg-gray-900 text-gray-300 hover:text-white hover:border-gray-500"
          >
            Load more ({filtered.length - visibleImages.length} remaining)
          </button>
        </div>
      )}

      {filtered.length === 0 && (
        <p className="text-gray-500 text-center py-12">No images match "{search}".</p>
      )}
    </div>
  );
}
