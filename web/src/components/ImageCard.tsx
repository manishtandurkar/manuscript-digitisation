import type { ImageMeta } from "../types";

interface Props {
  image: ImageMeta;
  selected: boolean;
  onToggle: (id: string) => void;
}

export default function ImageCard({ image, selected, onToggle }: Props) {
  return (
    <div
      onClick={() => onToggle(image.id)}
      className={`relative cursor-pointer rounded-lg overflow-hidden border-2 transition-all ${
        selected
          ? "border-indigo-500 shadow-lg shadow-indigo-500/30"
          : "border-gray-700 hover:border-gray-500"
      }`}
    >
      <img
        src={image.thumbnail_url || image.url}
        alt={image.filename}
        className="w-full h-36 object-cover bg-gray-800"
        loading="lazy"
        decoding="async"
        fetchPriority="low"
      />
      <div className="px-2 py-1 bg-gray-900">
        <p className="text-[11px] font-medium text-indigo-300 truncate">{image.language}</p>
        <p className="text-xs text-gray-300 truncate">{image.filename}</p>
      </div>
      {selected && (
        <div className="absolute top-2 right-2 w-5 h-5 bg-indigo-500 rounded-full flex items-center justify-center">
          <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 00-1.414 0L8 12.586 4.707 9.293a1 1 0 00-1.414 1.414l4 4a1 1 0 001.414 0l8-8a1 1 0 000-1.414z" clipRule="evenodd" />
          </svg>
        </div>
      )}
    </div>
  );
}
