import { useState } from "react";

interface Props {
  before: string;
  after: string;
  afterLabel: string;
  height?: string;
}

export default function ComparisonSlider({ before, after, afterLabel, height = "h-72" }: Props) {
  const [position, setPosition] = useState(25);

  return (
    <div className={`relative overflow-hidden rounded-xl ${height} select-none bg-gray-800`}>
      {/* Before image — base layer */}
      <img
        src={before}
        alt="Original"
        className="absolute inset-0 w-full h-full object-contain"
        draggable={false}
      />

      {/* After image — clipped from the right */}
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      >
        <img
          src={after}
          alt={afterLabel}
          className="absolute inset-0 w-full h-full object-contain"
          draggable={false}
        />
      </div>

      {/* Divider line */}
      <div
        className="absolute top-0 bottom-0 w-0.5 bg-white/80 shadow-[0_0_8px_rgba(0,0,0,0.8)]"
        style={{ left: `${position}%` }}
      />

      {/* Drag handle */}
      <div
        className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-9 h-9 bg-white rounded-full shadow-lg flex items-center justify-center pointer-events-none z-10"
        style={{ left: `${position}%` }}
      >
        <svg className="w-4 h-4 text-gray-700" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M8 9l-4 3 4 3M16 9l4 3-4 3" />
        </svg>
      </div>

      {/* Invisible range input for interaction */}
      <input
        type="range"
        min={0}
        max={100}
        value={position}
        onChange={(e) => setPosition(Number(e.target.value))}
        className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize z-20"
      />

      {/* Corner labels */}
      <span className="absolute bottom-2 left-2 text-xs bg-black/60 text-white px-2 py-0.5 rounded-md pointer-events-none">
        Original
      </span>
      <span className="absolute bottom-2 right-2 text-xs bg-indigo-600/80 text-white px-2 py-0.5 rounded-md pointer-events-none">
        {afterLabel}
      </span>
    </div>
  );
}
