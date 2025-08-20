export default function PrefChips({ prefs }) {
  if (!prefs) return null;
  const entries = Object.entries(prefs).filter(([k]) => !k.startsWith("__"));
  if (entries.length === 0) return null;
  const label = (k) => ({
    interests: "Interests",
    climate: "Climate",
    budget: "Budget",
    season: "Season",
    duration: "Duration",
    companions: "Companions",
    pace: "Pace",
    region: "Region",
    health: "Health",
  }[k] || k);

  const fmt = (v) => Array.isArray(v) ? v.join(", ") : v;

  return (
    <div className="flex flex-wrap gap-2 mb-2">
      {entries.map(([k, v]) => (
        <span key={k} className="text-xs bg-gray-900 text-white px-2 py-1 rounded-full">
          <b>{label(k)}:</b> {fmt(v)}
        </span>
      ))}
    </div>
  );
}