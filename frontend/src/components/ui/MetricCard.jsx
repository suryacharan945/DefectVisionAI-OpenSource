import React, { useEffect, useState } from 'react';
import Tooltip from './Tooltip';

export default function MetricCard({
  title,
  value,
  unit = '',
  description,
  tooltip,
  icon: Icon,
  accentColor = '#8b5cf6',
  badge,
  isPercent = false,
}) {
  const [displayVal, setDisplayVal] = useState(typeof value === 'number' ? 0 : value);

  useEffect(() => {
    if (typeof value !== 'number') {
      setDisplayVal(value);
      return;
    }

    let start = 0;
    const end = value;
    const duration = 750; // ms
    const stepTime = 25;
    const totalSteps = duration / stepTime;
    const increment = (end - start) / totalSteps;

    let current = 0;
    const timer = setInterval(() => {
      current += increment;
      if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
        setDisplayVal(end);
        clearInterval(timer);
      } else {
        setDisplayVal(parseFloat(current.toFixed(isPercent || end % 1 !== 0 ? 1 : 0)));
      }
    }, stepTime);

    return () => clearInterval(timer);
  }, [value, isPercent]);

  return (
    <div className="relative group overflow-hidden rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 backdrop-blur-md transition-all duration-300 hover:border-violet-500/40 hover:shadow-[0_0_20px_rgba(139,92,246,0.15)]">
      {/* Top ambient color glow */}
      <div
        className="absolute -top-10 -right-10 w-24 h-24 rounded-full opacity-20 blur-xl pointer-events-none transition-opacity group-hover:opacity-35"
        style={{ backgroundColor: accentColor }}
      />

      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
          {Icon && <Icon className="w-4 h-4 text-slate-300" />}
          <span>{title}</span>
          {tooltip && <Tooltip text={tooltip} />}
        </div>
        {badge && <div>{badge}</div>}
      </div>

      <div className="flex items-baseline gap-1 mt-1">
        <span className="font-mono text-3xl font-bold tracking-tight text-white">
          {typeof displayVal === 'number'
            ? isPercent
              ? displayVal.toFixed(1)
              : displayVal.toLocaleString()
            : displayVal}
        </span>
        {unit && <span className="font-mono text-sm text-slate-400 font-medium">{unit}</span>}
      </div>

      {description && <p className="text-xs text-slate-400 mt-2 font-normal leading-relaxed">{description}</p>}
    </div>
  );
}
