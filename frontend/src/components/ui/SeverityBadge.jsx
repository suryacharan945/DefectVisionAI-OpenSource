import React from 'react';
import { CheckCircle2, AlertTriangle, AlertOctagon } from 'lucide-react';

/**
 * SeverityBadge Component
 * Implements non-negotiable color guidelines:
 * - Green = Minor / Normal (CheckCircle2)
 * - Amber = Moderate (AlertTriangle)
 * - Red = Critical (AlertOctagon)
 * Colorblind-accessible by pairing color with distinct geometric icon.
 */
export default function SeverityBadge({ level, score, showScore = true, size = 'md' }) {
  const normLevel = (level || 'Normal').toLowerCase();

  let config = {
    label: 'Normal / Minor',
    icon: CheckCircle2,
    bg: 'bg-emerald-500/15',
    border: 'border-emerald-500/40',
    text: 'text-emerald-400',
    glow: 'shadow-[0_0_12px_rgba(16,185,129,0.25)]',
    scoreRange: '< 40 / 100',
  };

  if (normLevel === 'critical') {
    config = {
      label: 'Critical',
      icon: AlertOctagon,
      bg: 'bg-rose-500/15',
      border: 'border-rose-500/40',
      text: 'text-rose-400',
      glow: 'shadow-[0_0_15px_rgba(244,63,94,0.3)]',
      scoreRange: '≥ 70 / 100',
    };
  } else if (normLevel === 'moderate') {
    config = {
      label: 'Moderate',
      icon: AlertTriangle,
      bg: 'bg-amber-500/15',
      border: 'border-amber-500/40',
      text: 'text-amber-400',
      glow: 'shadow-[0_0_12px_rgba(245,158,11,0.25)]',
      scoreRange: '40 – 69 / 100',
    };
  } else if (normLevel === 'minor') {
    config = {
      label: 'Minor',
      icon: CheckCircle2,
      bg: 'bg-emerald-500/15',
      border: 'border-emerald-500/40',
      text: 'text-emerald-400',
      glow: 'shadow-[0_0_12px_rgba(16,185,129,0.25)]',
      scoreRange: '< 40 / 100',
    };
  }

  const Icon = config.icon;
  const isLg = size === 'lg';
  const isSm = size === 'sm';

  return (
    <span
      className={`inline-flex items-center gap-1.5 border rounded-full font-medium transition-all ${
        config.bg
      } ${config.border} ${config.text} ${config.glow} ${
        isLg ? 'px-4 py-1.5 text-base' : isSm ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm'
      }`}
      title={`Severity: ${config.label} (${config.scoreRange})`}
    >
      <Icon className={isLg ? 'w-5 h-5' : isSm ? 'w-3.5 h-3.5' : 'w-4 h-4'} />
      <span className="font-semibold tracking-wide uppercase">{config.label}</span>
      {showScore && score !== undefined && score !== null && (
        <span className="opacity-80 font-mono text-xs ml-0.5">
          ({typeof score === 'number' ? score.toFixed(1) : score})
        </span>
      )}
    </span>
  );
}
