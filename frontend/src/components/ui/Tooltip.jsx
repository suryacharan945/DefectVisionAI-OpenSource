import React, { useState } from 'react';
import { HelpCircle } from 'lucide-react';

export default function Tooltip({ text, children, position = 'top' }) {
  const [visible, setVisible] = useState(false);

  return (
    <div
      className="relative inline-flex items-center"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocus={() => setVisible(true)}
      onBlur={() => setVisible(false)}
    >
      {children || (
        <button
          type="button"
          aria-label="Info explanation"
          className="text-slate-400 hover:text-cyan-400 transition-colors ml-1 focus:outline-none"
        >
          <HelpCircle className="w-3.5 h-3.5" />
        </button>
      )}

      {visible && (
        <div
          role="tooltip"
          className={`absolute z-50 px-3 py-2 text-xs font-normal text-slate-200 bg-slate-900/95 border border-slate-700/80 rounded-lg shadow-xl backdrop-blur-md w-max max-w-xs transition-opacity duration-150 pointer-events-none ${
            position === 'top'
              ? 'bottom-full mb-2 left-1/2 -translate-x-1/2'
              : position === 'bottom'
              ? 'top-full mt-2 left-1/2 -translate-x-1/2'
              : position === 'left'
              ? 'right-full mr-2 top-1/2 -translate-y-1/2'
              : 'left-full ml-2 top-1/2 -translate-y-1/2'
          }`}
        >
          {text}
        </div>
      )}
    </div>
  );
}
