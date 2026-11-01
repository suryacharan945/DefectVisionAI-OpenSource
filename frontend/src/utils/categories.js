/**
 * 15 MVTec AD Category Metadata
 * Standardized across DefectVision AI for badges, filters, and dropdowns.
 * Colors avoid green, amber, red (which are strictly reserved for severity).
 */

export const CATEGORIES = [
  { id: 'bottle', label: 'Bottle', type: 'Object', color: '#818cf8', desc: 'Glass container bottles' },
  { id: 'cable', label: 'Cable', type: 'Object', color: '#38bdf8', desc: 'Insulated copper cables' },
  { id: 'capsule', label: 'Capsule', type: 'Object', color: '#a78bfa', desc: 'Pharmaceutical capsules' },
  { id: 'carpet', label: 'Carpet', type: 'Texture', color: '#2dd4bf', desc: 'Woven synthetic carpet' },
  { id: 'grid', label: 'Grid', type: 'Texture', color: '#60a5fa', desc: 'Metallic wire grids' },
  { id: 'hazelnut', label: 'Hazelnut', type: 'Object', color: '#c084fc', desc: 'Organic food kernels' },
  { id: 'leather', label: 'Leather', type: 'Texture', color: '#e879f9', desc: 'Tanned surface leather' },
  { id: 'metal_nut', label: 'Metal Nut', type: 'Object', color: '#94a3b8', desc: 'Hexagonal machine nuts' },
  { id: 'pill', label: 'Pill', type: 'Object', color: '#38bdf8', desc: 'Coated medicinal tablets' },
  { id: 'screw', label: 'Screw', type: 'Object', color: '#cbd5e1', desc: 'Threaded fastener screws' },
  { id: 'tile', label: 'Tile', type: 'Texture', color: '#22d3ee', desc: 'Polished floor ceramic tile' },
  { id: 'toothbrush', label: 'Toothbrush', type: 'Object', color: '#818cf8', desc: 'Polymer bristle heads' },
  { id: 'transistor', label: 'Transistor', type: 'Object', color: '#a855f7', desc: 'TO-92 electronic packages' },
  { id: 'wood', label: 'Wood', type: 'Texture', color: '#06b6d4', desc: 'Finished wood panels' },
  { id: 'zipper', label: 'Zipper', type: 'Object', color: '#64748b', desc: 'Interlocking metal zippers' },
];

export const CATEGORY_MAP = Object.fromEntries(
  CATEGORIES.map((c) => [c.id, c])
);

export function getCategoryMeta(id) {
  return CATEGORY_MAP[id] || {
    id: id || 'unknown',
    label: (id || 'Unknown').replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
    type: 'Component',
    color: '#94a3b8',
    desc: 'Industrial part',
  };
}
