#!/usr/bin/env python3
"""
DefectVision AI — Procedural Synthetic Demo Sample Generator

Generates 15 synthetic, license-clean sample textures for frontend demo mode
and UI preview, completely eliminating third-party dataset specimen crops.
"""

import os
import cv2
import numpy as np

KNOWN_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

def generate_texture(category, size=256):
    """Procedurally synthesizes an abstract category texture."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    h, w = size, size

    if category == "bottle":
        img[:] = (35, 40, 48)
        # Bottle silhouette
        cv2.ellipse(img, (w//2, h//2 + 30), (45, 80), 0, 0, 360, (70, 130, 180), -1)
        cv2.rectangle(img, (w//2 - 18, h//2 - 90), (w//2 + 18, h//2 - 40), (80, 140, 190), -1)
        cv2.rectangle(img, (w//2 - 22, h//2 - 105), (w//2 + 22, h//2 - 90), (100, 160, 210), -1)

    elif category == "cable":
        img[:] = (28, 32, 40)
        # Bundled industrial cables
        cv2.line(img, (20, 40), (236, 216), (220, 60, 60), 10)
        cv2.line(img, (40, 20), (216, 236), (60, 180, 220), 10)
        cv2.line(img, (30, 30), (226, 226), (220, 200, 60), 8)

    elif category == "capsule":
        img[:] = (32, 36, 44)
        # Two-tone pharmaceutical capsule
        cv2.ellipse(img, (w//2, h//2 - 30), (35, 45), 0, 180, 360, (230, 80, 80), -1)
        cv2.rectangle(img, (w//2 - 35, h//2 - 30), (w//2 + 35, h//2), (230, 80, 80), -1)
        cv2.rectangle(img, (w//2 - 35, h//2), (w//2 + 35, h//2 + 30), (230, 230, 230), -1)
        cv2.ellipse(img, (w//2, h//2 + 30), (35, 45), 0, 0, 180, (230, 230, 230), -1)

    elif category == "carpet":
        # Woven fiber pattern
        x = np.linspace(0, 10 * np.pi, w)
        y = np.linspace(0, 10 * np.pi, h)
        xx, yy = np.meshgrid(x, y)
        pattern = (np.sin(xx) * np.cos(yy) * 40 + 100).astype(np.uint8)
        img[:, :, 0] = pattern
        img[:, :, 1] = np.clip(pattern * 0.8, 0, 255).astype(np.uint8)
        img[:, :, 2] = np.clip(pattern * 0.6, 0, 255).astype(np.uint8)

    elif category == "grid":
        img[:] = (25, 30, 38)
        # Precision wire mesh
        for gx in range(20, w, 24):
            cv2.line(img, (gx, 20), (gx, h - 20), (140, 160, 180), 2)
        for gy in range(20, h, 24):
            cv2.line(img, (20, gy), (w - 20, gy), (140, 160, 180), 2)

    elif category == "hazelnut":
        img[:] = (35, 38, 45)
        # Organic nut shape with shell gradient
        cv2.circle(img, (w//2, h//2), 65, (50, 90, 140), -1)
        cv2.circle(img, (w//2 - 10, h//2 - 10), 55, (60, 110, 170), -1)
        cv2.ellipse(img, (w//2, h//2 + 45), (35, 18), 0, 0, 360, (90, 140, 190), -1)

    elif category == "leather":
        # Procedural Voronoi-like pebbled texture
        rng = np.random.RandomState(42)
        noise = rng.randint(70, 105, (h, w), dtype=np.uint8)
        kernel = np.ones((3, 3), np.float32) / 9
        smooth = cv2.filter2D(noise, -1, kernel)
        img[:, :, 0] = np.clip(smooth * 0.7, 0, 255).astype(np.uint8)
        img[:, :, 1] = np.clip(smooth * 0.5, 0, 255).astype(np.uint8)
        img[:, :, 2] = smooth

    elif category == "metal_nut":
        img[:] = (30, 34, 42)
        # Hexagonal metal nut with center bore
        pts = np.array([
            [w//2, h//2 - 75],
            [w//2 + 65, h//2 - 38],
            [w//2 + 65, h//2 + 38],
            [w//2, h//2 + 75],
            [w//2 - 65, h//2 + 38],
            [w//2 - 65, h//2 - 38]
        ], np.int32)
        cv2.fillPoly(img, [pts], (180, 190, 200))
        cv2.circle(img, (w//2, h//2), 35, (30, 34, 42), -1)
        cv2.circle(img, (w//2, h//2), 30, (140, 150, 160), 2)

    elif category == "pill":
        img[:] = (32, 38, 48)
        # Circular tablet with center score line
        cv2.circle(img, (w//2, h//2), 60, (220, 225, 230), -1)
        cv2.circle(img, (w//2, h//2), 58, (200, 205, 210), 2)
        cv2.line(img, (w//2 - 45, h//2), (w//2 + 45, h//2), (170, 175, 180), 2)

    elif category == "screw":
        img[:] = (28, 32, 40)
        # Threaded screw shaft and head
        cv2.rectangle(img, (w//2 - 40, 40), (w//2 + 40, 65), (190, 200, 210), -1)
        cv2.rectangle(img, (w//2 - 15, 65), (w//2 + 15, 210), (160, 170, 180), -1)
        for sy in range(80, 200, 15):
            cv2.line(img, (w//2 - 22, sy), (w//2 + 22, sy + 6), (120, 130, 140), 3)

    elif category == "tile":
        img[:] = (40, 45, 55)
        # Ceramic square tile with glossy highlights
        cv2.rectangle(img, (30, 30), (w - 30, h - 30), (210, 215, 220), -1)
        cv2.rectangle(img, (35, 35), (w - 35, h - 35), (235, 240, 245), 2)
        cv2.line(img, (40, 40), (w - 40, 40), (255, 255, 255), 2)

    elif category == "toothbrush":
        img[:] = (30, 35, 45)
        # Ergonomic handle and bristle cluster
        cv2.rectangle(img, (w//2 - 12, 30), (w//2 + 12, 220), (50, 160, 220), -1)
        cv2.rectangle(img, (w//2 - 25, 35), (w//2 - 12, 100), (240, 240, 240), -1)

    elif category == "transistor":
        img[:] = (28, 30, 38)
        # TO-220 electronic package with 3 leads
        cv2.rectangle(img, (w//2 - 45, 50), (w//2 + 45, 130), (45, 50, 60), -1)
        cv2.rectangle(img, (w//2 - 35, 40), (w//2 + 35, 50), (160, 170, 180), -1)
        cv2.circle(img, (w//2, 45), 6, (45, 50, 60), -1)
        for lx in [w//2 - 25, w//2, w//2 + 25]:
            cv2.line(img, (lx, 130), (lx, 225), (200, 210, 220), 4)

    elif category == "wood":
        # Procedural wood grain
        y = np.linspace(0, 1, h)
        x = np.linspace(0, 1, w)
        xx, yy = np.meshgrid(x, y)
        grain = np.sin(yy * 30 + np.sin(xx * 5) * 8) * 30 + 130
        img[:, :, 0] = np.clip(grain * 0.45, 0, 255).astype(np.uint8)
        img[:, :, 1] = np.clip(grain * 0.70, 0, 255).astype(np.uint8)
        img[:, :, 2] = np.clip(grain * 0.95, 0, 255).astype(np.uint8)

    elif category == "zipper":
        img[:] = (32, 36, 44)
        # Interlocking zipper teeth
        cv2.rectangle(img, (w//2 - 30, 0), (w//2 + 30, h), (70, 75, 85), -1)
        for zy in range(15, h, 18):
            cv2.rectangle(img, (w//2 - 18, zy), (w//2 - 2, zy + 8), (210, 215, 225), -1)
            cv2.rectangle(img, (w//2 + 2, zy + 9), (w//2 + 18, zy + 17), (210, 215, 225), -1)

    # Watermark synthetic demo label
    cv2.putText(img, "SYNTHETIC DEMO SPECIMEN", (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(img, category.upper(), (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return img

def main():
    target_dirs = [
        "frontend/public/outputs/similarity_search_samples",
        "DefectVisionAI/outputs/similarity_search_samples"
    ]
    for d in target_dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Generating synthetic specimens in {d}...")
        for cat in KNOWN_CATEGORIES:
            img = generate_texture(cat)
            out_file = os.path.join(d, f"{cat}_similarity_demo.png")
            cv2.imwrite(out_file, img)
            print(f"  Generated: {out_file} ({os.path.getsize(out_file):,} bytes)")

    print("\nAll 15 synthetic specimens generated successfully.")

if __name__ == "__main__":
    main()
