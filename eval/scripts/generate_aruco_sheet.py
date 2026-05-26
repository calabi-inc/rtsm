#!/usr/bin/env python3
"""
Generate printable ArUco markers for RTSM ground-truth collection.

Dictionary: DICT_4X4_50 (matches rtsm/evaluation/scenes/aruco.py)
Physical size: 5 cm per marker (matches the eval plan)

Outputs:
  eval/markers/marker_NN.png         individual markers, 5 cm @ 300 DPI
  eval/markers/printable_sheet.pdf   multi-page US Letter PDF, print at 100%

Usage:
  python eval/scripts/generate_aruco_sheet.py                # markers 0..14
  python eval/scripts/generate_aruco_sheet.py --count 20     # markers 0..19
  python eval/scripts/generate_aruco_sheet.py --size-cm 8    # larger markers
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DPI = 300
CM_PER_INCH = 2.54
LETTER_W_IN = 8.5
LETTER_H_IN = 11.0
MARGIN_CM = 1.5
LABEL_BAND_CM = 1.0
GAP_CM = 0.8

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKERS_DIR = REPO_ROOT / "eval" / "markers"


def cm_to_px(cm: float) -> int:
    return int(round(cm / CM_PER_INCH * DPI))


def generate_marker(marker_id: int, size_px: int) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    return img


def save_individual(marker_id: int, size_cm: float) -> Path:
    size_px = cm_to_px(size_cm)
    marker = generate_marker(marker_id, size_px)
    out = MARKERS_DIR / f"marker_{marker_id:02d}.png"
    cv2.imwrite(str(out), marker)
    return out


def build_pages(marker_ids: list[int], size_cm: float) -> list[np.ndarray]:
    """Return a list of US Letter pages (2550x3300 @ 300 DPI), markers laid out per page."""
    marker_px = cm_to_px(size_cm)
    label_px = cm_to_px(LABEL_BAND_CM)
    gap_px = cm_to_px(GAP_CM)
    margin_px = cm_to_px(MARGIN_CM)

    cell_w = marker_px
    cell_h = marker_px + label_px

    page_w = cm_to_px(LETTER_W_IN * CM_PER_INCH)
    page_h = cm_to_px(LETTER_H_IN * CM_PER_INCH)

    usable_w = page_w - 2 * margin_px
    cols = max(1, int((usable_w + gap_px) // (cell_w + gap_px)))

    usable_h = page_h - 2 * margin_px
    rows_per_page = max(1, int((usable_h + gap_px) // (cell_h + gap_px)))
    per_page = cols * rows_per_page

    pages: list[np.ndarray] = []
    for page_start in range(0, len(marker_ids), per_page):
        page = np.full((page_h, page_w), 255, dtype=np.uint8)
        chunk = marker_ids[page_start : page_start + per_page]
        for idx, mid in enumerate(chunk):
            r = idx // cols
            c = idx % cols
            x0 = margin_px + c * (cell_w + gap_px)
            y0 = margin_px + r * (cell_h + gap_px)

            marker_img = generate_marker(mid, marker_px)
            page[y0 : y0 + marker_px, x0 : x0 + marker_px] = marker_img

            label = f"ID {mid:02d}  ({size_cm:.1f} cm)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.9
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            tx = x0 + (marker_px - tw) // 2
            ty = y0 + marker_px + (label_px + th) // 2
            cv2.putText(page, label, (tx, ty), font, scale, 0, thickness, cv2.LINE_AA)
        pages.append(page)
    return pages


def save_pdf(pages: list[np.ndarray], out_path: Path) -> None:
    """Write pages as a single multi-page PDF at the correct physical size."""
    pil_pages = [Image.fromarray(p, mode="L") for p in pages]
    first, rest = pil_pages[0], pil_pages[1:]
    first.save(
        out_path,
        format="PDF",
        save_all=True,
        append_images=rest,
        resolution=float(DPI),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=15, help="Number of markers to generate (IDs 0..count-1). Max 50.")
    ap.add_argument("--size-cm", type=float, default=5.0, help="Physical size of each marker in cm.")
    args = ap.parse_args()

    if not (1 <= args.count <= 50):
        ap.error("--count must be between 1 and 50 (DICT_4X4_50 has 50 markers)")
    if args.size_cm <= 0:
        ap.error("--size-cm must be positive")

    MARKERS_DIR.mkdir(parents=True, exist_ok=True)

    marker_ids = list(range(args.count))
    for mid in marker_ids:
        out = save_individual(mid, args.size_cm)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")

    pages = build_pages(marker_ids, args.size_cm)
    pdf_path = MARKERS_DIR / "printable_sheet.pdf"
    save_pdf(pages, pdf_path)
    page_h, page_w = pages[0].shape
    print(
        f"\n  wrote {pdf_path.relative_to(REPO_ROOT)}  "
        f"({len(pages)} page(s), {page_w}x{page_h} px each, {DPI} DPI = "
        f"{LETTER_W_IN}\" x {LETTER_H_IN}\" US Letter)"
    )

    stale_png = MARKERS_DIR / "printable_sheet.png"
    if stale_png.exists():
        stale_png.unlink()

    print(f"\nPrint at 100% scale (NO fit-to-page). Measure one marker — should be {args.size_cm:.1f} cm.")


if __name__ == "__main__":
    main()
