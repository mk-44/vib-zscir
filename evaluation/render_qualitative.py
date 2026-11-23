# """
# evaluation/render_qualitative.py

# Render the single-column qualitative figure for the paper:
#   - top: reference image + the relative caption (the query)
#   - middle row: baseline (deterministic) top-5
#   - bottom row: VIB (ours) top-5
# GT target(s) are boxed green wherever they appear in a top-5; where the target is
# off-screen (baseline buries it) the row title states its rank honestly.

# Reads the candidate JSON produced by find_qualitative.py (top-5 ids, gt ids,
# reference, caption, ranks) -- no model needed -- and resolves ids to image paths
# via the CIRCO val dataset. Sized for a single column (~3.3in wide), thumbnails
# small but legible. Outputs PDF (vector, for LaTeX) + PNG (preview).

# RUN:
#   python evaluation/render_qualitative.py --query_id 194
# """

# import argparse
# import json
# import textwrap
# from pathlib import Path

# import matplotlib
# matplotlib.use('Agg')
# # Embed TrueType (Type 42) fonts, never Type 3 -- CVPR/WACV reject Type 3 fonts.
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['font.family'] = 'sans-serif'
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import PIL.Image

# from evaluation.stickiness_compare import load_circo_split

# GT_GREEN = '#1a9850'
# REF_RED = '#d73027'


# def load_thumb(path, size=256):
#     im = PIL.Image.open(path).convert('RGB')
#     w, h = im.size
#     s = min(w, h)
#     im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
#     return im.resize((size, size))


# def box_axes(ax, color, lw=3.0):
#     for sp in ax.spines.values():
#         sp.set_visible(True)
#         sp.set_edgecolor(color)
#         sp.set_linewidth(lw)


# def strip_axes(ax):
#     ax.set_xticks([]); ax.set_yticks([])
#     for sp in ax.spines.values():
#         sp.set_visible(False)


# def draw_row(subfig, title, ids, gt_set, path_for, topn=5):
#     subfig.suptitle(title, fontsize=6.5, fontweight='normal', x=0.5, y=0.99)
#     axes = subfig.subplots(1, topn)
#     for rank, (ax, img_id) in enumerate(zip(axes, ids[:topn]), start=1):
#         try:
#             ax.imshow(load_thumb(path_for(img_id)))
#         except Exception:
#             ax.text(0.5, 0.5, 'missing', ha='center', va='center', fontsize=5.5)
#         strip_axes(ax)
#         if img_id in gt_set:                       # the target -> green box + check
#             box_axes(ax, GT_GREEN, lw=2.2)
#             ax.set_title(f'#{rank} \u2713', fontsize=6, color=GT_GREEN,
#                          fontweight='bold', pad=1.2)
#         else:
#             ax.set_title(f'#{rank}', fontsize=5.5, color='#666666', pad=1.2)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--query_id', required=True, help='query_id to render (e.g. 194)')
#     ap.add_argument('--candidates', type=Path,
#                     default=Path('/notebooks/experiment/codes/metrics/qualitative_candidates.json'))
#     ap.add_argument('--circo_data_path', type=Path,
#                     default=Path('/notebooks/experiment/eval_data/circo'))
#     ap.add_argument('--out', type=Path,
#                     default=Path('/notebooks/experiment/codes/metrics/qualitative_fig'))
#     args = ap.parse_args()

#     cands = json.load(open(args.candidates))
#     qid = args.query_id
#     entry = next((c for c in cands if str(c['query_id']) == str(qid)), None)
#     if entry is None:
#         raise SystemExit(f"query_id {qid} not in {args.candidates}. "
#                          f"available: {[c['query_id'] for c in cands][:20]}")

#     ds, _ = load_circo_split(args.circo_data_path, (lambda x: x), 'val')
#     def path_for(img_id):
#         return ds.img_paths[ds.img_ids_indexes_map[str(int(img_id))]]

#     gt_set = set(int(g) for g in entry['gt_ids'])
#     base_ids = [int(i) for i in entry['baseline_top5']]
#     vib_ids = [int(i) for i in entry['vib_top5']]
#     b_rank, v_rank = entry['baseline_gt_rank'], entry['vib_gt_rank']

#     # honest row titles: where the target sits under each model
#     b_in = any(i in gt_set for i in base_ids)
#     base_title = (f'Deterministic baseline: target rank {b_rank}' if b_in
#                   else f'Deterministic baseline: target not in top-5 (rank {b_rank})')
#     vib_title = f'VIB mapper (ours): target rank {v_rank}'

#     # ---- single-column layout (~3.3in wide) ----
#     fig = plt.figure(figsize=(3.35, 3.05), dpi=300)
#     top, midrow, botrow = fig.subfigures(3, 1, height_ratios=[1.25, 1.0, 1.0])

#     # top: reference + caption
#     a_ref, a_cap = top.subplots(1, 2, width_ratios=[1.0, 2.15])
#     try:
#         a_ref.imshow(load_thumb(path_for(entry['reference_img_id'])))
#     except Exception:
#         a_ref.text(0.5, 0.5, 'ref missing', ha='center', va='center', fontsize=6)
#     strip_axes(a_ref)
#     box_axes(a_ref, REF_RED, lw=2.2)
#     a_ref.set_title('Reference', fontsize=6.5, color=REF_RED, fontweight='normal', pad=2)

#     a_cap.axis('off')
#     wrapped = textwrap.fill(entry['caption'], width=30)
#     a_cap.text(0.0, 0.5, f'\u201c{wrapped}\u201d', fontsize=6.8, va='center', ha='left',
#                style='italic', wrap=True)

#     draw_row(midrow, base_title, base_ids, gt_set, path_for)
#     draw_row(botrow, vib_title, vib_ids, gt_set, path_for)

#     args.out.parent.mkdir(parents=True, exist_ok=True)
#     pdf, png = args.out.with_suffix('.pdf'), args.out.with_suffix('.png')
#     fig.savefig(pdf, bbox_inches='tight', pad_inches=0.02)
#     fig.savefig(png, bbox_inches='tight', pad_inches=0.02)
#     print(f'query {qid}: "{entry["caption"]}"')
#     print(f'  baseline target rank {b_rank}  ->  VIB target rank {v_rank}')
#     print(f'  GT boxed green where present; reference boxed red.')
#     print(f'saved -> {pdf}\n        {png}')


# if __name__ == '__main__':
#     main()

"""
evaluation/render_qualitative.py

Render the single-column qualitative figure for the paper:
  - top: reference image + the relative caption (the query)
  - middle row: baseline (deterministic) top-5
  - bottom row: VIB (ours) top-5
GT target(s) are boxed green wherever they appear in a top-5; where the target is
off-screen (baseline buries it) the row title states its rank honestly.

Reads the candidate JSON produced by find_qualitative.py (top-5 ids, gt ids,
reference, caption, ranks) -- no model needed -- and resolves ids to image paths
via the CIRCO val dataset. Sized for a single column (~3.3in wide). This COMPACT
variant halves the figure height for tight page budgets: thumbnails are small,
row titles are short (rank detail lives in the LaTeX caption), fonts are reduced.
Outputs PDF (vector, for LaTeX) + PNG (preview).

RUN:
  python evaluation/render_qualitative.py --query_id 60
  python evaluation/render_qualitative.py --query_id 60 --figheight 1.55   # push smaller
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
# Embed TrueType (Type 42) fonts, never Type 3 -- CVPR/WACV reject Type 3 fonts.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import PIL.Image

from evaluation.stickiness_compare import load_circo_split

GT_GREEN = '#1a9850'
REF_RED = '#d73027'

# ---- compact sizing knobs (tweak if labels overlap after rendering) ----
TITLE_FS = 4.5     # row titles
RANK_FS = 4.0      # per-thumbnail #rank labels
CAP_FS = 5.0       # the query caption
BOX_LW = 1.6       # green/red box line width (thinner at small size)


def load_thumb(path, size=224):
    im = PIL.Image.open(path).convert('RGB')
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return im.resize((size, size))


def box_axes(ax, color, lw=BOX_LW):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)


def strip_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_row(subfig, ids, gt_set, path_for, topn=5):
    axes = subfig.subplots(1, topn)
    for rank, (ax, img_id) in enumerate(zip(axes, ids[:topn]), start=1):
        try:
            ax.imshow(load_thumb(path_for(img_id)))
        except Exception:
            ax.text(0.5, 0.5, 'missing', ha='center', va='center', fontsize=RANK_FS)
        strip_axes(ax)
        if img_id in gt_set:                       # the target -> green box + check
            box_axes(ax, GT_GREEN, lw=BOX_LW + 0.4)
            ax.set_title(f'#{rank} \u2713', fontsize=RANK_FS, color=GT_GREEN,
                         fontweight='bold', pad=0.8)
        else:
            ax.set_title(f'#{rank}', fontsize=RANK_FS, color='#666666', pad=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query_id', required=True, help='query_id to render (e.g. 60)')
    ap.add_argument('--candidates', type=Path,
                    default=Path('/notebooks/experiment/codes/metrics/qualitative_candidates.json'))
    ap.add_argument('--circo_data_path', type=Path,
                    default=Path('/notebooks/experiment/eval_data/circo'))
    ap.add_argument('--out', type=Path,
                    default=Path('/notebooks/experiment/codes/metrics/qualitative_fig'))
    ap.add_argument('--figwidth', type=float, default=3.35, help='inches (single column)')
    ap.add_argument('--figheight', type=float, default=1.70,
                    help='inches; ~half the original 3.05 for tight page budgets')
    args = ap.parse_args()

    cands = json.load(open(args.candidates))
    qid = args.query_id
    entry = next((c for c in cands if str(c['query_id']) == str(qid)), None)
    if entry is None:
        raise SystemExit(f"query_id {qid} not in {args.candidates}. "
                         f"available: {[c['query_id'] for c in cands][:20]}")

    ds, _ = load_circo_split(args.circo_data_path, (lambda x: x), 'val')
    def path_for(img_id):
        return ds.img_paths[ds.img_ids_indexes_map[str(int(img_id))]]

    gt_set = set(int(g) for g in entry['gt_ids'])
    base_ids = [int(i) for i in entry['baseline_top5']]
    vib_ids = [int(i) for i in entry['vib_top5']]
    b_rank, v_rank = entry['baseline_gt_rank'], entry['vib_gt_rank']

    # honest row titles, shortened so they fit at the compact height
    # (full rank detail also stated in the LaTeX caption)
    base_corner = f'Baseline  rank {b_rank}'
    vib_corner = f'VIB (ours)  rank {v_rank}'

    # ---- single-column layout, compact height ----
    fig = plt.figure(figsize=(args.figwidth, args.figheight), dpi=300)
    top, midrow, botrow = fig.subfigures(3, 1, height_ratios=[1.05, 1.0, 1.0])

    # top band: reference | caption | legend  (3 cells, legend is real drawn content)
    a_ref, a_cap, a_leg = top.subplots(1, 3, width_ratios=[1.0, 1.9, 1.05])
    try:
        a_ref.imshow(load_thumb(path_for(entry['reference_img_id'])))
    except Exception:
        a_ref.text(0.5, 0.5, 'ref missing', ha='center', va='center', fontsize=CAP_FS)
    strip_axes(a_ref)
    box_axes(a_ref, REF_RED, lw=BOX_LW + 0.4)
    a_ref.set_title('Reference', fontsize=TITLE_FS, color=REF_RED,
                    fontweight='normal', pad=1.5)

    a_cap.axis('off')
    wrapped = textwrap.fill(entry['caption'], width=26)
    a_cap.text(0.0, 0.5, f'\u201c{wrapped}\u201d', fontsize=CAP_FS, va='center',
               ha='left', style='italic', wrap=True)

    # legend cell: two color-coded rank lines, top-right, matplotlib-style
    a_leg.axis('off')
    LEG_FS = 4.8
    a_leg.text(1.0, 0.62, f'Baseline: target rank {b_rank}', fontsize=LEG_FS,
               color='#666666', ha='right', va='center', fontweight='bold',
               transform=a_leg.transAxes)
    a_leg.text(1.0, 0.40, f'VIB (ours): target rank {v_rank}', fontsize=LEG_FS,
               color=GT_GREEN, ha='right', va='center', fontweight='bold',
               transform=a_leg.transAxes)

    draw_row(midrow, base_ids, gt_set, path_for)
    draw_row(botrow, vib_ids, gt_set, path_for)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = args.out.with_suffix('.pdf'), args.out.with_suffix('.png')
    fig.savefig(pdf, bbox_inches='tight', pad_inches=0.015)
    fig.savefig(png, bbox_inches='tight', pad_inches=0.015)
    print(f'query {qid}: "{entry["caption"]}"')
    print(f'  baseline target rank {b_rank}  ->  VIB target rank {v_rank}')
    print(f'  size: {args.figwidth} x {args.figheight} in  (was 3.35 x 3.05)')
    print(f'  GT boxed green where present; reference boxed red.')
    print(f'saved -> {pdf}\n        {png}')


if __name__ == '__main__':
    main()