#!/usr/bin/env python3
"""
create_maxlen_sessions.py
─────────────────────────────────────────────────────────────────────────────
Tạo các thư mục filter/maxlen{N}/ và filter/fullen/ bên trong unified dir.

Input  : base_dir có đủ 7 file chuẩn (sessions_train/val/test.txt,
         item2cat.json, cat2items.json, cat_parent.json, meta.json)
Output : {base_dir}/filter/fullen/       ← copy nguyên, không lọc
         {base_dir}/filter/maxlen{N}/    ← chỉ giữ session có len ≤ N

Usage:
    python preprocessing/create_maxlen_sessions.py \
        --base_dir ~/data/retailrocket_unified \
        --maxlens 3 4 5 6

Notes:
  - FILTER mode: session có độ dài > maxlen bị BỎ HOÀN TOÀN (không truncate)
  - Chỉ giữ session có min_len <= len <= maxlen
  - fullen = copy nguyên 7 file từ base (không lọc theo length)
  - item2cat.json / cat2items.json / cat_parent.json được copy nguyên
  - meta.json được ghi lại với n_train/val/test cập nhật
  - cold_items = [] (sẽ được tạo bởi cold_start_split.py)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def process_session_file(src: Path, dst: Path, maxlen: int, min_len: int = 2):
    """
    Đọc file session, FILTER (không truncate), ghi ra dst.
    Chỉ giữ session có min_len <= len <= maxlen.
    Trả về (n_kept, n_dropped).
    """
    kept = dropped = 0
    with open(src) as f_in, open(dst, 'w') as f_out:
        for line in f_in:
            items = line.strip().split()
            if not items:
                continue
            n = len(items)
            if n < min_len or n > maxlen:
                dropped += 1
                continue
            f_out.write(' '.join(items) + '\n')
            kept += 1
    return kept, dropped


def copy_static_files(src_dir: Path, dst_dir: Path):
    """Copy item2cat, cat2items, cat_parent (và item2tax nếu có)."""
    static = ['item2cat.json', 'cat2items.json', 'cat_parent.json', 'item2tax.json']
    for fname in static:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)
            print(f"  ✓ Copied {fname}")


def write_meta(meta_orig: dict, out_dir: Path, variant_name: str,
               maxlen: int | None, counts: dict, min_len: int):
    """Ghi meta.json mới với thông tin cập nhật."""
    new_meta = dict(meta_orig)
    new_meta.update({
        'dataset':     variant_name,
        'maxlen':      maxlen,           # None nếu là fullen
        'filter_mode': 'filter',         # luôn là filter (không truncate)
        'min_len':     min_len,
        'n_train':     counts['train'][0],
        'n_val':       counts['val'][0],
        'n_test':      counts['test'][0],
        'cold_items':  [],               # placeholder — sẽ điền bởi cold_start_split.py
    })
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(new_meta, f, indent=2)
    print(f"  meta.json: n_train={new_meta['n_train']:,}, "
          f"n_val={new_meta['n_val']:,}, n_test={new_meta['n_test']:,}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Tạo filter/maxlen{N}/ và filter/fullen/ bên trong unified dir'
    )
    parser.add_argument('--base_dir', required=True,
                        help='Thư mục chứa 7 file chuẩn (sessions_train/val/test.txt + json). '
                             'Thường là {dataset}_unified/cold_20/ trên server.')
    parser.add_argument('--output_root', default=None,
                        help='Thư mục gốc để tạo filter/. '
                             'Mặc định: {base_dir}/filter/. '
                             'Dùng khi muốn tách source và output, ví dụ: '
                             '--base_dir unified/cold_20 --output_root unified/filter')
    parser.add_argument('--maxlens', nargs='+', type=int, default=[3, 4, 5, 6],
                        help='Danh sách maxlen cần tạo (mặc định: 3 4 5 6)')
    parser.add_argument('--min_len', type=int, default=2,
                        help='Session ngắn hơn min_len sẽ bị bỏ (mặc định: 2)')
    parser.add_argument('--no_fullen', action='store_true',
                        help='Bỏ qua việc tạo fullen/')
    args = parser.parse_args()

    base = Path(args.base_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"base_dir không tồn tại: {base}")

    # Kiểm tra 7 file bắt buộc
    required = ['sessions_train.txt', 'sessions_val.txt', 'sessions_test.txt',
                'item2cat.json', 'cat2items.json', 'cat_parent.json', 'meta.json']
    for fname in required:
        if not (base / fname).exists():
            raise FileNotFoundError(f"Thiếu {fname} trong {base}")

    with open(base / 'meta.json') as f:
        meta_orig = json.load(f)

    # Output root: dùng --output_root nếu có, không thì base_dir/filter/
    if args.output_root:
        filter_root = Path(args.output_root).expanduser().resolve()
    else:
        filter_root = base / 'filter'
    filter_root.mkdir(parents=True, exist_ok=True)

    # Lấy dataset stem (bỏ _unified suffix)
    stem = base.name
    if stem.endswith('_unified'):
        stem = stem[:-len('_unified')]

    # ── fullen variant ─────────────────────────────────────────────────────────
    if not args.no_fullen:
        variant = 'fullen'
        out_dir = filter_root / variant
        out_dir.mkdir(exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"[fullen]  →  {out_dir}")
        print(f"  (copy nguyên từ base — không lọc theo length)")
        print(f"{'─'*60}")

        copy_static_files(base, out_dir)

        counts = {}
        for split in ('train', 'val', 'test'):
            src = base / f'sessions_{split}.txt'
            dst = out_dir / f'sessions_{split}.txt'
            if not src.exists():
                print(f"  ⚠  sessions_{split}.txt không tìm thấy — bỏ qua")
                counts[split] = (0, 0)
                continue
            shutil.copy2(src, dst)
            # Đếm số session
            n = sum(1 for line in open(dst) if line.strip())
            counts[split] = (n, 0)
            print(f"  sessions_{split}: {n:,} sessions (copy nguyên)")

        write_meta(meta_orig, out_dir, f'{stem}_fullen', maxlen=None,
                   counts=counts, min_len=args.min_len)

    # ── maxlen variants ────────────────────────────────────────────────────────
    for maxlen in sorted(args.maxlens):
        variant = f'maxlen{maxlen}'
        out_dir = filter_root / variant
        out_dir.mkdir(exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"[{variant}]  →  {out_dir}")
        print(f"  FILTER: chỉ giữ session có len ∈ [{args.min_len}, {maxlen}]")
        print(f"{'─'*60}")

        copy_static_files(base, out_dir)

        counts = {}
        for split in ('train', 'val', 'test'):
            src = base / f'sessions_{split}.txt'
            dst = out_dir / f'sessions_{split}.txt'
            if not src.exists():
                print(f"  ⚠  sessions_{split}.txt không tìm thấy — bỏ qua")
                counts[split] = (0, 0)
                continue
            kept, dropped = process_session_file(src, dst, maxlen, args.min_len)
            counts[split] = (kept, dropped)
            total = kept + dropped
            pct = 100 * kept / total if total > 0 else 0
            print(f"  sessions_{split}: kept={kept:,} ({pct:.1f}%), dropped={dropped:,}")

        write_meta(meta_orig, out_dir, f'{stem}_{variant}', maxlen=maxlen,
                   counts=counts, min_len=args.min_len)

    # ── summary ───────────────────────────────────────────────────────────────
    variants = ([] if args.no_fullen else ['fullen']) + [f'maxlen{n}' for n in sorted(args.maxlens)]
    print(f"\n{'═'*60}")
    print(f"Hoàn thành! Cấu trúc đã tạo:")
    print(f"  {base.name}/filter/")
    for v in variants:
        print(f"    ├── {v}/")
    print(f"\nMode: FILTER (session > maxlen bị bỏ, không truncate)")
    print(f"\nBước tiếp theo — chạy cold_start_split.py cho từng variant:")
    for v in variants:
        print(f"  python preprocessing/cold_start_split.py "
              f"--data_dir {filter_root}/{v} --ratios 10 20 30")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
