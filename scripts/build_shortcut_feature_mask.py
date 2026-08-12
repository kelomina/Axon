#!/usr/bin/env python3
"""Emit a feature mask that zeroes the packing/entropy shortcut columns.

Round 0 arm C. Error attribution on the 1,197 samples every model gets wrong
showed the champion learned "packed / high entropy => malicious" -- packed_any
carried 66x lift on the false positives and very_high_entropy 26.5x -- so this
mask removes the columns that shortcut rides on and leaves everything else.

Which columns are worth removing is decided by one rule: a feature is only worth
masking if the byte branch cannot recompute it. That rule is not theoretical --
`has_signature` was masked-by-accident (a dead constant-0 column) for the whole
lineage and the model still used the signature signal at 7.2x lift on false
negatives, because the certificate directory sits in the optional header, inside
the 8192 bytes the byte branch reads. Measured on 200,145 cached samples:

    signal                      byte-branch recovery
    certificate directory       exact (benign 60.01% / malicious 9.20%)
    packer section names        partial (7.74% of malicious, 91.8% precision)
    full-file section entropy   weak (AUC 0.608)

Only full-file section entropy is genuinely out of the byte branch's reach: the
high-entropy packed sections lie beyond byte 8192, so the visible window stays
low-entropy whether or not the file is packed. Those columns are therefore the
ones this mask drops, together with the packer keyword counts.

Window entropy (`stat_*_entropy_normalized`) is deliberately KEPT by default:
the byte branch trivially recomputes it from the bytes it already sees, so
masking it would cost real signal and remove nothing. Pass --strict to drop it
anyway as a second arm.

Masks are keep-lists: listed indices pass through, everything else is zeroed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvd_features.schema_names import (  # noqa: E402
    fixed_v2_feature_names,
    fixed_v3_feature_names,
    stat_feature_names,
)

SCHEMA_BUILDERS = {
    "fixed_v2": fixed_v2_feature_names,
    "fixed_v3": fixed_v3_feature_names,
}

# Substrings identifying the shortcut columns, matched against the schema names
# so the mask survives any future index shift.
PE_DROP_PATTERNS = ("section_entropy_", "section_high_entropy_ratio", "packer_keyword_hits_")
STAT_DROP_PATTERNS_STRICT = ("entropy_normalized",)


def select_indices(names: list[str], drop_patterns: tuple[str, ...]) -> tuple[list[int], list[str]]:
    dropped = [
        index for index, name in enumerate(names)
        if any(pattern in name for pattern in drop_patterns)
    ]
    kept = [index for index in range(len(names)) if index not in set(dropped)]
    return kept, [names[index] for index in dropped]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", choices=sorted(SCHEMA_BUILDERS), default="fixed_v3")
    parser.add_argument("--pe-feature-dim", type=int, default=256)
    parser.add_argument("--section-slots", type=int, default=32)
    parser.add_argument("--stat-segment-count", type=int, default=3)
    parser.add_argument("--stat-chunk-count", type=int, default=10)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also drop the window-entropy stat columns the byte branch can recompute",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    pe_names = SCHEMA_BUILDERS[args.schema](section_slots=args.section_slots)
    stat_names = stat_feature_names(args.stat_segment_count, args.stat_chunk_count)
    pe_search_dim = len(pe_names)
    stat_feature_dim = len(stat_names)

    if args.pe_feature_dim < pe_search_dim:
        sys.exit(f"pe_feature_dim {args.pe_feature_dim} < used dim {pe_search_dim}")

    kept_pe, dropped_pe_names = select_indices(pe_names, PE_DROP_PATTERNS)
    stat_patterns = STAT_DROP_PATTERNS_STRICT if args.strict else ()
    kept_stat, dropped_stat_names = select_indices(stat_names, stat_patterns)

    individual = [False] * (pe_search_dim + stat_feature_dim)
    for index in kept_pe:
        individual[index] = True
    for index in kept_stat:
        individual[pe_search_dim + index] = True

    variant = "strict" if args.strict else "pe_only"
    payload = {
        "version": 1,
        "type": "axon_feature_mask",
        "source_report": "round0 error attribution: 1197 universally-wrong samples",
        "checkpoint": None,
        "note": (
            f"Round 0 arm C ({variant}): zero the packing/entropy shortcut columns of the "
            f"{args.schema} PE schema. Dropped columns were chosen because the byte branch "
            "cannot recompute them (full-file section entropy measures AUC 0.608 from the "
            "visible 8192-byte window). Signature and packer-name signals are deliberately "
            "NOT masked -- they are readable straight out of the header bytes, so masking "
            "them removes nothing."
        ),
        "mask_spec": {
            "pe_feature_dim": args.pe_feature_dim,
            "pe_search_dim": pe_search_dim,
            "ignored_pe_dim": args.pe_feature_dim - pe_search_dim,
            "stat_feature_dim": stat_feature_dim,
            "search_dim": pe_search_dim + stat_feature_dim,
            "pe_schema_version": args.schema,
        },
        "kept_total": len(kept_pe) + len(kept_stat),
        "kept_pe": len(kept_pe),
        "kept_stat": len(kept_stat),
        "selected_pe_indices": kept_pe,
        "selected_stat_indices": kept_stat,
        "individual": individual,
        "dropped_pe_features": dropped_pe_names,
        "dropped_stat_features": dropped_stat_names,
    }

    output = args.output or (
        PROJECT_ROOT / "config" / "feature_masks" / f"round0_shortcut_{variant}_{args.schema}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"schema        : {args.schema} (pe used {pe_search_dim}, stat {stat_feature_dim})")
    print(f"kept          : {payload['kept_total']} of {pe_search_dim + stat_feature_dim}")
    print(f"dropped PE    : {len(dropped_pe_names)}")
    for name in dropped_pe_names:
        print(f"    {name}")
    print(f"dropped stat  : {len(dropped_stat_names)}")
    for name in dropped_stat_names:
        print(f"    {name}")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
