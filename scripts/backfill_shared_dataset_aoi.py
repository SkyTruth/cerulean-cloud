#!/usr/bin/env python3
"""
Operator wrapper for shared-dataset AOI slick_to_aoi backfills.

The common path only needs the shared-datasets asset slug:

    scripts/backfill_shared_dataset_aoi.py prepare <asset-slug>
    scripts/backfill_shared_dataset_aoi.py inspect <asset-slug>
    scripts/backfill_shared_dataset_aoi.py run <asset-slug>
    scripts/backfill_shared_dataset_aoi.py status <asset-slug>
    scripts/backfill_shared_dataset_aoi.py validate <asset-slug>
    scripts/backfill_shared_dataset_aoi.py finish <asset-slug>

Optional flags exist for the fields that cannot always be inferred safely.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Sequence

from cerulean_cloud.cloud_run_aoi_backfill import service


LOGGER = logging.getLogger("backfill_shared_dataset_aoi")


def inspect_asset(args: argparse.Namespace) -> None:
    result = service.inspect_asset(
        args.asset_slug,
        short_name=args.short_name,
        catalog_source=args.catalog_source,
        version=args.version,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
        long_name=args.long_name,
        ext_id_field=args.ext_id_field,
        display_name_field=args.display_name_field,
        stage_table=args.stage_table,
        source_url=args.source_url,
        citation=args.citation,
    )
    print(f"input\t{result['input']}")
    print(f"asset_slug\t{result['asset_slug']}")
    print(f"short_name\t{result['short_name']}")
    print(f"long_name\t{result['long_name']}")
    print(f"ext_id_field\t{result['ext_id_field']}")
    print(f"display_name_field\t{result['display_name_field']}")
    print(f"stage_table\t{result['stage_table']}")
    print(f"dataset_version\t{result['dataset_version']}")
    print(f"source_url\t{result['source_url']}")
    print(f"cache_path\t{result['cache_path']}")
    print("fields\t" + ",".join(result["fields"]))
    print(f"chunk_bounds\t{result['chunk_plan']['bounds']}")
    print(f"chunk_grid_side\t{result['chunk_plan']['grid_side']}")
    print(f"target_chunk_count\t{result['chunk_plan']['target_chunk_count']}")
    for key in (
        "feature_count",
        "crs",
        "geometry_types",
        "null_or_empty_geometry_rows",
        "invalid_geometry_rows",
        "empty_ext_id_rows",
        "duplicate_ext_id_values",
        "duplicate_ext_id_rows",
    ):
        print(f"{key}\t{result[key]}")


def prepare(args: argparse.Namespace) -> None:
    service.prepare_backfill(
        args.asset_slug,
        db_url=args.db_url,
        short_name=args.short_name,
        catalog_source=args.catalog_source,
        version=args.version,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
        long_name=args.long_name,
        ext_id_field=args.ext_id_field,
        display_name_field=args.display_name_field,
        stage_table=args.stage_table,
        source_url=args.source_url,
        citation=args.citation,
        batch_size=args.batch_size,
    )


def run(args: argparse.Namespace) -> None:
    service.run_backfill(
        args.asset_slug,
        db_url=args.db_url,
        short_name=args.short_name,
        catalog_source=args.catalog_source,
        max_batches=args.max_batches,
        sleep_seconds=args.sleep_seconds,
        lock_timeout=args.lock_timeout,
        statement_timeout=args.statement_timeout,
    )


def validate(args: argparse.Namespace) -> None:
    rows = service.validate_backfill(
        args.asset_slug,
        db_url=args.db_url,
        short_name=args.short_name,
        catalog_source=args.catalog_source,
    )
    for check_name, value in rows:
        print(f"{check_name}\t{value}")


def status(args: argparse.Namespace) -> None:
    resolved_short_name = args.short_name or service.slug_to_short_name(
        service.resolve_asset_slug(args.asset_slug, args.catalog_source)
    )
    rows = service.get_backfill_status(
        args.asset_slug,
        db_url=args.db_url,
        short_name=args.short_name,
        catalog_source=args.catalog_source,
    )
    if not rows:
        print(f"No prepared backfill run for {resolved_short_name}")
        return
    headers = (
        "short_name",
        "status",
        "total_chunks",
        "completed_chunks",
        "pending_chunks",
        "running_chunks",
        "failed_chunks",
        "staged_rows_loaded",
        "candidate_slick_rows",
        "matches",
        "aois_inserted",
        "links_inserted",
        "updated_at",
    )
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(value) for value in row))


def finish(args: argparse.Namespace) -> None:
    service.finish_backfill(
        args.asset_slug,
        db_url=args.db_url,
        short_name=args.short_name,
        catalog_source=args.catalog_source,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url",
        help="Database URL. Defaults to DB_URL.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log debug output.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    add_common_asset_args(inspect_parser)
    add_dataset_args(inspect_parser)
    add_config_args(inspect_parser)
    inspect_parser.set_defaults(func=inspect_asset)

    prepare_parser = subparsers.add_parser("prepare")
    add_common_asset_args(prepare_parser)
    add_dataset_args(prepare_parser)
    add_config_args(prepare_parser)
    prepare_parser.add_argument("--batch-size", type=int, default=5000)
    prepare_parser.set_defaults(func=prepare)

    run_parser = subparsers.add_parser("run")
    add_common_asset_args(run_parser)
    run_parser.add_argument(
        "--max-batches",
        type=int,
        default=service.DEFAULT_RUN_MAX_BATCHES,
    )
    run_parser.add_argument("--sleep-seconds", type=float, default=0.05)
    run_parser.add_argument("--lock-timeout", default="1s")
    run_parser.add_argument(
        "--statement-timeout",
        default=service.DEFAULT_RUN_STATEMENT_TIMEOUT,
    )
    run_parser.set_defaults(func=run)

    validate_parser = subparsers.add_parser("validate")
    add_common_asset_args(validate_parser)
    validate_parser.set_defaults(func=validate)

    status_parser = subparsers.add_parser("status")
    add_common_asset_args(status_parser)
    status_parser.set_defaults(func=status)

    finish_parser = subparsers.add_parser("finish")
    add_common_asset_args(finish_parser)
    finish_parser.set_defaults(func=finish)

    return parser


def add_common_asset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "asset_slug",
        metavar="asset_slug_or_gs_uri",
        help="Shared-datasets asset slug, or exact gs:// URI from the catalog.",
    )
    parser.add_argument(
        "--short-name",
        help="Override derived AOI type short_name. Defaults to upper snake-case slug.",
    )
    parser.add_argument(
        "--catalog-source",
        default=os.getenv("SHARED_DATASETS_CATALOG_SOURCE"),
        help=(
            "Optional catalog CSV path/URL/gs:// URI. "
            "Defaults to shared-datasets GCS catalog."
        ),
    )


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--version", default="latest")
    parser.add_argument(
        "--cache-dir",
        default=service.DEFAULT_CACHE_DIR,
    )
    parser.add_argument("--force-download", action="store_true")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--long-name")
    parser.add_argument("--ext-id-field")
    parser.add_argument("--display-name-field")
    parser.add_argument("--stage-table")
    parser.add_argument("--source-url")
    parser.add_argument("--citation")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        args.func(args)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
