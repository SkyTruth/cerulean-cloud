"""Discover scenes or submit a tracked historical processing batch."""

import csv
import json
import subprocess
import sys
from pathlib import Path
from datetime import date

import click
import httpx

# Support the existing `python scripts/historical_run.py` entry point too.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cerulean_cloud.cloud_function_historical_run.inventory import (
    filter_sentinel1_scene_ids,
    list_sentinel1_scene_ids,
)


def post_historical_run(payload, url, api_key):
    """Use the authenticated historical-run endpoint and surface HTTP failures."""
    response = httpx.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


@click.group()
@click.option("--stage", default="staging", help="Pulumi stack name.")
@click.option(
    "--url", envvar="HISTORICAL_RUN_URL", help="Override the deployed endpoint URL."
)
@click.option("--api-key", envvar="CERULEAN_API_KEY")
@click.pass_context
def cli(ctx, stage, url, api_key):
    """List scenes or queue inference through the selected stack's historical-run service."""
    ctx.obj = {"url": url, "api_key": api_key, "stage": stage}


def execution_options(command):
    """Options shared by every selector; omitted dry-run uses the stack setting."""
    command = click.option(
        "--list-only",
        is_flag=True,
        help="Return IDs without creating a batch or tasks.",
    )(command)
    command = click.option(
        "--manual-trigger",
        is_flag=True,
        help="Use run-specific downstream source-association task names.",
    )(command)
    return click.option("--dry-run/--no-dry-run", default=None)(command)


def submit(ctx, payload, dry_run, manual_trigger, list_only):
    payload["action"] = "list" if list_only else "enqueue"
    payload["manual_trigger"] = manual_trigger
    if dry_run is not None:
        payload["dry_run"] = dry_run
    if not ctx.obj["api_key"]:
        raise click.ClickException("Set CERULEAN_API_KEY or supply --api-key")
    url = ctx.obj["url"]
    if url is None:
        url = subprocess.check_output(
            [
                "pulumi",
                "stack",
                "output",
                "historical_run_url",
                "--stack",
                ctx.obj["stage"],
            ],
            text=True,
        ).strip()
    result = post_historical_run(payload, url, ctx.obj["api_key"])
    click.echo(json.dumps(result, indent=2))


@cli.command()
@click.option(
    "--date-start", type=click.DateTime(formats=["%Y-%m-%d"]), default=str(date.today())
)
@click.option(
    "--date-end", type=click.DateTime(formats=["%Y-%m-%d"]), default=str(date.today())
)
@click.option("--geometry", type=click.File(mode="r"), required=True)
@execution_options
@click.pass_context
def eodag(ctx, date_start, date_end, geometry, **options):
    """Search a geographic area and retain scenes intersecting the ocean."""
    with geometry as src:
        geometry_json = json.load(src)
    submit(
        ctx,
        {
            "start": date_start.strftime("%Y-%m-%d"),
            "end": date_end.strftime("%Y-%m-%d"),
            "geometry": geometry_json,
        },
        **options,
    )


@cli.command()
@click.option("--date-start", type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option("--date-end", type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option(
    "--mode", multiple=True, help="Repeat to select modes; omitted includes all."
)
@click.option(
    "--polarization",
    multiple=True,
    help="Repeat to select polarizations; omitted includes all.",
)
@click.option(
    "--geometry", type=click.File(mode="r"), help="GeoJSON region FeatureCollection."
)
@click.option(
    "--relevancy/--no-relevancy",
    default=True,
    help="Apply ocean, VV, and resolution checks (default: enabled). Region filtering is independent.",
)
@execution_options
@click.pass_context
def s3(ctx, date_start, date_end, mode, polarization, geometry, relevancy, **options):
    """Discover archive scenes and apply scene-relevancy before listing or queueing."""
    if date_end < date_start:
        raise click.UsageError("date-end must be on or after date-start")
    geometry_json = json.load(geometry) if geometry is not None else None
    if options["list_only"]:
        ids = list_sentinel1_scene_ids(
            date_start.date(),
            date_end.date(),
            modes=mode or None,
            polarizations=polarization or None,
        )
        selected = filter_sentinel1_scene_ids(
            ids, geometry=geometry_json, apply_relevancy=relevancy
        )
        click.echo(
            json.dumps(
                {
                    "scene_count": len(ids),
                    "filtered_scene_count": len(selected),
                    "scene_ids": selected,
                },
                indent=2,
            )
        )
        return
    submit(
        ctx,
        {
            "source": "s3",
            "start": date_start.strftime("%Y-%m-%d"),
            "end": date_end.strftime("%Y-%m-%d"),
            "modes": list(mode) or None,
            "polarizations": list(polarization) or None,
            **({"geometry": geometry_json} if geometry is not None else {}),
            "apply_relevancy": relevancy,
        },
        **options,
    )


@cli.command()
@click.option("--scene-id", multiple=True, help="Repeat for each scene ID.")
@click.option(
    "--csv", "csv_file", type=click.File(mode="r"), help="CSV with a scene_id column."
)
@execution_options
@click.pass_context
def scenes(ctx, scene_id, csv_file, **options):
    """Submit explicit scene IDs without another catalog search."""
    ids = list(scene_id)
    if csv_file is not None:
        with csv_file as src:
            reader = csv.DictReader(src)
            if not reader.fieldnames or "scene_id" not in reader.fieldnames:
                raise click.ClickException("CSV must contain a scene_id column")
            ids.extend(row["scene_id"] for row in reader)
    if not ids:
        raise click.UsageError("Supply --scene-id or --csv")
    submit(ctx, {"scene_ids": ids}, **options)


if __name__ == "__main__":
    cli()
