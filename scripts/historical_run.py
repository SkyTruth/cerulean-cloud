"""Discover scenes or submit historical processing locally or through the service."""

import asyncio
import csv
import json
import os
import shlex
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from datetime import date, datetime, timezone

import click
import httpx
import shapely
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

# Support the existing `python scripts/historical_run.py` entry point too.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cerulean_cloud.database_schema import HistoricalRun, Trigger
from cerulean_cloud.cloud_function_historical_run.inventory import (
    filter_sentinel1_scene_ids,
    list_sentinel1_scene_ids,
    region_from_geojson,
)


def post_historical_run(payload, url, api_key):
    """Use the authenticated historical-run endpoint and surface HTTP failures."""
    response = httpx.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=1860,
    )
    response.raise_for_status()
    return response.json()


@click.group()
@click.option("--stage", default="staging", help="Pulumi stack name.")
@click.option(
    "--url", envvar="HISTORICAL_RUN_URL", help="Override the deployed endpoint URL."
)
@click.option("--api-key", envvar="CERULEAN_API_KEY")
@click.option(
    "--local", is_flag=True, help="Run locally and submit Cloud Tasks directly."
)
@click.option(
    "--env-file",
    type=click.Path(exists=True, dir_okay=False),
    help="Local mode: load environment variables, including STACK_SECRETS, from this file.",
)
@click.pass_context
def cli(ctx, stage, url, api_key, local, env_file):
    """List scenes or queue inference locally (--local) or through historical-run."""
    if env_file:
        if not local:
            raise click.UsageError("--env-file requires --local")
        from dotenv import load_dotenv

        load_dotenv(env_file)
        api_key = api_key or os.getenv("CERULEAN_API_KEY")
    ctx.obj = {"url": url, "api_key": api_key, "stage": stage, "local": local}


def configure_local(stage, api_key, *, list_only):
    """Use the notebook's stack secrets or explicit service environment variables."""
    if os.getenv("STACK_SECRETS"):
        from cerulean_cloud.cloud_function_historical_run.local import (
            resolve_orchestrator_queue,
        )

        stack_key = {
            "test": "TEST",
            "staging": "STAGING",
            "prod20240903": "PRODUCTION",
        }.get(stage, stage)
        try:
            config = json.loads(os.environ["STACK_SECRETS"])[stack_key]
            os.environ["DB_URL"] = config["DB_URL"]
            os.environ.setdefault("GCPPROJECT", "cerulean-338116")
            os.environ.setdefault("GCPREGION", "europe-west1")
            if not list_only:
                os.environ.update(
                    {
                        "ORCHESTRATOR_URL": config["ORCHESTRATOR_URL"],
                        "API_KEY": config["API_KEY"],
                    }
                )
        except (KeyError, TypeError, ValueError):
            raise click.ClickException(
                f"STACK_SECRETS must contain configuration for {stack_key}"
            ) from None
        if not list_only:
            queue = config.get("HISTORICAL_QUEUE")
            try:
                os.environ["QUEUE"] = queue or resolve_orchestrator_queue(
                    os.environ["GCPPROJECT"], os.environ["GCPREGION"], stage
                )
            except ValueError as exc:
                raise click.ClickException(str(exc)) from None
    db_url = os.getenv("HISTORICAL_DB_URL") or os.getenv("DB_URL")
    if db_url:
        os.environ["DB_URL"] = db_url.replace(
            "postgresql+asyncpg://", "postgresql://", 1
        )
    if api_key:
        os.environ["API_KEY"] = api_key
    required = ["DB_URL"]
    if not list_only:
        required += ["GCPPROJECT", "GCPREGION", "QUEUE", "ORCHESTRATOR_URL", "API_KEY"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise click.ClickException(f"Local execution requires: {', '.join(missing)}")


def history_metadata(payload):
    """Describe the requested selection, redacted command, and local revision."""
    if "scene_ids" in payload:
        try:  # Sentinel-1 IDs hold acquisition start and end times in fields 4 and 5.
            dates = [
                datetime.strptime(time, "%Y%m%dT%H%M%S").date()
                for scene in payload["scene_ids"]
                for time in scene.strip().split("_")[4:6]
            ]
            date_start, date_end = min(dates), max(dates)
        except ValueError:
            raise ValueError("Cannot read dates from scene IDs") from None
    else:
        date_start = date.fromisoformat(payload["start"])
        date_end = date.fromisoformat(payload["end"])
    geometry = None
    if payload.get("geometry") is not None:
        region = region_from_geojson(payload["geometry"])
        geometry = f"SRID=4326;{shapely.multipolygons(shapely.get_parts(region)).wkt}"
    args = list(sys.argv)
    for index, argument in enumerate(args):
        if argument == "--api-key" and index + 1 < len(args):
            args[index + 1] = "[REDACTED]"
        elif argument.startswith("--api-key="):
            args[index] = "--api-key=[REDACTED]"

    def git(*arguments):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(["git", *arguments], cwd=root, capture_output=True)
        return result.stdout.decode().strip() or None

    return {
        "cli_command": shlex.join([sys.executable, *args]),
        "date_start": date_start,
        "date_end": date_end,
        "geometry": geometry,
        "git_hash": git("rev-parse", "HEAD"),
        "git_tag": git("describe", "--tags", "--exact-match", "HEAD"),
    }


async def save_historical_run(record):
    """Insert or update a history row and its trigger in one transaction."""
    url = os.environ["DB_URL"].replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(url)
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            async with session.begin():
                record = await session.merge(record)
        return record
    finally:
        await engine.dispose()


async def run_local(payload):
    """Discover once, then submit rechecked batches under one trigger and history row."""
    from cerulean_cloud.cloud_function_historical_run.main import (
        run_historical,
        validate_request,
    )

    # Keep stdout as JSON for saving; runner progress and confirmed batches go to stderr.
    with redirect_stdout(sys.stderr):
        if payload["action"] == "list":
            return await run_historical(payload)
        try:  # Reject bad requests before a trigger or history row exists.
            validate_request(payload)
            record = HistoricalRun(
                **history_metadata(payload),
                dry_run=payload["dry_run"],
                submission_start_time=datetime.now(timezone.utc),
                trigger_=Trigger(trigger_logs="", trigger_type="HISTORICAL_RUN"),
            )
        except ValueError as exc:
            raise click.UsageError(str(exc)) from None
        # Insert first: a missing table stops the run, and a killed run keeps its row.
        record = await save_historical_run(record)
        selection, batches = {}, []
        try:
            selection = await run_historical({**payload, "action": "list"})
            if "scene_ids" not in payload and payload.get("source", "eodag") == "eodag":
                record.eodag_count = selection["scene_count"]
            scenes = selection["scene_ids"]
            for start in range(0, len(scenes), 100):
                result = await run_historical(
                    {
                        "scene_ids": scenes[start : start + 100],
                        "dry_run": payload["dry_run"],
                        "manual_trigger": payload["manual_trigger"],
                    },
                    trigger_id=record.trigger,
                )
                batches.append(result)
                click.echo(
                    f"Batch {len(batches)} confirmed: {json.dumps(result)}", err=True
                )
        except BaseException as exc:
            record.submission_error = repr(exc)
            raise
        finally:
            # A failed batch may have accepted tasks, so its total is unknown.
            record.submission_end_time = datetime.now(timezone.utc)
            record.submission_success = record.submission_error is None
            record.queued_task_count = (
                sum(batch["queued_count"] for batch in batches)
                if record.submission_success
                else None
            )
            record.trigger_.scene_count = selection.get("scene_count")
            record.trigger_.filtered_scene_count = (
                sum(batch["filtered_scene_count"] for batch in batches)
                if record.submission_success
                else None
            )
            try:
                await save_historical_run(record)
            except Exception as save_error:
                if record.submission_error is None:
                    raise
                click.echo(f"Could not finalize historical_run: {save_error}", err=True)
    return {
        "trigger_id": record.trigger,
        "historical_run_id": record.id,
        "scene_count": selection["scene_count"],
        "filtered_scene_count": sum(batch["filtered_scene_count"] for batch in batches),
        "already_processed_scene_count": selection["already_processed_scene_count"]
        + sum(batch["already_processed_scene_count"] for batch in batches),
        "queued_count": sum(batch["queued_count"] for batch in batches),
        "dry_run": payload["dry_run"],
        "batches": batches,
    }


def execution_options(command):
    """Options shared by every selector; local enqueue requires an explicit dry-run flag."""
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
    if ctx.obj["local"]:
        if not list_only and dry_run is None:
            raise click.UsageError(
                "Local submission requires --dry-run or --no-dry-run"
            )
        configure_local(ctx.obj["stage"], ctx.obj["api_key"], list_only=list_only)
        result = asyncio.run(run_local(payload))
        click.echo(json.dumps(result, indent=2))
        return
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
    if options["list_only"] and not ctx.obj["local"]:
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
