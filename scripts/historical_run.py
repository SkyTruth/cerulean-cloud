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
from click.core import ParameterSource
import httpx
import shapely
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

# Support the existing `python scripts/historical_run.py` entry point too.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cerulean_cloud.database_schema import HistoricalRun, Trigger
from cerulean_cloud.cloud_function_historical_run.queues import historical_queue_name
from cerulean_cloud.cloud_function_historical_run.relevance import region_from_geojson


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


class HistoricalRunGroup(click.Group):
    """Route search options to EODAG when no selector is supplied."""

    def resolve_command(self, ctx, args):
        if args and args[0].startswith("-"):
            return "eodag", self.get_command(ctx, "eodag"), args
        return super().resolve_command(ctx, args)


@click.group(
    cls=HistoricalRunGroup,
    context_settings={"ignore_unknown_options": True},
    subcommand_metavar="[COMMAND] [ARGS]...",
)
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
@click.option(
    "--db-host", help="Local mode: override the database host, e.g. 127.0.0.1."
)
@click.option(
    "--db-port",
    type=click.IntRange(1, 65535),
    help="Local mode: override the database port.",
)
@click.pass_context
def cli(ctx, stage, url, api_key, local, env_file, db_host, db_port):
    """List scenes or queue inference locally (--local) or through historical-run.

    EODAG discovery is the default; the eodag selector is optional. Use
    eodag --help for search options, or scenes to supply explicit IDs or a CSV.
    """
    if not local and (db_host is not None or db_port is not None):
        raise click.UsageError("--db-host and --db-port require --local")
    if env_file:
        if not local:
            raise click.UsageError("--env-file requires --local")
        from dotenv import load_dotenv

        load_dotenv(env_file)
        api_key = api_key or os.getenv("CERULEAN_API_KEY")
    ctx.obj = {
        "url": url,
        "api_key": api_key,
        "stage": stage,
        "local": local,
        "db_host": db_host,
        "db_port": db_port,
    }


def configure_local(stage, api_key, *, list_only, db_host=None, db_port=None):
    """Use the notebook's stack secrets or explicit service environment variables."""
    if not list_only:
        try:
            historical_queue_name(stage)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from None
        os.environ["HISTORICAL_RUN_STAGE"] = stage
    if os.getenv("STACK_SECRETS"):
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
    db_url = os.getenv("HISTORICAL_DB_URL") or os.getenv("DB_URL")
    if db_url:
        overrides = {
            key: value
            for key, value in (("host", db_host), ("port", db_port))
            if value is not None
        }
        if overrides:
            try:
                # Query-string host/port settings otherwise override URL components.
                url = make_url(db_url).difference_update_query(overrides)
                db_url = url.set(**overrides).render_as_string(hide_password=False)
            except (ArgumentError, ValueError):
                raise click.ClickException("Invalid local database URL") from None
        os.environ["DB_URL"] = db_url.replace(
            "postgresql+asyncpg://", "postgresql://", 1
        )
    if api_key:
        os.environ["API_KEY"] = api_key
    required = ["DB_URL"]
    if not list_only:
        required += ["GCPPROJECT", "GCPREGION", "ORCHESTRATOR_URL", "API_KEY"]
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
    """Discover once, then submit batches under one trigger and history row."""
    from cerulean_cloud.cloud_function_historical_run.main import (
        enqueue_batches,
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
            batches = await enqueue_batches(
                selection["scene_ids"],
                record.trigger,
                queue_name=historical_queue_name(os.environ["HISTORICAL_RUN_STAGE"]),
                dry_run=payload["dry_run"],
                manual_trigger=payload["manual_trigger"],
                recheck_history=not payload.get("overwrite", False),
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
    }


def execution_options(command):
    """Options shared by every selector; processing is dry-run by default."""
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
    command = click.option(
        "--overwrite",
        is_flag=True,
        help="Include previously processed scenes without checking their run history.",
    )(command)
    command = click.option(
        "--no-dry-run", is_flag=True, help="Alias for --dry-run=false."
    )(command)
    return click.option(
        "--dry-run",
        type=click.Choice(["true", "false"], case_sensitive=False),
        is_flag=False,
        flag_value="true",
        default="true",
        show_default=True,
        help="Skip inference. Use --dry-run=false to enable processing.",
    )(command)


def submit(
    ctx, payload, dry_run, manual_trigger, list_only, overwrite, no_dry_run=False
):
    dry_run = dry_run == "true"
    if no_dry_run:
        if ctx.get_parameter_source("dry_run") == ParameterSource.COMMANDLINE:
            raise click.UsageError("Use either --dry-run or --no-dry-run, not both")
        dry_run = False
    payload["action"] = "list" if list_only else "enqueue"
    payload["manual_trigger"] = manual_trigger
    payload["dry_run"] = dry_run
    if overwrite:
        payload["overwrite"] = True
    if ctx.obj["local"]:
        if not (list_only and overwrite):
            configure_local(
                ctx.obj["stage"],
                ctx.obj["api_key"],
                list_only=list_only,
                db_host=ctx.obj["db_host"],
                db_port=ctx.obj["db_port"],
            )
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
@click.option(
    "--geometry",
    type=click.File(mode="r"),
    help="GeoJSON FeatureCollection file. Omit for a global search.",
)
@click.option(
    "--mode",
    "modes",
    type=click.Choice(["EW", "IW", "SM"], case_sensitive=False),
    multiple=True,
    help="Acquisition mode; repeat for multiple modes. Omit to search all modes.",
)
@execution_options
@click.pass_context
def eodag(ctx, date_start, date_end, geometry, modes, **options):
    """Search globally or within a region and retain scenes intersecting the ocean."""
    payload = {
        "start": date_start.strftime("%Y-%m-%d"),
        "end": date_end.strftime("%Y-%m-%d"),
    }
    if geometry is not None:
        with geometry as src:
            payload["geometry"] = json.load(src)
    if modes:
        payload["modes"] = list(dict.fromkeys(modes))
    submit(ctx, payload, **options)


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
