"""Utility to ruin historical inference
Client for historical run cloud functions
i.e. python scripts/historical_run.py --stage test eodag --date-start 2022-01-01 --date-end 2022-01-02 --geometry test/test_cerulean_cloud/fixtures/search_geom.geojson
"""
from datetime import date

import click
import geojson
import httpx


def handler_historical_run(date_start, date_end, geometry, url):
    """makes a post request to the cloud function historical run"""
    with geometry as src:
        fc = geojson.load(src)

    payload = {
        "start": date_start.strftime("%Y-%m-%d"),
        "end": date_end.strftime("%Y-%m-%d"),
        "geometry": dict(fc),
    }

    res = httpx.post(url, json=payload, timeout=None)
    return res


@click.group()
@click.option(
    "--stage", default="staging", type=click.Choice(["staging", "test", "production"])
)
@click.pass_context
def cli(ctx, stage):
    """Command line tool to add tasks to Cloud Task queue, to run inference on"""
    URLS = dict(
        staging="https://europe-west1-cerulean-338116.cloudfunctions.net/cerulean-cloud-staging-cloud-function-historical-run",
        test="https://europe-west1-cerulean-338116.cloudfunctions.net/cerulean-cloud-test-cloud-function-historical-run",
        production="",
    )
    ctx.ensure_object(dict)
    ctx.obj["URL"] = URLS[stage]
    ctx.obj["STAGE"] = stage


@click.command()
@click.option(
    "--date-start", type=click.DateTime(formats=["%Y-%m-%d"]), default=str(date.today())
)
@click.option(
    "--date-end", type=click.DateTime(formats=["%Y-%m-%d"]), default=str(date.today())
)
@click.option("--geometry", type=click.File(mode="r"))
@click.pass_context
def eodag(ctx, date_start, date_end, geometry):
    """Use start and end date to add Sentinel-1 scenes to Cloud Task queue"""
    click.echo(f"Start: {date_start}, End: {date_end} ")
    res = handler_historical_run(date_start, date_end, geometry, url=ctx.obj["URL"])
    click.echo(f"{res}")


cli.add_command(eodag)

if __name__ == "__main__":
    cli()
