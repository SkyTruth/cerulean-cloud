"""Utility to ruin historical inference"""
from datetime import date

import click
import geojson
from eodag import EODataAccessGateway, setup_logging
from shapely.geometry import MultiPolygon, shape

setup_logging(2)


@click.group()
def cli():
    """Command line tool to add tasks to Cloud Task queue, to run inference on"""
    pass


@click.command()
@click.option(
    "--date-start", type=click.DateTime(formats=["%Y-%m-%d"]), default=str(date.today())
)
@click.option(
    "--date-end", type=click.DateTime(formats=["%Y-%m-%d"]), default=str(date.today())
)
@click.option("--geometry", type=click.File(mode="r"))
@click.option("--scihub-username", envvar="SCIHUB_USERNAME")
@click.option("--scihub-password", envvar="SCIHUB_PASSWORD")
def eodag(date_start, date_end, geometry, scihub_username, scihub_password):
    """Use start and end date to add Sentinel-1 scenes to Cloud Task queue"""
    click.echo(f"Start: {date_start}, End: {date_end} ")

    dag = EODataAccessGateway()
    dag.update_providers_config(
        f"""
    scihub:
        api:
            credentials:
                username: "{scihub_username}"
                password: "{scihub_password}"
    """
    )
    dag.set_preferred_provider("scihub")

    with geometry as src:
        fc = geojson.load(src)

    overall_geom = MultiPolygon([shape(f.geometry) for f in fc.features])

    default_search_criteria = {
        "productType": "S1_SAR_GRD",
        "polarization": "VV",
        "start": date_start.strftime("%Y-%m-%d"),
        "end": date_end.strftime("%Y-%m-%d"),
        "geom": overall_geom.wkt,
    }

    search_results = dag.search_all(**default_search_criteria)
    print(f"Got a hand on {len(search_results)} products.")


cli.add_command(eodag)

if __name__ == "__main__":
    cli()
