"""Utility to ruin historical inference"""
from datetime import date

import click
from eodag import EODataAccessGateway, setup_logging

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

    default_search_criteria = {
        "productType": "S1_SAR_GRD",
        "polarization": "VV",
        "start": date_start.strftime("%Y-%m-%d"),
        "end": date_end.strftime("%Y-%m-%d"),
        "geom": {"lonmin": 1, "latmin": 43, "lonmax": 2, "latmax": 44},
    }

    search_results = dag.search_all(**default_search_criteria)
    print(f"Got a hand on {len(search_results)} products.")


cli.add_command(eodag)

if __name__ == "__main__":
    cli()
