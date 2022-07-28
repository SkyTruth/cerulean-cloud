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
def eodag(date_start, date_end):
    """Use start and end date to add Sentinel-1 scenes to Cloud Task queue"""
    click.echo(f"Start: {date_start}, End: {date_end} ")

    dag = EODataAccessGateway()

    default_search_criteria = {
        "productType": "S1_SAR_GRD",
        "polarization": "VV",
        "start": "2021-03-01",
        "end": "2021-03-31",
        "geom": {"lonmin": 1, "latmin": 43, "lonmax": 2, "latmax": 44},
    }

    search_results, total_count = dag.search(**default_search_criteria)
    print(
        f"Got a hand on {len(search_results)} products and an estimated total number of {total_count} products available."
    )


cli.add_command(eodag)

if __name__ == "__main__":
    cli()
