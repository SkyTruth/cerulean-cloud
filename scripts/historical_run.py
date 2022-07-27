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
    dag.set_preferred_provider("peps")

    default_search_criteria = {
        "productType": "S2_MSI_L1C",
        "start": "2021-03-01",
        "end": "2021-03-31",
        "geom": {"lonmin": 1, "latmin": 43, "lonmax": 2, "latmax": 44},
    }

    products_first_page, estimated_total_number = dag.search(**default_search_criteria)
    print(
        f"Got a hand on {len(products_first_page)} products and an estimated total number of {estimated_total_number} products available."
    )


cli.add_command(eodag)

if __name__ == "__main__":
    cli()
