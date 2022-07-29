"""Utility to ruin historical inference"""
import json
import os
from datetime import date

import click
import geojson
import urlparse
from eodag import EODataAccessGateway, setup_logging
from google.cloud import tasks_v2
from shapely.geometry import MultiPolygon, shape

setup_logging(2)


def handler_queue(filtered_scenes):
    """handler queue"""
    # Create a client.
    client = tasks_v2.CloudTasksClient()

    project = os.getenv("GCP_PROJECT")
    queue = os.getenv("QUEUE")
    location = os.getenv("GCP_LOCATION")
    url = os.getenv("ORCHESTRATOR_URL")

    # Construct the fully qualified queue name.
    parent = client.queue_path(project, location, queue)

    for scene in filtered_scenes:
        # Construct the request body.
        # TODO: Add orchestrate and POST method instead
        task = {
            "http_request": {  # Specify the type of request.
                "http_method": tasks_v2.HttpMethod.POST,
                "url": urlparse.urljoin(
                    url, "orchestrate"
                ),  # The full url path that the task will be sent to.
            }
        }

        payload = {"sceneid": scene, "trigger": 1, "dry_run": True}
        print(payload)
        # Add the payload to the request.
        if payload is not None:
            if isinstance(payload, dict):
                # Convert dict to JSON string
                payload = json.dumps(payload)
                # specify http content-type to application/json
                task["http_request"]["headers"] = {"Content-type": "application/json"}

            # The API expects a payload of type bytes.
            converted_payload = payload.encode()

            # Add the payload to the request.
            task["http_request"]["body"] = converted_payload

        # Use the client to build and send the task.
        response = client.create_task(request={"parent": parent, "task": task})

        print("Created task {}".format(response.name))


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

    filtered_scenes = []
    for result in search_results:
        print(f"Adding {result}...")
        filtered_scenes.append(result.properties.get("id"))

    handler_queue(filtered_scenes[0])


cli.add_command(eodag)

if __name__ == "__main__":
    cli()
