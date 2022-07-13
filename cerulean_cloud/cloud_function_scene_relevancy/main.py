"""cloud function scene relevancy handler
inspired by https://github.com/jonaraphael/ceruleanserver/tree/master/lambda/Machinable
"""

import asyncio
import json
import os

import asyncpg
import shapely.geometry as sh  # https://docs.aws.amazon.com/lambda/latest/dg/python-package.html
from google.cloud import tasks_v2


def load_ocean_poly():
    """load ocean boundary polygon"""
    with open("OceanGeoJSON_lowres.geojson") as f:
        ocean_features = json.load(f)["features"]
    geom = sh.GeometryCollection(
        [sh.shape(feature["geometry"]).buffer(0) for feature in ocean_features]
    )[0]
    return geom


ocean_poly = load_ocean_poly()


async def get_row():
    """get a row"""
    conn = await asyncpg.connect(os.getenv("DB_URL"))
    row = await conn.fetchrow("SELECT * FROM trigger")
    return row


def main(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    row = loop.run_until_complete(get_row())
    print(row)
    request_json = request.get_json()
    if request.args and "message" in request.args:
        return request.args.get("message")
    elif request_json and "message" in request_json:
        return request_json["message"]
    else:
        return "Hello World!"


def lambda_handler(event, context):
    """handle lambda"""
    print("shapely imported!")
    if event.get("Records"):
        sns = event["Records"][0]["Sns"]
        msg = json.loads(sns["Message"])
        scene_poly = sh.polygon.Polygon(msg["footprint"]["coordinates"][0][0])

        is_highdef = "H" == msg["id"][10]
        is_vv = (
            "V" == msg["id"][15]
        )  # we don't want to process any polarization other than vv XXX This is hardcoded in the server, where we look for a vv.grd file
        is_oceanic = scene_poly.intersects(ocean_poly)
        print(is_highdef, is_vv, is_oceanic)
        """
        if is_highdef and is_vv and is_oceanic:
            client = boto3.client("sqs", region_name="eu-central-1")
            response = client.send_message(
                QueueUrl="https://sqs.eu-central-1.amazonaws.com/162277344632/New_Machinable",
                MessageBody=json.dumps(event),
            )

        sns_db_row(sns['MessageId'], sns['Subject'], sns['Timestamp'], msg, is_oceanic)
        TODO: insert SNS into DB
        https://docs.aws.amazon.com/lambda/latest/dg/services-rds-tutorial.html
        """

        return {"statusCode": 200}
    else:
        return {"statusCode": 400}


def sns_db_row(sns_id, sns_sub, sns_ts, msg, is_oceanic):
    # Warning! PostgreSQL hates capital letters, so the keys are different between the SNS and the DB
    """Creates a dictionary that aligns with our SNS DB columns

    Returns:
        dict -- key for each column in our SNS DB, value from this SNS's content
    """
    tbl = "sns"
    row = {
        "sns_messageid": f"'{sns_id}'",  # Primary Key
        "sns_subject": f"'{sns_sub}'",
        "sns_timestamp": f"'{sns_ts}'",
        "grd_id": f"'{msg['id']}'",
        "grd_uuid": f"'{msg['sciHubId']}'",  # Unique Constraint
        "absoluteorbitnumber": f"{msg['absoluteOrbitNumber']}",
        "footprint": f"ST_GeomFromGeoJSON('{json.dumps(msg['footprint'])}')",
        "mode": f"'{msg['mode']}'",
        "polarization": f"'{msg['polarization']}'",
        "s3ingestion": f"'{msg['s3Ingestion']}'",
        "scihubingestion": f"'{msg['sciHubIngestion']}'",
        "starttime": f"'{msg['startTime']}'",
        "stoptime": f"'{msg['stopTime']}'",
        "isoceanic": f"{is_oceanic}",
    }
    return (row, tbl)


def handler_queue():
    """handler queue"""
    # Create a client.
    client = tasks_v2.CloudTasksClient()

    # TODO(developer): Uncomment these lines and replace with your values.
    project = os.getenv("GCP_PROJECT")
    queue = os.getenv("QUEUE")
    location = os.getenv("GCP_LOCATION")
    url = os.getenv("ORCHESTRATOR_URL")

    payload = None

    # Construct the fully qualified queue name.
    parent = client.queue_path(project, location, queue)

    # Construct the request body.
    task = {
        "http_request": {  # Specify the type of request.
            "http_method": tasks_v2.HttpMethod.POST,
            "url": url,  # The full url path that the task will be sent to.
        }
    }

    if payload is not None:
        # The API expects a payload of type bytes.
        converted_payload = payload.encode()

    # Add the payload to the request.
    task["http_request"]["body"] = converted_payload

    # Use the client to build and send the task.
    response = client.create_task(request={"parent": parent, "task": task})

    print("Created task {}".format(response.name))
    return response
