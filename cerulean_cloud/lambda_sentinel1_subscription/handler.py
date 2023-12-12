"""handler for SNS topic, to cloud function"""
import json
import os
from http.client import HTTPSConnection


def lambda_handler(event, context):
    """handle lambda"""
    function_url = os.getenv("FUNCTION_URL")
    domain, path = function_url.replace("https://", "").split("/", 1)
    print(event)
    conn = HTTPSConnection(domain)
    conn.request(
        "POST",
        "/" + path,
        body=json.dumps(event),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('API_KEY')}",
        },
    )
    res = conn.getresponse()
    return {"statusCode": res.status}
