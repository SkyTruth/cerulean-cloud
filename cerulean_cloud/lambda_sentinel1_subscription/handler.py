"""handler for SNS topic, to cloud function"""
import os

import requests


def lambda_handler(event, context):
    """handle lambda"""
    function_url = os.getenv("FUNCTION_URL")
    print(event)
    res = requests.post(function_url, json=event)
    return {"statusCode": res.status_code}
