import datetime
import os
import requests

from google.cloud import secretmanager

BASE_URL = "https://api.cerulean.skytruth.org/collections/public.slick_plus"


def get_secret(secret_id):
    """
    Get secrets
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.environ['GCP_PROJECT']}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def check_recent_slicks():
    """
    Hits the Cerulean API to see if any new slicks have been added in the past 24 hours
    """

    limit = 10
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=1)

    st = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    fn = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    cerulean_url = f"{BASE_URL}/items?limit={limit}&datetime={st}/{fn}"

    data = requests.get(cerulean_url).json()

    return True if len(data["features"]) == limit else False


def send_alert_no_recent_slicks():
    """
    Sends `no slick` warning to Slack channel
    """
    webhook_url = get_secret(os.environ["SECRET_NAME"])
    _ = requests.post(webhook_url, json={"text": "No new slicks in the last 24 hours"})


def success():
    """
    Perform success function (generally used for testing)
    """
    webhook_url = get_secret(os.environ["SECRET_NAME"])
    _ = requests.post(
        webhook_url, json={"text": "All is going to according to plan . . ."}
    )


def main(request):
    if check_recent_slicks():
        send_alert_no_recent_slicks()
    else:
        success()
