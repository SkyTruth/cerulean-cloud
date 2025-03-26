import datetime
import os
import requests

from google.cloud import secretmanager


def get_secret(secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.environ['GCP_PROJECT']}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def check_recent_slicks():
    limit = 10
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=1)

    st = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    fn = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    cerulean_url = f"https://api.cerulean.skytruth.org/collections/public.slick_plus/items?limit={limit}&datetime={st}/{fn}"

    data = requests.get(cerulean_url).json()

    if len(data["features"]) == limit:
        return True
    else:
        return False


def send_alert_recent_slicks():
    webhook_url = get_secret(os.environ["SLACK_WEBHOOK_SECRET"])
    requests.post(webhook_url, json={"text": "No new slicks in the last 24 hours"})
    return None


def main():
    if check_recent_slicks():
        send_alert_recent_slicks()
