from flask import Request, make_response
import datetime
import os
import requests
import time

BASE_URL = "https://api.cerulean.skytruth.org/collections/public.slick_plus"

WEBHOOK_URL = os.environ["SLACK_ALERTS_WEBHOOK"]


def send_alert_no_recent_slicks():
    """
    Sends `no slick` warning to Slack channel
    """
    _ = requests.post(WEBHOOK_URL, json={"text": "No new slicks in the last 24 hours"})


def send_alert_failed_connection():
    """
    Sends `no slick` warning to Slack channel
    """
    _ = requests.post(WEBHOOK_URL, json={"text": f"Failed connection to {BASE_URL}"})


def check_recent_slicks():
    """
    Hits the Cerulean API to see if any new slicks have been added in the past 24 hours
    """

    LIMIT = 10
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2

    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=1)

    st = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    fn = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    cerulean_url = f"{BASE_URL}/items?limit={LIMIT}&datetime={st}/{fn}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(cerulean_url)
            response.raise_for_status()  # raises HTTPError for bad responses (4xx, 5xx)
            data = response.json()
            break
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                send_alert_failed_connection()
                return True

    data = requests.get(cerulean_url).json()

    return True if len(data["features"]) == 0 else False


def success():
    """
    Perform success function (generally used for testing)
    """
    print("No errors detected")


def main(request: Request):
    if check_recent_slicks():
        send_alert_no_recent_slicks()
    else:
        success()
    return make_response("Function executed", 200)
