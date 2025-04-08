from flask import Request, make_response
import datetime
import os
import requests
import time

WEBHOOK_URL = os.environ["SLACK_ALERTS_WEBHOOK"]
TIPG_URL = os.environ["TIPG_URL"]
DRY_RUN = os.getenv("IS_DRY_RUN", "").lower() == "true"

BASE_URL = f"{TIPG_URL}/collections/public.source_plus"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


def send_alert_no_recent_slicks_message(slick_type, dry_run=DRY_RUN):
    """
    Sends `no slick` warning to Slack channel
    """
    if dry_run:
        print(f"No new {slick_type} slicks in the last 24 hours in {TIPG_URL}")
    else:
        _ = requests.post(
            WEBHOOK_URL,
            json=f"No new {slick_type} slicks in the last 24 hours in {TIPG_URL}",
        )


def send_alert_failed_connection_message(dry_run=DRY_RUN):
    """
    Sends `no slick` warning to Slack channel
    """
    if dry_run:
        print(f"Failed connection to {BASE_URL}")
    else:
        _ = requests.post(
            WEBHOOK_URL, json={"text": f"Failed connection to {BASE_URL}"}
        )


def send_success_message():
    """
    Perform success function (generally used for testing)
    """
    print("No errors detected")


def fetch_with_retries(st, fn, slick_type=None):
    """
    Attempts to fetch data from the slick detection API with retry logic.

    If data is found (`numberMatched > 0`), a success message is sent.
    If no data is found, a "no slicks" alert is triggered.
    On repeated connection failures, a failure alert is sent.

    Parameters:
        st (str): Start datetime (ISO format).
        fn (str): End datetime (ISO format).
        slick_type (str, optional): Optional source type to filter the query.
    """
    src_var = "" if slick_type is None else f"&source_type={slick_type}"
    url = f"{BASE_URL}/items?limit=1{src_var}&datetime={st}/{fn}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            if response.json()["numberMatched"] > 0:
                send_success_message()
            else:
                send_alert_no_recent_slicks_message(slick_type)
            return

        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                send_alert_failed_connection_message(url)


def check_recent_slicks():
    """
    Hits the Cerulean API to see if any new slicks have been added in the past 24 hours
    """

    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=1)
    st = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    fn = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    for slick_type in ["VESSEL", "INFRA", "DARK"]:
        fetch_with_retries(st, fn, slick_type=slick_type)


def main(request: Request):
    check_recent_slicks()
    return make_response("Function executed", 200)
