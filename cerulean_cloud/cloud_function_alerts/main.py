from flask import Request, make_response
import datetime
import os
import requests
import time

WEBHOOK_URL = os.environ["SLACK_ALERTS_WEBHOOK"]
TIPG_URL = os.environ["TIPG_URL"]

# TODO: replace
DRY_RUN = False
# DRY_RUN = os.getenv("IS_DRY_RUN", "").lower() == "true"

# TODO: replace!!
# BASE_URL = f"{TIPG_URL}/collections/public.source_plus"
BASE_URL = "https://api.cerulean.skytruth.org/collections/public.source_plus"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
MINIMUM_RESULT_COUNT = 1
SLICK_TYPES = ["VESSEL", "INFRA"]  # , "DARK", "NATURAL"]


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


def send_alert_failed_connection_message(url, dry_run=DRY_RUN):
    """
    Sends `no slick` warning to Slack channel
    """
    if dry_run:
        print(f"Failed connection to {url}")
    else:
        _ = requests.post(WEBHOOK_URL, json={"text": f"Failed connection to {url}"})


def send_success_message(slick_type, num_matched):
    """
    Perform success function (generally used for testing)
    """
    print(f"No errors detected; found {num_matched} {slick_type} slicks")


def fetch_with_retries(st, fn, base_url=BASE_URL, dry_run=DRY_RUN, slick_type=None):
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
    url = f"{base_url}/items?limit=1{src_var}&datetime={st}/{fn}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            num_matched = response.json()["numberMatched"]

            if num_matched > MINIMUM_RESULT_COUNT:
                send_success_message(slick_type, num_matched)
            else:
                send_alert_no_recent_slicks_message(slick_type)
            return

        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                send_alert_failed_connection_message(url, dry_run=dry_run)


def check_recent_slicks():
    """
    Hits the Cerulean API to see if any new slicks have been added in the past 24 hours
    """

    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=1)
    st = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    fn = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    for slick_type in SLICK_TYPES:
        fetch_with_retries(
            st, fn, base_url=BASE_URL, dry_run=DRY_RUN, slick_type=slick_type
        )


def main(request: Request):
    # TODO: remove
    print(f"dry run: {os.getenv('IS_DRY_RUN', '').lower()}")
    check_recent_slicks()
    return make_response("Function executed", 200)
