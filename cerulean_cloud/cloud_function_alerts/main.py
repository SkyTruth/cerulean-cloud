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
RETRY_DELAY_SECONDS = 10
MINIMUM_RESULT_COUNT = 1
SLICK_TYPES = ["VESSEL", "INFRA"]  # , "DARK", "NATURAL"]


def send_slack_alert(webhook_url, message, dry_run=DRY_RUN):
    """
    Sends a slack message if dry_run=False otherwise logs alert in CloudRun Logs
    """
    if dry_run:
        print(f"WARNING: {message}")
    else:
        message_json = {"text": message}
        try:
            response = requests.post(webhook_url, json=message_json, timeout=10)
            response.raise_for_status()
            print("Slack alert sent successfully")
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error: {http_err}")
            print(f"Response text: {response.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"Request failed: {req_err}")


def send_success_message(slick_type, num_matched):
    """
    Perform success function (generally used for testing)
    """
    print(f"No errors detected; found {num_matched} {slick_type} slicks")


def fetch_with_retries(
    st, fn, base_url=BASE_URL, dry_run=DRY_RUN, webhook_url=WEBHOOK_URL, slick_type=None
):
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

    print(f"fetching slicks ({slick_type})")
    src_var = "" if slick_type is None else f"&source_type={slick_type}"
    url = f"{base_url}/items?limit=1{src_var}&datetime={st}/{fn}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            num_matched = response.json()["numberMatched"]

            if num_matched > MINIMUM_RESULT_COUNT:
                send_success_message(slick_type, num_matched)
            else:
                send_slack_alert(
                    webhook_url,
                    f"No new {slick_type} slicks in the last 24 hours in {TIPG_URL}",
                    dry_run=dry_run,
                )
            return

        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES:
                print(f"Unsuccessful attempt to connect to {url}: ATTEMPT {attempt}")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                send_slack_alert(
                    webhook_url, f"Failed connection to {url}", dry_run=dry_run
                )


def check_recent_slicks(dry_run=DRY_RUN, webhook_url=WEBHOOK_URL):
    """
    Hits the Cerulean API to see if any new slicks have been added in the past 24 hours
    """

    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=1)
    st = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    fn = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    for slick_type in SLICK_TYPES:
        fetch_with_retries(
            st,
            fn,
            base_url=BASE_URL,
            dry_run=dry_run,
            webhook_url=webhook_url,
            slick_type=slick_type,
        )


def main(request: Request):
    print(f"Base URL: {BASE_URL}; dry run: {DRY_RUN}")
    check_recent_slicks(dry_run=DRY_RUN, webhook_url=WEBHOOK_URL)
    print("Check complete!")
    return make_response("Function executed", 200)
