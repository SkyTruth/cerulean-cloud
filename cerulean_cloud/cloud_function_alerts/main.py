from flask import Request, make_response
import datetime
import os
import requests
import time

import google.auth
from google.auth.transport.requests import AuthorizedSession

WEBHOOK_URL = os.environ["SLACK_ALERTS_WEBHOOK"]
TIPG_URL = os.environ["TIPG_URL"]
DRY_RUN = os.getenv("IS_DRY_RUN", "").lower() == "true"

BASE_URL = f"{TIPG_URL}/collections/public.source_plus"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10
MINIMUM_RESULT_COUNT = 1
SLICK_TYPES = ["VESSEL", "INFRA"]  # , "DARK", "NATURAL"]

# projects/<project>/locations/<region>/jobs/<name> of the ASA daily batch's job
ASA_SCHEDULER_JOB = os.getenv("ASA_SCHEDULER_JOB", "")
# The batch job runs once a day. Its last attempt may start a few seconds before
# the scheduled time, so a missed run is one with no attempt within this of it.
ASA_BATCH_PERIOD = datetime.timedelta(days=1)
ASA_BATCH_TOLERANCE = datetime.timedelta(hours=1)
# google.rpc.Code names for the codes a Scheduler HTTP attempt usually ends with
STATUS_NAMES = {
    2: "UNKNOWN",
    4: "DEADLINE_EXCEEDED",
    7: "PERMISSION_DENIED",
    13: "INTERNAL",
    14: "UNAVAILABLE",
    16: "UNAUTHENTICATED",
}


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


def parse_time(value):
    """UTC datetime from an RFC 3339 timestamp such as 2026-11-23T15:00:00.849635Z."""
    if not value:
        return None
    return datetime.datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=datetime.timezone.utc
    )


def check_asa_batch(dry_run=DRY_RUN, webhook_url=WEBHOOK_URL, session=None, now=None):
    """
    Alerts when the ASA daily batch's last Scheduler attempt failed (after its
    retries), or when its most recent scheduled run made no attempt. A paused job
    isn't checked.
    """
    if not ASA_SCHEDULER_JOB:
        print("ASA_SCHEDULER_JOB is unset; skipping the ASA batch check")
        return
    name = ASA_SCHEDULER_JOB.rsplit("/", 1)[-1]
    if session is None:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        session = AuthorizedSession(credentials)
    now = now or datetime.datetime.now(datetime.timezone.utc)

    try:
        response = session.get(
            f"https://cloudscheduler.googleapis.com/v1/{ASA_SCHEDULER_JOB}", timeout=60
        )
        response.raise_for_status()
        job = response.json()
    except requests.exceptions.RequestException as e:
        send_slack_alert(
            webhook_url,
            f"Couldn't read the ASA daily batch job {name}: {e}",
            dry_run=dry_run,
        )
        return

    if job.get("state") == "PAUSED":
        print(f"ASA daily batch job {name} is paused; not checked")
        return
    last = job.get("lastAttemptTime")
    last_time = parse_time(last)
    # scheduleTime is the next run, so the most recent one was a period earlier.
    next_run = parse_time(job.get("scheduleTime")) or now
    expected = next_run - ASA_BATCH_PERIOD - ASA_BATCH_TOLERANCE
    code = (job.get("status") or {}).get("code", 0)
    # A job created or edited since then may not have had its run yet.
    changed = parse_time(job.get("userUpdateTime"))
    missed = (last_time is None or last_time < expected) and not (
        changed and changed > expected
    )
    if missed:
        send_slack_alert(
            webhook_url,
            f"ASA daily batch ({name}) hasn't run since {last or 'it was created'}",
            dry_run=dry_run,
        )
    elif code:
        message = (job.get("status") or {}).get("message") or STATUS_NAMES.get(
            code, f"code {code}"
        )
        send_slack_alert(
            webhook_url,
            f"ASA daily batch ({name}) failed, including its retries. Last attempt "
            f"{last}: {message}. The ASA function's logs list any scenes it "
            "skipped ('scenes skipped after errors').",
            dry_run=dry_run,
        )
    else:
        print(f"ASA daily batch ({name}) last attempt {last} succeeded")


def main(request: Request):
    print(f"Base URL: {BASE_URL}; dry run: {DRY_RUN}")
    body = request.get_json(silent=True) or {}
    if body.get("check") == "asa_batch":
        check_asa_batch(dry_run=DRY_RUN, webhook_url=WEBHOOK_URL)
    else:
        check_recent_slicks(dry_run=DRY_RUN, webhook_url=WEBHOOK_URL)
    print("Check complete!")
    return make_response("Function executed", 200)
