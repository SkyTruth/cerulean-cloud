"""Environment and history conventions shared by the historical-run CLIs."""

import os


AUTH_FAILURE = "Google authentication failed."
SELECTION_FAILURE = "Historical scene selection failed."
SUBMISSION_FAILURE = "Historical task submission failed."
SAFE_FAILURES = (AUTH_FAILURE, SELECTION_FAILURE, SUBMISSION_FAILURE)


def load_local_env(path):
    from dotenv import load_dotenv

    key = "GOOGLE_APPLICATION_CREDENTIALS"
    supplied_by_caller = key in os.environ
    load_dotenv(path)
    # The notebook uses inline JSON for other tools. Google ADC expects a path.
    if not supplied_by_caller and os.environ.get(key, "").lstrip().startswith("{"):
        del os.environ[key]
