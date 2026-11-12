"""Resolve local submission targets without assuming Pulumi's generated suffixes."""

import google.auth
from google.auth.transport.requests import AuthorizedSession


def resolve_orchestrator_queue(project, region, stage):
    """Find the unique deployed orchestrator queue for a stack."""
    prefix = f"cerulean-cloud-{stage}-queue-cr-orchestrator"
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    matches = []
    with AuthorizedSession(credentials) as session:
        session.headers["x-goog-user-project"] = project
        params = {"pageSize": 1000}
        while True:
            response = session.get(
                f"https://cloudtasks.googleapis.com/v2/projects/{project}/locations/{region}/queues",
                params=params,
                timeout=30,
                max_allowed_time=30,
            )
            response.raise_for_status()
            page = response.json()
            for queue in page.get("queues", []):
                name = queue["name"].rsplit("/", 1)[-1]
                if name == prefix or name.startswith(prefix + "-"):
                    matches.append(name)
            token = page.get("nextPageToken")
            if not token:
                break
            params["pageToken"] = token
    if len(matches) != 1:
        raise ValueError(
            f"Expected one orchestrator queue for {stage}; found {len(matches)}"
        )
    return matches[0]
