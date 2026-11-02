"""Inspect and apply the database_target output of a deployed Pulumi stack."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
PROD_STACK = "prod20240903"
PROD_UIS = ("cerulean-ui-production", "cerulean-ui-prod-preview")
BACKENDS = {
    "cr-orchestrator": ("DB_URL", True),
    "cr-tipg": ("DATABASE_URL", True),
    "cf-sr": ("DB_URL", False),
    "cf-historical-run": ("DB_URL", False),
    "cf-asa": ("DB_URL", False),
}


def json_command(command):
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"{command[0]} failed")
    return json.loads(result.stdout)


def load_target(stack):
    target = json_command(
        [
            "pulumi",
            "stack",
            "output",
            "database_target",
            "--stack",
            stack,
            "--json",
            "--non-interactive",
        ]
    )
    required = {
        "project",
        "region",
        "connection_name",
        "private_ip",
        "database",
        "port",
    }
    if not isinstance(target, dict) or set(target) != required:
        raise ValueError(
            "Deploy the stack's database_target output before using this helper"
        )
    if not all(isinstance(target[k], str) and target[k] for k in required - {"port"}):
        raise ValueError("The selected target must have a private IP and database name")
    parts = target["connection_name"].split(":")
    if (
        len(parts) != 3
        or parts[:2] != [target["project"], target["region"]]
        or not parts[2]
    ):
        raise ValueError("Target instance, project, and region disagree")
    ipaddress.ip_address(target["private_ip"])
    if target["port"] != 5432:
        raise ValueError("Cloud SQL connections must use port 5432")
    return target


def gcloud(target, *args):
    return json_command(
        ["gcloud", *args, f"--project={target['project']}", "--quiet", "--format=json"]
    )


def service(target, name, region):
    return gcloud(target, "run", "services", "describe", name, f"--region={region}")


def revision(target, name, region):
    return gcloud(target, "run", "revisions", "describe", name, f"--region={region}")


def serving_revision(target, deployed, region):
    traffic = [t for t in deployed["status"].get("traffic", []) if t.get("percent", 0)]
    if len(traffic) != 1 or traffic[0]["percent"] != 100:
        raise ValueError(
            f"{deployed['metadata']['name']}: expected one revision serving 100%"
        )
    name = traffic[0].get("revisionName")
    if not name:
        raise ValueError("Serving revision name is missing")
    return revision(target, name, region)


def environment(config):
    return {
        e["name"]: e.get("value")
        for e in config["spec"]["containers"][0].get("env", [])
    }


def sql_attachment(config):
    return (
        config.get("metadata", {})
        .get("annotations", {})
        .get("run.googleapis.com/cloudsql-instances", "")
    )


def connection_errors(config, target, backend=None):
    env = environment(config)
    if backend:
        key, socket = BACKENDS[backend]
        try:
            url = urlsplit(env.get(key) or "")
            host = (
                parse_qs(url.query).get("host", [None])[0] if socket else url.hostname
            )
            expected = (
                f"/cloudsql/{target['connection_name']}"
                if socket
                else target["private_ip"]
            )
            valid = (
                url.scheme in ("postgresql", "postgresql+asyncpg")
                and unquote(url.path.removeprefix("/")) == target["database"]
                and host == expected
                and (url.port or 5432) == target["port"]
            )
        except ValueError:
            valid = False
        errors = (
            [] if valid else [f"{key} does not select the target instance and database"]
        )
        attachment_required = socket
    else:
        errors = [
            f"{key} differs from target"
            for key, value in {
                "DB_HOST": target["private_ip"],
                "DB_NAME": target["database"],
                "DB_PORT": str(target["port"]),
            }.items()
            if env.get(key) != value
        ]
        attachment_required = True
    if attachment_required and sql_attachment(config) != target["connection_name"]:
        errors.append("Cloud SQL attachment differs from target")
    return errors


def inspect(target, name, region, backend=None):
    deployed = service(target, name, region)
    active = serving_revision(target, deployed, region)
    errors = []
    for label, config in (
        ("template", deployed["spec"]["template"]),
        ("serving revision", active),
    ):
        errors.extend(
            f"{label}: {error}" for error in connection_errors(config, target, backend)
        )
    return deployed, active, {"service": name, "region": region, "errors": errors}


def backend_reports(target, stack):
    return [
        inspect(target, f"cerulean-cloud-{stack}-{suffix}", target["region"], suffix)[2]
        for suffix in BACKENDS
    ]


def require_matching(reports):
    errors = [f"{r['service']}: {e}" for r in reports for e in r["errors"]]
    if errors:
        raise ValueError("\n".join(errors))


def ui_settings(config):
    env = environment(config)
    return {
        "environment": {k: env.get(k) for k in ("DB_HOST", "DB_NAME", "DB_PORT")},
        "cloudsql_instances": sql_attachment(config),
        "image": config.get("status", {}).get("imageDigest")
        or config["spec"]["containers"][0]["image"],
    }


def ui_plan(target, names, region):
    rows = []
    for name in names:
        deployed, active, report = inspect(target, name, region)
        rows.append(
            {
                **report,
                "serving_revision": active["metadata"]["name"],
                "serving": ui_settings(active),
                "template": ui_settings(deployed["spec"]["template"]),
            }
        )
    return rows


def save_snapshot(path, stack, target, rows):
    # Exclusive creation preserves previous cutover records. No credentials are saved.
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
    ) as f:
        json.dump({"stack": stack, "target": target, "uis": rows}, f, indent=2)
        f.write("\n")


def apply_ui(target, stack, names, region, snapshot):
    # Deploy/verify the backend first. This helper never runs Pulumi up or migrations.
    require_matching(backend_reports(target, stack))
    gcloud(
        target,
        "sql",
        "databases",
        "describe",
        target["database"],
        f"--instance={target['connection_name'].split(':')[2]}",
    )
    rows = ui_plan(target, names, region)
    save_snapshot(snapshot, stack, target, rows)
    print(f"Saved previous UI settings to {snapshot}")
    values = {
        "DB_HOST": target["private_ip"],
        "DB_NAME": target["database"],
        "DB_PORT": str(target["port"]),
    }
    delimiter = uuid.uuid4().hex
    while any(delimiter in value for value in values.values()):
        delimiter = uuid.uuid4().hex
    env_flag = f"^{delimiter}^" + delimiter.join(f"{k}={v}" for k, v in values.items())
    prepared = []
    for row in rows:
        name = row["service"]
        gcloud(
            target,
            "run",
            "services",
            "update",
            name,
            f"--region={region}",
            f"--image={row['serving']['image']}",
            f"--update-env-vars={env_flag}",
            f"--set-cloudsql-instances={target['connection_name']}",
            "--no-traffic",
        )
        updated = service(target, name, region)
        status = updated["status"]
        ready = status["latestReadyRevisionName"]
        if ready != status["latestCreatedRevisionName"]:
            raise ValueError(
                f"{name}: new revision is not ready; keep maintenance active"
            )
        config = revision(target, ready, region)
        require_matching(
            [{"service": name, "errors": connection_errors(config, target)}]
        )
        if ui_settings(config)["image"] != row["serving"]["image"]:
            raise ValueError(
                f"{name}: image changed during preparation; keep maintenance active"
            )
        prepared.append((name, ready))
    # Both revisions must be ready before either UI receives traffic. Freeze builds
    # and keep writers paused throughout; the two traffic updates are not atomic.
    require_matching(backend_reports(target, stack))
    for name, ready in prepared:
        gcloud(
            target,
            "run",
            "services",
            "update-traffic",
            name,
            f"--region={region}",
            f"--to-revisions={ready}=100",
        )
    reports = backend_reports(target, stack) + [
        inspect(target, n, region)[2] for n in names
    ]
    require_matching(reports)
    print(
        "All backend and UI configurations select the target. Verify application reads/writes before reopening traffic."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "apply-ui", "verify", "proxy"))
    parser.add_argument("--stack", required=True, help="Deployed backend Pulumi stack")
    parser.add_argument(
        "--ui-service",
        action="append",
        help="Defaults to both PROD UIs for prod20240903",
    )
    parser.add_argument("--ui-region", default="us-east4")
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Required new file for apply-ui; previous UI settings, without credentials",
    )
    parser.add_argument("--port", type=int, default=5432, help="Local port for proxy")
    args = parser.parse_args(argv)
    names = args.ui_service or (list(PROD_UIS) if args.stack == PROD_STACK else [])
    if args.command != "proxy" and (not names or len(set(names)) != len(names)):
        parser.error(
            "Specify distinct --ui-service names for stacks other than prod20240903"
        )
    if args.stack == PROD_STACK and set(names) != set(PROD_UIS):
        parser.error("PROD checks and updates must include both production UIs")
    if args.command == "apply-ui" and not args.snapshot:
        parser.error("apply-ui requires --snapshot pointing to a new file")
    try:
        target = load_target(args.stack)
        if args.command == "proxy":
            print(
                f"Database: {target['database']}; local port: {args.port}", flush=True
            )
            return subprocess.run(
                ["cloud-sql-proxy", f"--port={args.port}", target["connection_name"]]
            ).returncode
        if args.command == "apply-ui":
            apply_ui(target, args.stack, names, args.ui_region, args.snapshot)
        else:
            reports = backend_reports(target, args.stack)
            rows = ui_plan(target, names, args.ui_region)
            print(
                json.dumps(
                    {
                        "stack": args.stack,
                        "target": target,
                        "backends": reports,
                        "uis": rows,
                    },
                    indent=2,
                )
            )
            if args.command == "verify":
                require_matching(reports + rows)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Database target check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
