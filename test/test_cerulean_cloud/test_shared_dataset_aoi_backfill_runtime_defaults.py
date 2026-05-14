from scripts.backfill_shared_dataset_aoi import build_parser

from cerulean_cloud.cloud_run_aoi_backfill import service
from cerulean_cloud.cloud_run_aoi_backfill.schema import RunRequest


def test_run_request_defaults_match_service_runtime_budget():
    payload = RunRequest(asset_slug="mpa")

    assert payload.max_batches == service.DEFAULT_RUN_MAX_BATCHES
    assert payload.statement_timeout == service.DEFAULT_RUN_STATEMENT_TIMEOUT


def test_backfill_cli_run_defaults_match_service_runtime_budget():
    parser = build_parser()
    args = parser.parse_args(["run", "mpa", "--short-name", "MPA"])

    assert args.max_batches == service.DEFAULT_RUN_MAX_BATCHES
    assert args.statement_timeout == service.DEFAULT_RUN_STATEMENT_TIMEOUT


def test_backfill_cli_requires_short_name_for_run():
    parser = build_parser()

    try:
        parser.parse_args(["run", "mpa"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected parse failure when --short-name is omitted")
