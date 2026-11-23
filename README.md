# Cerulean API Documentation
## Introduction
For most users, we recommend using the [Cerulean web application](https://cerulean.skytruth.org/), which provides a visual interface for exploring the complete set of Cerulean data layers.

For users who want to directly access and download potential oil slick detection data, we provide programmatic free access to an [OGC compliant API](https://api.cerulean.skytruth.org/).

Start with the [API guide](https://api.cerulean.skytruth.org/guide) for queries, data definitions and interpretation, or [run the example notebook in Colab](https://colab.research.google.com/github/SkyTruth/cerulean-cloud/blob/main/notebooks/Cerulean%20API%20Guide.ipynb). Automated clients can also use the [Cerulean investigation skill](https://api.cerulean.skytruth.org/skills/cerulean-investigation/SKILL.md). All are maintained in this repository alongside the API.


# cerulean-cloud
Pulumi repository with infrastructure for Cerulean, including all cloud services and database structure.

## Architecture

The cerulean-cloud architecture diagram can be found [here](https://lucid.app/lucidchart/26eb638d-3ba3-4461-aa1d-04ae4ec55d52/edit?viewport_loc=-2228%2C-1113%2C2759%2C3315%2CHowzCfBBZfS3&invitationId=inv_65f7accd-dc09-4ef1-9c28-81a34ce0976e#).

![Cerulean Cloud Architecture](architecture.png)

## Deployment

Deployment is fully managed by GitHub actions and Pulumi and the entire workflow defined in [this](.github/workflows/test_and_deploy.yml) YAML file.

We have defined three development stages / stacks:
- __TEST__: this is the Work in Progress (WIP) deployment stage. Will often be broken. Deployment can be triggered to TEST using the [workflow_dispatch](https://github.blog/changelog/2020-07-06-github-actions-manual-triggers-with-workflow_dispatch/) from any branch.
- __STAGING__: this is the stable development deployment. Should be used to perform more complex integration tests, with close to real data. Deployment to STAGING is triggered with any merge commit to `main`.
- __PRODUCTION__: this is the production / publicly available deployment. Any deployment into PRODUCTION should pass rigorous integration tests in STAGING. Deployment can be triggered into PRODUCTION by adding a tag to a commit (`git tag v0.0.1 && git push --tags`).

__Pulumi deployments__

In order to make development easier we have defined two pulumi deployments that are intended to work in tandem:
- [cerulean-cloud-images](images/Pulumi.yaml): deploys all necessary docker images into Google Cloud Registry (GCR), in preparation for deploying the Cloud Run functions that require those images.
- [cerulean-cloud](Pulumi.yaml): deploys all cerulean-cloud infrastructure (except docker images).

For each of these deployments there exists a configuration directory that includes a YAML configuration file per stage / stack (named with the stage name itself i.e. `Pulumi.test.yaml`, `Pulumi.staging.yaml`, `Pulumi.production.yaml`). These files include configuration that is stage / stack specific, such as deployment regions, usernames and passwords for external services, etc. They should be managed using the Pulumi CLI (`pulumi config set someparam`) but can also be edited directly.

__Initial deployment__

If you are deploying a completely new stack, make sure to create matching configuration files in `cerulean-cloud-images` and `cerulean-cloud`, with matching stack names. In addition, specifically for the tipg deployment, since the database is empty when a stack is deployed for the first time (alembic migrations occur after the initial migration), if you want to access `tipg` after this initial deployment make sure to poll the `/register` endpoint of the resulting URL in order to correctly load the tables (i.e. `curl https://some-tipg-url.app/register`). For any deployments after the first one, this is not required.

__Decreasing cold starts for Cloud Run__

In order to decrease response time for Cloud Run (especially in the production services, since for test and staging this will increase costs) you can set the minimal instances to 1 to run at any given moment in time (see [documentation](https://cloud.google.com/run/docs/tips/general#:~:text=Using%20minimum%20instances%20to%20reduce%20cold%20starts,-You%20can%20configure&text=You%20should%20set%20a%20number,for%20longer%20than%2010%20seconds.)). Due to a `pulumi` limitation, when the service is first created this property cannot be set so we advise to set this value manually once the deployment has been completed.

<!--- __Metrics__

Google Cloud provides nice dashboards for tracking the stability, response time and resource consumption of the cloud resources. The links below point to the PRODUCTION deployment, but for every stack's components you'll find similar dashboards:
- [Cloud Run orchestrator metrics](https://console.cloud.google.com/run/detail/europe-west1/cerulean-cloud-production-cr-orchestrator/metrics?project=cerulean-338116)
- [Cloud Run offset tile metrics](https://console.cloud.google.com/run/detail/europe-west1/cerulean-cloud-production-cr-infer/metrics?project=cerulean-338116)
- [Cloud Tasks queue metrics](https://console.cloud.google.com/cloudtasks/queue/europe-west1/cerulean-cloud-production-queue-cr-orchestrator-d8ed51e/metrics?project=cerulean-338116)
- [Cloud Run tipg metrics](https://console.cloud.google.com/run/detail/europe-west1/cerulean-cloud-production-cr-tipg/metrics?project=cerulean-338116)
- [Cloud Function scene relevancy metrics](https://console.cloud.google.com/functions/details/europe-west1/cerulean-cloud-production-cf-scene-relevancy?env=gen1&project=cerulean-338116)
- [Cloud Function historical run metrics](https://console.cloud.google.com/functions/details/europe-west1/cerulean-cloud-production-cf-historical-run?env=gen1&project=cerulean-338116)
--->
## Development

In order to develop in cerulean-cloud repository we recommend the following system wide requirements (for MacOS), in addition to the python specific requirements listed below:
- [Docker](https://www.docker.com/products/docker-desktop/)
- [brew](https://brew.sh/)
- [Python 3.8](https://www.python.org/downloads/release/python-380/)
- [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html#)
- [Google Cloud Platform (GCP) CLI](https://cloud.google.com/sdk/docs/install)
- [Amazon Web Services (AWS) CLI](https://aws.amazon.com/cli/)
- [Pulumi](https://www.pulumi.com/docs/get-started/install/) - can be installed with `brew install pulumi`
- [pgAdmin](https://www.pgadmin.org/) - Recommend tool for database interactions

### Setup cloud authentication
#### GCP authentication
```sh
gcloud config set account rodrigo@developmentseed.org
gcloud config configurations create cerulean --project cerulean-338116 --account rodrigo@developmentseed.org
gcloud config configurations activate cerulean
```

Also, make sure to authenticate into docker with GCP to allow interaction with GCR:
```sh
gcloud auth configure-docker
```
#### AWS authentication
```sh
aws configure --profile cerulean
export AWS_PROFILE=cerulean
```

### Setup your python virtualenv

__WARNING__: Setting up your local virtualenv can vary slightly depending on your Operating System (OS) and python installation (native or `conda`). Procede with caution!

Make sure that you have setup your shell script as defined in the `mkvirtualenv` documentation. This will vary slightly with your python installation but you will need to change you bash profile file, by adding the following variables:
```sh
export VIRTUALENVWRAPPER_PYTHON="$(command -v python3)" # the Python installation where you installed mkvirtualenv
export WORKON_HOME="${WORKON_HOME:-$HOME/.virtualenvs}"
export PROJECT_HOME="${PROJECT_HOME:-$HOME/src}"
source /path/to/virtualenvwrapper.sh # this path can vary depending on your installation
```

Then you'll be able to run:
```sh
mkvirtualenv cerulean-cloud --python=$(which python3.8)
pip install -r requirements.txt
pip install -r requirements-test.txt
# Additional requirements files
pip install -r cerulean_cloud/cloud_run_infer/requirements.txt
pip install -r cerulean_cloud/cloud_run_orchestrator/requirements.txt
pip install -r cerulean_cloud/cloud_run_tipg/requirements.txt
pip install -r cerulean_cloud/titiler_sentinel/requirements.txt
# Setup pre-commit
pre-commit install
```

To activate your virtual environment:
```sh
workon cerulean-cloud
```
#### For notebook development
```sh
pip install ipykernel
python -m ipykernel install --user --name=cerulean-cloud
```

### Running tests

You can run tests using `pytest` commands:
```sh
pytest
pytest test/test_cerulean_cloud/test_tiling.py # run only tests in a specific module
pytest test/test_cerulean_cloud/test_tiling.py::test_from_base_tiles_create_infer # run only a specific test
```

The PostgreSQL fixtures use Psycopg 3, which is included in `requirements-test.txt`.

### Pulumi
#### Check available stages
```
pulumi stack ls
```
Select another stage
```
pulumi stack select test
```

#### Set config
Set secret values with (passwords, keys, etc):
```
pulumi config set db:db-password --secret
```

Set other config values with:
```
pulumi config set infra_distance
```
#### Preview changes (no need to run locally)

Make sure docker is running in your machine before running this command.
```
pulumi preview
```

This would be the output:
```
Previewing update (test):

docker:index:RemoteImage cerulean-cloud-images-test-remote-offset  completing deletion from previous update
docker:index:RemoteImage cerulean-cloud-images-test-remote-orchestrator  completing deletion from previous update
docker:index:RemoteImage cerulean-cloud-images-test-remote-tipg  completing deletion from previous update
-  docker:index:RemoteImage cerulean-cloud-images-test-remote-offset delete completing deletion from previous update
-  docker:index:RemoteImage cerulean-cloud-images-test-remote-tipg delete completing deletion from previous update
-  docker:index:RemoteImage cerulean-cloud-images-test-remote-orchestrator delete completing deletion from previous update
...
pulumi:pulumi:Stack cerulean-cloud-test running Creating lambda package [running in Docker]...
pulumi:pulumi:Stack cerulean-cloud-test running Building docker image...
pulumi:pulumi:Stack cerulean-cloud-test running Copying package.zip ...
pulumi:pulumi:Stack cerulean-cloud-test running Copied package package.zip ...
pulumi:pulumi:Stack cerulean-cloud-test  4 messages
```

This process is run on push on any open PRs, and you'll be able to see the output as a comment in your PR as this [one](https://github.com/SkyTruth/cerulean-cloud/pull/61#issuecomment-1205573107).
#### Deploy changes (no need to run locally)
```
pulumi up
```

#### Destroy and rebuild
pulumi destroy
Use GUI to delete the database
pulumi refresh
pulumi state delete <URN of IAM if sticky>
pulumi refresh
pulumi destroy
test and deploy
If there is a lock on the stack, you can delete that lock in gs://cerulean-cloud-state/cerulean-cloud-images

## Database

![Database Schema as of 2023-08-01](<Database Schema 2023-08-01.png>)

### Connecting

In order to connect to the deployed database, you can use the [Cloud SQL proxy for authentication](https://cloud.google.com/sql/docs/mysql/connect-admin-proxy). First install the proxy in your local machine (instructions [here](https://cloud.google.com/sql/docs/mysql/connect-admin-proxy#install)).

You can then find the instance connection name and the connection string in the outputs of your active pulumi stack:
```sh
pulumi stack --show-secrets
# use `database_instance_name` in Cloud SQL proxy
# use `database_url_alembic` to connect in your client
```

Start the Cloud SQL proxy (make sure you are properly authenticated with GCP):
```sh
cd . # where cloud_sql_proxy
./cloud_sql_proxy -instances=$'{database_instance_name}'=tcp:0.0.0.0:5432
```

In order to connect in pgAdmin, you can take apart the connection string that you get from the pulumi output:
```sh
postgresql://cerulean-cloud-test-database:some_password@127.0.0.1:5432/cerulean-cloud-test-database
# postgresql://${USER}:${PASSWORD}@${HOST}:${PORT}/${DATABASE_NAME}
# HOST and PORT refer to the cloud sql proxy host (your localhost)
```

In another process connect to the database (i.e. with `psql`):
```sh
psql ${database_url_alembic}
```

### Migrations

We are using [alembic](https://alembic.sqlalchemy.org/en/latest/tutorial.html) to run migration in our database. You can create a new revision using:
```sh
alembic revision -m "Add new table"
```
And apply this revision with:
```sh
# Ensure you have access to your database and have setup DB_URL environment variable with the connection string above
alembic upgrade head
```

If you want to look at common operations with `alembic` make sure to check out the previously run migrations in the `alembic/versions` folder. For instance:
- [Adding tables](alembic/versions/3c4693517ef6_add_tables.py)
- [Adding views](alembic/versions/39277f6278f4_add_view.py)
- [Adding records](alembic/versions/c941681a050d_add_initial_records.py)
- [Adding columns to existing tables](alembic/versions/0a4536575154_add_simplified_eez_geom.py [OUTDATED])
- And much more!

During the deployment process with GitHub Actions, migrations will be automatically run when new revisions are included in the branch/commit.

## Authentication

Most services deployed with cerulean-cloud are safeguarded against abuse by outside actors using an API key authentication. This means that when interacting with the majority of the endpoints in your client of choice (i.e. `httpx` in Python, `curl` in your terminal, Postman or QGIS) you should make sure to include the following authentication header:

```json
{"Authorization": "Bearer SOME_API_KEY"}
```

The API_KEY we use is set on the stack configuration file with pulumi and is encrypted. In order to access the API key for the currently selected stack you can run:
```sh
pulumi stack output api_key
```

You could then save this value as an environment variable for later use.

As an example, to place a request to the cloud run orchestrator, using `httpx` you'd do the following:
```python
import httpx
URL = "https://cerulean-cloud-test-cr-orchestrator-5qkjkyomta-ew.a.run.app"
API_KEY= "SOME_API_KEY"
SCENES = [
    "S1A_IW_GRDH_1SDV_20230523T224049_20230523T224114_048667_05DA7A_91D1", # INDONESIA
    "S1A_IW_GRDH_1SDV_20230320T062953_20230320T063018_047724_05BB92_AC28", # UK
    "S1A_IW_GRDH_1SDV_20210523T005625_20210523T005651_038008_047C68_FE94", # INDIA
    "S1A_IW_GRDH_1SDV_20230711T160632_20230711T160657_049378_05F013_448A", # EGYPT
    "S1A_IW_GRDH_1SDV_20230330T015937_20230330T020003_047867_05C077_A6AC", # OMAN
    "S1A_IW_GRDH_1SDV_20230302T001557_20230302T001622_047458_05B29B_DF03", # MEXICO
    "S1A_IW_GRDH_1SDV_20230618T232014_20230618T232039_049047_05E5E0_718C", # USA
]
for scene_id in SCENES:
	orchestrator_result = httpx.post(URL+"/orchestrate",
									json={"scene_id": f"{scene_id}"},
									timeout=None,
									headers={"Authorization": f"Bearer {API_KEY}"})
	print(orchestrator_result)
```

The services deployed by cerulean-cloud that DO NOT require this API key are:
- tipg Cloud Run
- Scene relevancy Cloud Function


## Historical processing and scene inventory

Historical-run discovers Sentinel-1 scenes through **EODAG's GEODES catalog**.
The `eodag` selector takes UTC start/end boundaries and an optional GeoJSON
FeatureCollection region, searches `S1_SAR_GRD` products containing VV
(SV `VV` or DV `VV VH`), and retains H/M-resolution scenes intersecting the ocean.
Discovery uses the same low-resolution ocean mask as live scene relevancy, which
excludes the Caspian Sea.
Filtering selects whole scenes; it does not clip inference to the requested region.
Explicit `scene_ids` or the CLI's `scenes` command accept already selected IDs,
including CSV input.
Omit `--geometry` for a global search, or pass `--geometry region.geojson` to
restrict the region. HTTP requests may omit `geometry` or set it to `null`.
Global runs store `NULL` in the backfill record's geometry column and use the
same ocean and scene history filters as regional searches.
Use `--mode EW --mode IW --mode SM` to select multiple acquisition modes, or
repeat any subset of those choices. Mode values are case-insensitive in the CLI.
Omitting `--mode` leaves acquisition mode unrestricted. HTTP requests accept a
nonempty `modes` list, such as `["EW", "IW", "SM"]`, for EODAG discovery.
Discovery uses the bundled `geodes.yml`, compatible with the pinned EODAG version.
Public metadata search requires no GEODES key and does not download imagery.
Catalog URNs are mapped to Sentinel scene names before queueing.
Discovery splits date-range searches into seven-day windows and starts up to **32
GEODES requests at once** across windows and pages. The first page gives the total
for its window. The runner requests 80 products per page and reuses each worker's
HTTP connection. Set `GEODES_PAGE_WORKERS` to an integer from 1 to 32 to lower
concurrency when needed.

The runner makes up to three attempts per page for transient errors. An HTTP 400
on page 1 stops that search immediately. On a later page advertised by the catalog,
the runner retries the same request with backoff. It validates
page sizes, consistent totals, and unique catalog records within each window before
submitting tasks. A record on a shared date boundary is counted once. A missing
or repeated page stops submission.

Every path deduplicates IDs. By default, historical runs exclude scenes with any
`orchestrator_run.success IS TRUE` in the selected stack's database, including in
previews and dry runs. Failed or unfinished runs and scene metadata alone remain
eligible. Use `--overwrite` to skip that lookup and include previously processed
scenes for either date-range discovery or `scenes`. HTTP requests and direct runner
calls use `"overwrite": true`. The response's `already_processed_scene_count`
counts excluded scenes and is zero with `--overwrite` because the lookup is skipped.
Already queued or running scenes are not atomically reserved, so inspect partial
failures before resubmitting.

The CLI defaults to EODAG discovery, so the `eodag` selector can be omitted.
Use `scenes` for explicit IDs or a CSV; the explicit `eodag` selector also works.
Run `python -m scripts.historical_run eodag --help` for discovery options.
The CLI uses the deployed historical-run endpoint
by default, with `CERULEAN_API_KEY` and either `HISTORICAL_RUN_URL`/`--url` or
Pulumi's `historical_run_url` output for the selected `--stage` stack. Use `--local`
to run the same Python runner directly, with database access and Cloud Tasks
credentials. All historical submissions default to dry-run processing:

| CLI option | Processing behavior |
| --- | --- |
| Omitted | Dry-run |
| `--dry-run` or `--dry-run=true` | Dry-run |
| `--dry-run=false` | Inference enabled |

`--dry-run True` and `--dry-run False` also work. The previous `--no-dry-run`
flag remains an alias for `--dry-run=false`; supplying both options is rejected.
Dry-run submissions still create backfill records and queue tasks; the
orchestrator stops before inference. HTTP requests and direct runner calls also
default to `dry_run: true`; explicitly pass the JSON boolean `false` to process.
`--list-only` returns IDs without creating triggers or tasks. Combine it with
`--overwrite` to preview the selection without a database connection. To rerun
inference, use `--overwrite --dry-run=false`.

```bash
# Preview a full UTC day locally through EODAG/GEODES.
python -m scripts.historical_run --local --stage test --env-file notebooks/.env \
  --date-start 2025-01-01 --date-end 2025-01-02 \
  --list-only

# Use a local Cloud SQL proxy on port 5434 with the selected stack's credentials.
python -m scripts.historical_run --local --stage test --env-file notebooks/.env \
  --db-host 127.0.0.1 --db-port 5434 \
  --date-start 2025-01-01 --date-end 2025-01-02 \
  --list-only

# Queue a month of dry-run tasks for the selected modes (the default is dry-run).
python -m scripts.historical_run --local --stage test --env-file notebooks/.env \
  --date-start 2025-01-01 --date-end 2025-02-01 \
  --mode EW --mode IW --mode SM

# Enable inference explicitly for a CSV with a scene_id column.
python -m scripts.historical_run --local --stage test --env-file notebooks/.env scenes \
  --csv notebooks/scene_ids.csv --dry-run=false --manual-trigger
```

`python scripts/historical_run.py` supports the same options. EODAG uses time
boundaries: for a day-long search, pass that day's date and the following date.
Polarization must include VV (SV `VV` or DV `VV VH`), independently of mode
selection. Only high (`GRDH`) and medium (`GRDM`) resolution scenes are selected;
full resolution (`GRDF`) is excluded. These resolution, polarization, and ocean
filters always apply to date-range discovery.

Local `--env-file` loads the notebook's `STACK_SECRETS` configuration.
`--stage test`, `staging`, and `prod20240903` select its `TEST`, `STAGING`, and
`PRODUCTION` entries. With `STACK_SECRETS`, project/region default to
`cerulean-338116`/`europe-west1` (override with `GCPPROJECT`/`GCPREGION`). Pulumi
creates one historical queue per stack and exports its name as `historical_run_queue`.
The CLI, notebook, and deployed endpoint select the queue for their stack:

| Stage | Historical queue |
| --- | --- |
| TEST | `orchestrator-backlog-test` |
| STAGING | `orchestrator-backlog-staging` |
| PROD | `orchestrator-backlog-prod20240903` |

Each queue has its own dispatch limits and can be purged independently. A
[purge](https://docs.cloud.google.com/tasks/docs/reference/rest/v2/projects.locations.queues/purge)
applies to all historical runs in that stage's queue and can take up to a minute.
The earlier shared `orchestrator-backlog` queue can drain tasks submitted before
this change.
Missing or inaccessible queues stop submission. Legacy `QUEUE` and
`HISTORICAL_QUEUE` settings cannot redirect submissions.

Alternatively, export `DB_URL`, `GCPPROJECT`, `GCPREGION`,
`ORCHESTRATOR_URL`, and `API_KEY` directly. `HISTORICAL_DB_URL` overrides the DB URL
for a local proxy; `postgresql+asyncpg://` URLs are normalized for asyncpg.
With `--local`, `--db-host` and `--db-port` override only the corresponding
connection settings, preserving the selected URL's credentials, database name,
and other query parameters. Pass these options before the selector or search
options. They apply after `HISTORICAL_DB_URL` (when set), replacing matching
`host`/`port` query parameters so a Cloud SQL socket cannot override the explicit
TCP destination. Without these flags, existing socket and TCP connections keep
their current behavior. These options are rejected without `--local`.
Local previews need DB access unless using `--overwrite`. Queueing needs DB
access for run tracking and Google application default credentials with permission
to create tasks in the selected stage's queue.
No queue-list permission or queue-name secret is needed. The local path uses no
historical-run endpoint or Pulumi lookup once the stage's queue exists.

Local discovery runs once. Each run submits batches of up to 100 tasks through
Cloud Tasks' `tasks:batchCreate` API with up to **32 operations in flight**. Set
`HISTORICAL_BATCH_WORKERS` to an integer from 1 to 32 to lower submission concurrency.
The local CLI rechecks scene history before each batch by default, with up to four
of those checks in flight. All batches share one `HISTORICAL_RUN` trigger.
On a failed or partial batch, the runner stops launching more batches and lets
in-flight operations finish reporting their outcomes. It does not retry submissions
automatically. Tasks carry the trigger ID and a 30-minute execution deadline.
`manual_trigger: true` gives
source-association tasks run-specific names. Queue submission success does not
mean inference has completed. Local stdout contains one final JSON object with
run IDs and totals. Progress and batch confirmations go to stderr.

Local enqueue records the run in the `historical_run` table: selection, dry-run
setting, redacted command, Git revision, start and end times, confirmed task count,
success, and error. The row is inserted before discovery, so a killed run keeps a
`NULL` end time and a missing table stops submission. Failed submissions leave the
task total unknown (`NULL`). List-only runs create no history row.

HTTP submission and discovery share a 450-second budget from function entry.
Local batches have separate 450-second queue-submission budgets, with no overall
function deadline on local discovery. HTTP requests default to dry-run
when omitted and record a cloud function log link. Local runs print
progress locally and leave the trigger log link empty.

`notebooks/TriggerOnDemand.ipynb` uses the same local runner. Choose one scene
source: EODAG date/region search, an explicit list, or a CSV. EODAG applies catalog,
H/M resolution, and ocean filters; lists and CSVs provide already selected IDs.
Common options support an additional exclusion CSV and deduplication. CSV input/exclusion accepts
`scene_id` or legacy `s1_scene_id`. Setup uses `STACK_SECRETS`, `GCP_PROJECT`, and
`GCP_REGION`; rerun it after changing stacks. The orchestration cell checks the DB
and passes the selected scenes to the same concurrent runner. Set `OVERWRITE = True`
to skip the scene history lookup. `DRY_RUN` must be `True` or `False` (default:
`True`). All batches share one trigger and confirmations are printed as they arrive.
The ASA and imagery download utilities have their own execution paths.

### Historical run diagnostics

`scripts/historical_run_diagnostics.py` reports how a historical run is going. It
only reads: the database session is opened read-only, and the GCP calls are lookups.

```sh
# Latest non-dry-run historical run on the test stack.
python scripts/historical_run_diagnostics.py --stage test --env-file notebooks/.env

# A specific run, saved as JSON, with the baseline taken from another database.
python scripts/historical_run_diagnostics.py --stage test --historical-run-id 12 \
  --baseline-db-url "$PROD_DB_URL" --json run12.json

# Database sections only (no Google credentials needed).
python scripts/historical_run_diagnostics.py --stage test --skip-gcp
```

The database URL comes from `--db-url`, then `STACK_SECRETS` (same stage keys as
`historical_run.py`), then `HISTORICAL_DB_URL` or `DB_URL`. GCP lookups use Google
application default credentials and `GCPPROJECT`/`GCPREGION` (defaults
`cerulean-338116`/`europe-west1`). A run is selected by `--historical-run-id` or
`--trigger-id` (both repeatable); with neither, the latest non-dry-run
`historical_run` row is used. Queue lookups default to the selected stage's
historical queue. Use `--queue orchestrator-backlog` to inspect a run submitted
to the earlier shared queue.

The report covers:

1. **Run**: IDs, scene dates, scenes found by EODAG, tasks queued, the git hash the
   run was submitted from, submission start.
2. **Orchestrator**: scenes succeeded, failed, crashed, and in progress, counted per
   scene because each retry adds an `orchestrator_run` row; queued scenes that never
   reached the database (antimeridian, no ocean tiles, dry run); backlog queue depth;
   earliest start and latest end time. Verdict `RUNNING`, `COMPLETE`, or `IDLE` when
   the queue was not checked.
3. **ASA**: queue depth, slicks reached and pending, source associations by type,
   and slicks reached with no source. ASA records no run of its own, so a slick
   counts as reached once its `geometric_slick_potential` is set or it has a source.
   `STALLED` means the queue is empty while slicks are still pending.
4. **Runtime**: submission start to the last slick, orchestrator end, or source
   association; provisional until both verdicts are `COMPLETE`.
5. **Benchmarks**: slicks per scene and sources per slick against live
   (`SNS_TOPIC`) scenes from the past year (`--baseline-start`/`--baseline-end`),
   flagged beyond `--tolerance-pct` (default 50). Baselines under 100 scenes are
   shown but not graded.
6. **Bottlenecks**: peak orchestrator, inference, and ASA instance counts against
   their configured limits, peak database connections against `max_connections`,
   and peak backlog depth, each with the time of the peak. Cloud Monitoring keeps
   these metrics for about six weeks, so save `--json` output for the record.

Queue depths can include other runs in the same stage. ASA tasks are narrowed to
the run by the scene date in the task name when the queue holds 5,000 tasks or fewer.

## ASA batch runs

INFRA source association runs right after inference. VESSEL and DARK run in a daily
batch at 15:00 UTC, 3 h after Global Fishing Watch publishes AIS. A second Cloud
Scheduler job handles historical backfills at 30 minutes past each hour. Both
jobs POST to the ASA Cloud Function using the same authorized service account.
Each deploy pauses staging's daily job. The backfill job starts paused on every stack.

At 18:30 UTC, after the daily job's last possible retry, the `cf-alerts` function
reads that job's last attempt and posts to the pipeline alerts Slack channel if it
failed or the day's run never started.

### What you send

The daily job sends every option at its default:

```json
{"batch": true, "lookback_days": 32, "pivot": null, "retry_incomplete": false}
```

A one-off run reaching older scenes changes some of them, for example:

```json
{"batch": true, "lookback_days": 1500, "pivot": "2024-06-01", "retry_incomplete": true}
```

| Field | Meaning |
|---|---|
| `batch` | Required, `true`: a batch run, not a single scene. |
| `lookback_days` | Whole days back from now to look for scenes. Default 32, the daily run. Above 32 makes a one-off run and allows a `pivot`. |
| `pivot` | A UTC date, `"YYYY-MM-DD"`. The run lines up scenes oldest to newest, starts at that date's midnight, goes forward, then wraps around to the oldest. It must fall between the lookback start and the publication cutoff, about 3½ days before the run. Default: a random start. |
| `retry_incomplete` | Set to `true` to requeue eligible unfinished work immediately, including after a purge. Each force run creates fresh task names, so it can duplicate work still queued or running. With the default `false`, the daily batch waits one hour before retrying and a one-off batch skips previously queued runs. |

A bad value gets HTTP 400 naming the problem, for example
`pivot must fall from <first date> to <last date>`.

### Working through a backlog

1. Open Cloud Scheduler in project `cerulean-338116`, region `europe-west1`, and
   select the job whose name starts with `cerulean-cloud-<stack>-cf-asa-backfill-scheduler`.
2. Check the request body. The default lookback is 2,557 days (about seven years).
   Choose a lookback longer than 32 days that still covers the oldest intended
   capture at the expected finish date. Keep `retry_incomplete` false.
3. Click **Resume** once. Each hourly invocation skips previously queued runs.
   Each batch considers up to 1,000 runs and limits uncached VESSEL work to 30 AIS dates.
4. Click **Pause** when you want to stop the hourly runs. Tasks already queued
   continue processing.

Both jobs take the same database lock before selecting work. A busy hourly job
skips that tick. The daily job returns HTTP 503 on contention, and Scheduler
retries failed daily requests up to four times at ten-minute intervals. CI
refreshes Pulumi state and preserves the backfill job's body and pause setting.

The function logs `Created task ...` for each scene queued,
`Scene <id>: ... failed` for a scene that failed in the batch, and
`N scenes wait for GFW to publish their AIS` for scenes past GFW's published data.
Each queued run gets `orchestrator_run.asa_batch_queued_at`.

## Adding a new Pulumi Stack
Don't forget to edit the following places:
- Historical-run discovers its URL from the selected stack’s `historical_run_url` output.
- TODO add other places/steps
- create a new git branch with the same name
- copy changes in Git hash 7f1dcda and b55e6c7 (but use more modern yamls as base), commit and push
- if commit fails due to pre-commit, review files and accept changes, retry
- go to Git Actions, and manually run Test and Deploy on the new branch (takes about 25 minutes)

If you are getting this error on test&deploy, then you should try to pulumi refresh and run the github action again.
`Error 409: Conflict for resource 'cerulean-cloud-test-cr-tipg': version '1690549980909798' was specified but current version is '1690554168522592'`

## Updating the trained model
If you are going to deploy a new scripted model, first save it as a tracing model using the function "save_icevision_model_state_dict_and_tracing" in the cerulean-ml repo.
Then, upload the experiment folder and contents to the GCP `ceruleanml` bucket.
Update the value of the Pulumi pamater `cerulean-cloud-images:weights_name` found in your local version of 
`cerulean-cloud/images/stack_config/Pulumi.STACK_NAME_OF_INTEREST.yaml` to match the experiment naming you just uploaded. 
--- You must then push these changes to the git repo. ---
Finally, do the following stsps:
1. Go to `https://github.com/SkyTruth/cerulean-cloud/actions` 
2. Then click "Test and Deploy"
3. Click on "Run Workflow" and choose either the local branch you are working on
4. Click "Run Workflow" to kick off the model upload

## Troubleshooting

If pulumi throws funky errors at deployment, you can run in your current stack:
```sh
pulumi refresh
```

If you need to completely restart your stack, these steps have been found to work:
'''sh
pulumi destroy
(use the GUI delete database)
pulumi refresh
pulumi state delete {URN of any sticky resources}
pulumi refresh
pulumi destroy
(If there is a lock on the stack, you can delete that lock in gs://cerulean-cloud-state/cerulean-cloud-images)
'''

If you get the following error, make sure you have docker running in your machine:
```sh
Exception: invoke of docker:index/getRegistryImage:getRegistryImage failed: invocation of docker:index/getRegistryImage:getRegistryImage returned an error: 1 error occurred:
        * Error pinging Docker server: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
    error: an unhandled error occurred: Program exited with non-zero exit code: 1
```

The local Compose database uses PostgreSQL 18 with PostGIS and stores its data
in the `db18-data` volume. To run the database migrations locally:
1. Launch Docker on your machine
2. Run `docker compose up --build` in the folder where `alembic.ini` is
3. Run 'DB_URL=postgresql://user:password@localhost:5432/db alembic upgrade head' (or downgrade base) in the folder where the alembic.ini is (you may have to activate an environment and install dependencies, like tempdbenv)
4. To connect with pgAdmin, add a new server with these properties:
 - Name: local
 - Host name: localhost
 - Port: 5432
 - Username: user
 - Password: password

If you have a container from the old PostgreSQL 14 setup, run
`docker compose rm --stop --force db` before starting PostgreSQL 18. This
removes the container and keeps its old data volume.

## Human in the loop (HITL) workflows using SQL

In general the HITL process is as follows in all cases:

- Select a target slick id/s that is intended to be changed;
- Update the slick "active" field to False;
- Select all the "stable" values from the slick id/s;
- Save these values along with the changed values using `INSERT INTO` (when we have multiple slicks in the mix make sure to use aggregation functions);
- Ensure to set slick column to the original slick/slicks ids to keep the audit of changes across time.
- Ensure to set active column in the new slick/slicks to True.

You can then use the function `slick_history` with a given slick_id to inspect the changes that occurred in the slick.

All these examples I have been run in the test environment so you can inspect the results with the `slick_history` query there.

As a side note, ideally this process would be intermediated by the streamlit app that we had envisioned - these queries could all be encoded in a module that interacts with a frontend, to make it really use to run these common HITL changes to the slick table.
### Validate Slick Class (Include Confidence Level)
```sql
UPDATE slick SET active=False WHERE id=34913;
INSERT INTO slick (
  slick_timestamp,
	geometry,
	machine_confidence,
  human_confidence,
	active,
	validated,
	slick,
	notes,
	meta,
	orchestrator_run,
	slick_class
    )
SELECT
	slick_timestamp,
	geometry,
  machine_confidence,
	0.9,
	True,
	True,
	'{ 34913 }',
	notes,
	meta,
	orchestrator_run,
	slick_class
FROM slick WHERE id=34913
RETURNING id;
```
```sql
SELECT * FROM slick_history(34927)
```

### Change slick class
```sql
UPDATE slick SET active=False WHERE id=34927;
INSERT INTO slick (
  slick_timestamp,
	geometry,
	machine_confidence,
  human_confidence,
	active,
	validated,
	slick,
	notes,
	meta,
	orchestrator_run,
	slick_class
    )
SELECT
	slick_timestamp,
	geometry,
  machine_confidence,
	0.9,
	True,
	True,
	'{ 34927 }',
	notes,
	meta,
	orchestrator_run,
	3
FROM slick WHERE id=34927
RETURNING id;
```
```sql
SELECT * FROM slick_history(34930)
```

### Combine two slicks into one
```sql
UPDATE slick SET active=False WHERE id IN (34817, 34816);
INSERT INTO slick (
  slick_timestamp,
	geometry,
	machine_confidence,
  human_confidence,
	active,
	validated,
	slick,
	notes,
	meta,
	orchestrator_run,
	slick_class
    )
SELECT
	MAX(slick_timestamp),
	ST_Union(geometry::geometry),
  MIN(machine_confidence),
	0.9,
	True,
	True,
	'{ 34817, 34816 }',
	string_agg(notes, ','),
	jsonb_agg(meta),
	MAX(orchestrator_run),
	2
FROM slick WHERE id IN (34817, 34816)
RETURNING id;
```
```sql
SELECT * FROM slick_history(34931)
```

### Break one slick into two (or n slicks)
```sql
UPDATE slick SET active=False WHERE id IN (34817);
INSERT INTO slick (
  slick_timestamp,
	geometry,
	machine_confidence,
  human_confidence,
	active,
	validated,
	slick,
	notes,
	meta,
	orchestrator_run,
	slick_class
    )
SELECT
	slick_timestamp,
	ST_Multi(ST_Subdivide(geometry::geometry)),
  machine_confidence,
	0.9,
	True,
	True,
	'{ 34817 }',
	notes,
	meta,
	orchestrator_run,
	3
FROM slick WHERE id IN (34817)
RETURNING id; # Returns 3 records
```
```sql
SELECT * FROM slick_history(34933)
SELECT * FROM slick_history(34934)
SELECT * FROM slick_history(34935)
```

### Add note field to slick (e.g. record some action taken, or a reference ID from NOAA, ...)
```sql
UPDATE slick SET active=False WHERE id=34935;
INSERT INTO slick (
  slick_timestamp,
	geometry,
	machine_confidence,
  human_confidence,
	active,
	validated,
	slick,
	notes,
	meta,
	orchestrator_run,
	slick_class
    )
SELECT
	slick_timestamp,
	geometry,
  machine_confidence,
	0.9,
	True,
	True,
	'{ 34935 }',
	'This is a funny slick!',
	meta,
	orchestrator_run,
	slick_class
FROM slick WHERE id= 34935
RETURNING id;
```
```sql
SELECT * FROM slick_history(34936)
```
