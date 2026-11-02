# Cerulean API Documentation
## Introduction
For most users, we recommend using the [Cerulean web application](https://cerulean.skytruth.org/), which provides a visual interface for exploring the complete set of Cerulean data layers.

For users who want to directly access and download oil slick detection data, we provide programmatic free access to an [OGC compliant API](api.cerulean.skytruth.org). 

We strongly recommend interested users to look at our [developer’s guide](https://colab.research.google.com/drive/1SYWIzPH_2NeqvfqcVzY43y1hv1Ug-YoJ#scrollTo=NW_mI607d6Jq) for some examples of how to use the API.


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

If you get an error while running tests mentioning that `psycopg` is not installed run:
```sh
pip install "psycopg[binary]"
```
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

### Selecting one database target for backend services and the UI

Each stack defaults to its managed instance and database. To connect to an
existing instance and an existing database inside it, set:

```sh
pulumi config set db:connection-instance CLONE_INSTANCE_NAME --stack prod20240903
pulumi config set db:connection-database DATABASE_NAME --stack prod20240903
```

Database names use ASCII letters, digits, underscores, and hyphens so every
connection driver interprets the name identically. Unsupported names are rejected
before any infrastructure resource is registered.

These are connection settings. They do not create or rename the selected database,
change the original managed instance/database/user, or upgrade PostgreSQL. The
selected database must already exist and grant access to the existing stack user
and password. The UI continues to use the account selected by its `DB_SECRET_NAME`;
that account also needs access to the selected database. Either connection setting
can be omitted to use its original default independently.

The selected target supplies all five backend consumers (orchestrator, TiPG, scene
relevancy, historical runs, and ASA), their SQL attachments, and the migration
proxy/connection URL. Pulumi exports `database_target`, a nonsecret object with
`project`, `region`, `connection_name`, `private_ip`, `database`, and `port`.
Credentials remain in their existing secret stores. The existing connection URL
outputs remain available for clients that need their driver-specific format.

Changing config alone does not change running services. Review `pulumi preview`:
the original instance, database, and user must remain unchanged, and the selected
instance should only be read. Deploy through the normal release process. PROD
uses the `prod20240903` stack and a tagged release. The deploy still runs Alembic
against the selected database, so review pending migrations before releasing.

#### Coordinated production cutover

There is no atomic switch across backend and UI deployments. Keep public traffic
and all writers paused from the final consistent clone/snapshot through backend
and UI verification. Pause ingestion, queues, scheduled/manual jobs, and UI writes
(including authentication/session writes); drain in-flight work. Confirm a backend
deployment will preserve those pauses. An older clone does not receive later
writes automatically. Freeze the production UI Cloud Build trigger during the
cutover so it cannot replace the selected image or revision concurrently.

Deploy the selected backend target first, then inspect the proposed UI changes:

```sh
python scripts/database_target.py plan --stack prod20240903
```

The helper reads `database_target` once from the deployed stack. It checks the
templates and revisions actually serving traffic, and includes both
`cerulean-ui-production` and `cerulean-ui-prod-preview` in `us-east4`. The preview
also uses production data. Split traffic is rejected so a partial target switch
cannot be mistaken for a completed deployment. `plan` is read-only and reports
differences without requiring the UIs to have switched yet.

While maintenance remains active, apply the same target to both UIs:

```sh
python scripts/database_target.py apply-ui --stack prod20240903 \
  --snapshot /tmp/cerulean-ui-before-cutover.json
python scripts/database_target.py verify --stack prod20240903
```

`apply-ui` first requires all five backends to select the target and confirms that
the database exists. It saves previous UI settings to a new, owner-readable file
without credentials. It preserves each UI's currently serving image, updates
`DB_HOST`, `DB_NAME`, `DB_PORT`, and the SQL attachment, and leaves credential
bindings and other service settings in place. Both new revisions must be ready
with the correct target before it routes 100% of traffic to their explicit
revision names. A failure leaves maintenance in effect; the helper does not
reopen traffic, resume writers, or automatically roll back. The two UI traffic
updates are sequential, so keep maintenance active throughout.

`verify` exits unsuccessfully if any backend or UI template or serving revision
selects a different target. This checks connection configuration, not database
connectivity or application behavior. Before reopening public traffic, verify
uncached TiPG/UI reads, login/session behavior, map queries, and a controlled HITL
write. Review DB-derived caches separately. Re-enable the UI build trigger and
resume writers only after validation succeeds. Ordinary UI image builds preserve
these service settings; they do not contain another database target definition.

Before resuming writes, rollback selects the previous instance/database in Pulumi,
deploys the backend, and runs the same `apply-ui`/`verify` commands with a new
snapshot filename. To return to the original defaults, remove both connection
settings. Resolve SQL attachments from that target rather than restoring stale
names from an old UI revision. After writes resume, reconciling new data is
required before any rollback to an older database.

For other stacks, pass each intended `--ui-service` explicitly. A different UI
region can be supplied with `--ui-region`. PROD always includes both production
UIs. These resource names identify consumers; the database endpoint and name
come exclusively from the stack output.

#### Local database tools

Start the installed Cloud SQL Auth Proxy against the selected target without a
hard-coded instance name:

```sh
python scripts/database_target.py proxy --stack prod20240903 --port 1234
pulumi stack output database_target --stack prod20240903 --json
```

Use the output's `database` field when configuring DBeaver, pgAdmin, or a notebook
against that local proxy. Use credentials for that client separately. Existing
saved client connections and shell `DB_URL` values are not rewritten by a deploy.

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
- Historical run Cloud Function
- Scene relevancy Cloud Function


## Adding a new Pulumi Stack
Don't forget to edit the following places:
- historical_run.py (add the stack name to stage options)
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

If you want to run tests on the database migrations locally, you can do the following steps:
1. Launch Docker on your machine
2. Run 'docker-compose up --build' in the folder where the alembic.ini is
3. Run 'DB_URL=postgresql://user:password@localhost:5432/db alembic upgrade head' (or downgrade base) in the folder where the alembic.ini is (you may have to activate an environment and install dependencies, like tempdbenv)
4. To connect with pgAdmin, add a new server with these properties:
 - Name: local
 - Host name: localhost
 - Port: 5432
 - Username: user
 - Password: password

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
