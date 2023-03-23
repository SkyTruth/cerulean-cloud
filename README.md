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

If you are deploying a completely new stack, make sure to create matching configuration files in `cerulean-cloud-images` and `cerulean-cloud`, with matching stack names. In addition, specifically for the tifeatures deployment, since the database is empty when a stack is deployed for the first time (alembic migrations occur after the initial migration), if you want to access `tifeatures` after this initial deployment make sure to poll the `/register` endpoint of the resulting URL in order to correctly load the tables (i.e. `curl https://some-tifeatures-url.app/register`). For any deployments after the first one, this is not required.

__Decreasing cold starts for Cloud Run__

In order to decrease response time for Cloud Run (especially in the production services, since for test and staging this will increase costs) you can set the minimal instances to 1 to run at any given moment in time (see [documentation](https://cloud.google.com/run/docs/tips/general#:~:text=Using%20minimum%20instances%20to%20reduce%20cold%20starts,-You%20can%20configure&text=You%20should%20set%20a%20number,for%20longer%20than%2010%20seconds.)). Due to a `pulumi` limitation, when the service is first created this property cannot be set so we advise to set this value manually once the deployment has been completed.

__Metrics__

Google Cloud provides nice dashboards for tracking the stability, response time and resource consumption of the cloud resources. The links below point to the PRODUCTION deployment, but for every stack's components you'll find similar dashboards:
- [Cloud Run orchestrator metrics](https://console.cloud.google.com/run/detail/europe-west1/cerulean-cloud-production-cloud-run-orchestrator/metrics?project=cerulean-338116)
- [Cloud Run offset tile metrics](https://console.cloud.google.com/run/detail/europe-west1/cerulean-cloud-production-cloud-run-offset-tiles/metrics?project=cerulean-338116)
- [Cloud Tasks queue metrics](https://console.cloud.google.com/cloudtasks/queue/europe-west1/cerulean-cloud-production-queue-cloud-run-orchestrator-d8ed51e/metrics?project=cerulean-338116)
- [Cloud Run tifeatures metrics](https://console.cloud.google.com/run/detail/europe-west1/cerulean-cloud-production-cloud-run-tifeatures/metrics?project=cerulean-338116)
- [Cloud Function scene relevancy metrics](https://console.cloud.google.com/functions/details/europe-west1/cerulean-cloud-production-cloud-function-scene-relevancy?env=gen1&project=cerulean-338116)
- [Cloud Function historical run metrics](https://console.cloud.google.com/functions/details/europe-west1/cerulean-cloud-production-cloud-function-historical-run?env=gen1&project=cerulean-338116)

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
export VIRTUALENVWRAPPER_PYTHON=/Users/jonathanraphael/mambaforge/bin/python # the path to the python installation where you installed mkvirtualenv
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel
source /usr/local/bin/virtualenvwrapper.sh # this path can vary depending on your installation
```

Then you'll be able to run:
```sh
mkvirtualenv cerulean-cloud --python=$(which python3.8)
pip install -r requirements.txt
pip install -r requirements-test.txt
# Additional requirements files
pip install -r cerulean_cloud/cloud_run_offset_tiles/requirements.txt
pip install -r cerulean_cloud/cloud_run_orchestrator/requirements.txt
pip install -r cerulean_cloud/cloud_run_tifeatures/requirements.txt
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
pytest test/test_cerulean_cloud/test_tiling.py::test_from_base_tiles_create_offset_tiles # run only a specific test
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
docker:index:RemoteImage cerulean-cloud-images-test-remote-tifeatures  completing deletion from previous update
-  docker:index:RemoteImage cerulean-cloud-images-test-remote-offset delete completing deletion from previous update
-  docker:index:RemoteImage cerulean-cloud-images-test-remote-tifeatures delete completing deletion from previous update
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

## Database

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
- [Adding columns to existing tables](alembic/versions/0a4536575154_add_simplified_eez_geom.py)
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
URL = "https://cerulean-cloud-test-cloud-run-orchestrator-5qkjkyomta-ew.a.run.app"
API_KEY= "SOME_API_KEY"
orchestrator_result = httpx.post(URL+"/orchestrate",
                                json={"sceneid": "S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF"},
                                timeout=None,
                                headers={"Authorization": f"Bearer {API_KEY}"})
```

The services deployed by cerulean-cloud that DO NOT require this API key are:
- Tifeatures Cloud Run
- Historical run Cloud Function
- Scene relevancy Cloud Function

## Troubleshooting

If pulumi throws funky errors at deployment, you can run in your current stack:
```sh
pulumi refresh
```

If you get the following error, make sure you have docker running in your machine:
```sh
Exception: invoke of docker:index/getRegistryImage:getRegistryImage failed: invocation of docker:index/getRegistryImage:getRegistryImage returned an error: 1 error occurred:
        * Error pinging Docker server: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
    error: an unhandled error occurred: Program exited with non-zero exit code: 1
```

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
