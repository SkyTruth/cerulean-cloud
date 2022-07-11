# cerulean-cloud
Pulumi repository with infrastructure for Cerulean, including all cloud services and database structure.

## Setup cloud authentication
### GCS auth
```
gcloud config set account rodrigo@developmentseed.org
gcloud config configurations create cerulean --project cerulean-338116 --account rodrigo@developmentseed.org
gcloud config configurations activate cerulean
```

Also, make sure to authenticate into docker with GCP to allow interaction with GCR:
```
gcloud auth configure-docker
```
### AWS auth
```
aws configure --profile cerulean
export AWS_PROFILE=cerulean
```

## Setup your python virtualenv
```
mkvirtualenv cerulean-cloud --python=$(which python3.8)
pip install -r requirements.txt
pip install -r requirements-test.txt
# Setup pre-commit
pre-commit install
```
## Install pulumi
```
brew install pulumi
```

And login to state management:
```
pulumi login gs://cerulean-cloud-state
```

## Check available stages
```
pulumi stack ls
```
Select another stage
```
pulumi stack select test
```

## Preview changes
```
pulumi preview
```

## Deploy changes
```
pulumi up
```

## Connect to database

In order to connect to the deployed database, you can use the [Cloud SQL proxy for authentication](https://cloud.google.com/sql/docs/mysql/connect-admin-proxy). First install the proxy in your local machine (instructions [here](https://cloud.google.com/sql/docs/mysql/connect-admin-proxy#install)).

You can then find the instance connection name and the connection string in the outputs of your active pulumi stack:
```
pulumi stack --show-secrets
# use `database_instance_name` in Cloud SQL proxy
# use `database_url_alembic` to connect in your client
```

Start the Cloud SQL proxy (make sure you are properly authenticated with GCP):
```
  ./cloud_sql_proxy -instances=database_instance_name=tcp:0.0.0.0:5432
```

In another process connect to the database (i.e. with `psql`):
```
psql database_url_alembic
```

## Troubleshooting

If pulumi throws funky errors at deployment, you can run in your current stack:
```
pulumi refresh
```