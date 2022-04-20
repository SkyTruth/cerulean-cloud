# cerulean-cloud
All cloud services including inference and database structure

## Setup cloud authentication
### GCS auth
```
gcloud config set account rodrigo@developmentseed.org
gcloud config configurations create cerulean --project cerulean-338116 --account rodrigo@developmentseed.org
gcloud config configurations activate cerulean
```
### AWS auth
```
export AWS_PROFILE=skytruth
```

## Run plan with workspace tfvars
```
terraform init
terraform workspace list
# you should see test, staging and production
```

```
terraform plan --var-file=workspaces/test.tfvars
```