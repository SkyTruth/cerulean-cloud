# cerulean-cloud
All cloud services including inference and database structure

## GCS auth
```
gcloud config set account rodrigo@developmentseed.org
gcloud config configurations create cerulean --project cerulean-338116 --account rodrigo@developmentseed.org
gcloud config configurations activate cerulean
```

## AWS auth
```
export AWS_PROFILE=skytruth
```