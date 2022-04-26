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

## Install pulumi
```
brew install pulumi
```

And login to state management:
```
pulumi login gs://cerulean-cloud-state
```

## Check available stage
```
pulumi stack ls
```
Select another stage
```
pulumi stack select test
```

## Deploy
```
pulumi up
```
