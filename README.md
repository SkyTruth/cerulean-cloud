# cerulean-cloud
Pulumi repository with infrastructure for Cerulean, including all cloud services and database structure.

## Setup cloud authentication
### GCS auth
```
gcloud config set account rodrigo@developmentseed.org
gcloud config configurations create cerulean --project cerulean-338116 --account rodrigo@developmentseed.org
gcloud config configurations activate cerulean
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