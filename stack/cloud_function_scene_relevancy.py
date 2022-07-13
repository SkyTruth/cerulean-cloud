"""cloud function to select appropriate scenes (over water and IW) from SNS notification"""
import os
import time

import pulumi
from database import sql_instance_url
from pulumi_gcp import cloudfunctions, serviceaccount, storage
from utils import construct_name

stack = pulumi.get_stack()
# We will store the source code to the Cloud Function in a Google Cloud Storage bucket.
bucket = storage.Bucket(
    construct_name("bucket-cloud-function"),
    location="EU",
    labels={"pulumi": "true", "environment": pulumi.get_stack()},
)

config_values = {"DB_URL": sql_instance_url}

# The Cloud Function source code itself needs to be zipped up into an
# archive, which we create using the pulumi.AssetArchive primitive.
PATH_TO_SOURCE_CODE = "../cerulean_cloud/cloud_function_scene_relevancy"
assets = {}
for file in os.listdir(PATH_TO_SOURCE_CODE):
    location = os.path.join(PATH_TO_SOURCE_CODE, file)
    asset = pulumi.FileAsset(path=location)
    assets[file] = asset

archive = pulumi.AssetArchive(assets=assets)

# Create the single Cloud Storage object, which contains all of the function's
# source code. ("main.py" and "requirements.txt".)
source_archive_object = storage.BucketObject(
    construct_name("source-cloud-function-scene-relevancy"),
    name="handler.py-%f" % time.time(),
    bucket=bucket.name,
    source=archive,
)

# Assign access to cloud SQL
cloud_function_service_account = serviceaccount.Account(
    construct_name("cloud-function"),
    account_id=f"{stack}-cloud-function",
    display_name="Service Account for cloud function.",
)
cloud_function_service_account_iam = serviceaccount.IAMMember(
    construct_name("cloud-function-iam"),
    service_account_id=cloud_function_service_account.name,
    role="roles/cloudsql.client",
    member=cloud_function_service_account.email.apply(
        lambda email: f"serviceAccount:{email}"
    ),
)

fxn = cloudfunctions.Function(
    construct_name("cloud-function-scene-relevancy"),
    entry_point="main",
    environment_variables=config_values,
    region=pulumi.Config("gcp").require("region"),
    runtime="python38",
    source_archive_bucket=bucket.name,
    source_archive_object=source_archive_object.name,
    trigger_http=True,
    service_account_email=cloud_function_service_account.email,
)

invoker = cloudfunctions.FunctionIamMember(
    construct_name("cloud-function-scene-relevancy-invoker"),
    project=fxn.project,
    region=fxn.region,
    cloud_function=fxn.name,
    role="roles/cloudfunctions.invoker",
    member="allUsers",
)
