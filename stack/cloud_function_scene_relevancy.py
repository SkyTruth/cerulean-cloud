"""cloud function to select appropriate scenes (over water and IW) from SNS notification"""
import os
import time

import pulumi
import pulumi_gcp as gcp
from pulumi_gcp import cloudfunctions, storage
from utils import construct_name

# We will store the source code to the Cloud Function in a Google Cloud Storage bucket.
bucket = storage.Bucket(
    construct_name("bucket-cloud-function"),
    location="EU",
    labels={"pulumi": "true", "environment": pulumi.get_stack()},
)

config_values = {"DESTINATION": "abc"}

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

fxn = cloudfunctions.Function(
    construct_name("cloud-function-scene-relevancy"),
    entry_point="handler",
    environment_variables=config_values,
    region=pulumi.Config("gcp").require("region"),
    runtime="python38",
    source_archive_bucket=bucket.name,
    source_archive_object=source_archive_object.name,
    trigger_http=True,
)

policy_data = gcp.organizations.get_iam_policy(
    bindings=[
        gcp.organizations.GetIAMPolicyBindingArgs(
            role="roles/cloudsql.client",
            members=["domain:appspot.gserviceaccount.com"],
        )
    ]
)
sql_access = cloudfunctions.FunctionIamPolicy(
    construct_name("cloud-function-scene-relevancy-invoker"),
    project=fxn.project,
    region=fxn.region,
    cloud_function=fxn.name,
    policy_data=policy_data.policy_data,
)

invoker = cloudfunctions.FunctionIamMember(
    construct_name("cloud-function-scene-relevancy-invoker"),
    project=fxn.project,
    region=fxn.region,
    cloud_function=fxn.name,
    role="roles/cloudfunctions.invoker",
    member="allUsers",
)
