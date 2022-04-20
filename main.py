#!/usr/bin/env python
from constructs import Construct
from cdktf import App, TerraformStack, GcsBackend, TerraformOutput, TerraformHclModule
from cdktf_cdktf_provider_aws import AwsProvider
from cdktf_cdktf_provider_google import GoogleProvider

from imports.example import Example


class MyStack(TerraformStack):
    def __init__(self, scope: Construct, ns: str):
        super().__init__(scope, ns)

        workspace = "test"
        project = "cerulean-338116"
        region = "europe-west1"

        aws = AwsProvider(self, "AWS")
        google = GoogleProvider(
            self, "Google", project=project, region=region, zone="europe-west1-b")

        # define resources here
        bucket = Example(self, "example",
            bucket_name=f"{project}-{workspace}-a-bucket", region=region, tags={"terraform": "true", "environment": workspace})

        TerraformOutput(self, "url",
                        value=bucket.domain_output,
                        )


app = App()
stack = MyStack(app, "cerulean-cloud-test")

GcsBackend(stack,
           bucket="cerulean-cloud-state",
           prefix="cerulean-cloud/"
           )

app.synth()
