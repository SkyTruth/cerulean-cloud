"""Infra for AOI backfill Cloud Run service."""

import cloud_run_images
import git
import pulumi
import pulumi_gcp as gcp
from cloud_run_infer import noauth_iam_policy_data
from database import instance, sql_instance_url
from utils import construct_name

stack = pulumi.get_stack()

repo = git.Repo(search_parent_directories=True)
git_sha = repo.head.object.hexsha

service_account = gcp.serviceaccount.Account(
    construct_name("cr-aoi-backfill"),
    account_id=f"{stack}-cr-aoi-backfill",
    display_name="Service Account for AOI backfill cloud run.",
)

gcp.projects.IAMMember(
    construct_name("cr-aoi-backfill-cloudSqlClient"),
    project=pulumi.Config("gcp").require("project"),
    role="roles/cloudsql.client",
    member=service_account.email.apply(lambda email: f"serviceAccount:{email}"),
)

gcp.projects.IAMMember(
    construct_name("cr-aoi-backfill-secretmanagerSecretAccessor"),
    project=pulumi.Config("gcp").require("project"),
    role="roles/secretmanager.secretAccessor",
    member=service_account.email.apply(lambda email: f"serviceAccount:{email}"),
)

gcp.projects.IAMMember(
    construct_name("cr-aoi-backfill-storageObjectViewer"),
    project=pulumi.Config("gcp").require("project"),
    role="roles/storage.objectViewer",
    member=service_account.email.apply(lambda email: f"serviceAccount:{email}"),
)

secret_accessor_binding = gcp.secretmanager.SecretIamMember(
    construct_name("cr-aoi-backfill-secret-accessor-binding"),
    secret_id=pulumi.Config("cerulean-cloud").require("keyname"),
    role="roles/secretmanager.secretAccessor",
    member=pulumi.Output.concat("serviceAccount:", service_account.email),
    opts=pulumi.ResourceOptions(depends_on=[service_account]),
)

service_name = construct_name("cr-aoi-backfill")
default = gcp.cloudrun.Service(
    service_name,
    opts=pulumi.ResourceOptions(depends_on=[secret_accessor_binding]),
    name=service_name,
    location=pulumi.Config("gcp").require("region"),
    autogenerate_revision_name=True,
    template=gcp.cloudrun.ServiceTemplateArgs(
        spec=gcp.cloudrun.ServiceTemplateSpecArgs(
            service_account_name=service_account.email,
            containers=[
                gcp.cloudrun.ServiceTemplateSpecContainerArgs(
                    image=cloud_run_images.cloud_run_aoi_backfill_image.name,
                    envs=[
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="UVICORN_PORT",
                            value="8080",
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="DB_URL",
                            value=sql_instance_url,
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="PROJECT_ID",
                            value=pulumi.Config("gcp").require("project"),
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="GOOGLE_CLOUD_PROJECT",
                            value=pulumi.Config("gcp").require("project"),
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="GIT_HASH",
                            value=git_sha,
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="API_KEY",
                            value_from=gcp.cloudrun.ServiceTemplateSpecContainerEnvValueFromArgs(
                                secret_key_ref=gcp.cloudrun.ServiceTemplateSpecContainerEnvValueFromSecretKeyRefArgs(
                                    name=pulumi.Config("cerulean-cloud").require(
                                        "keyname"
                                    ),
                                    key="latest",
                                )
                            ),
                        ),
                    ],
                    resources=dict(limits=dict(memory="8Gi", cpu="2000m")),
                ),
            ],
            timeout_seconds=3600,
            container_concurrency=1,
        ),
        metadata=dict(
            annotations={
                "run.googleapis.com/cloudsql-instances": instance.connection_name,
                "autoscaling.knative.dev/maxScale": "3",
            },
        ),
    ),
    traffics=[
        gcp.cloudrun.ServiceTrafficArgs(
            percent=100,
            latest_revision=True,
        )
    ],
)

noauth_iam_policy = gcp.cloudrun.IamPolicy(
    construct_name("cr-noauth-iam-policy-aoi-backfill"),
    location=default.location,
    project=default.project,
    service=default.name,
    policy_data=noauth_iam_policy_data.policy_data,
)
