"""infra for cloud run function for orchestration
Reference doc: https://www.pulumi.com/blog/build-publish-containers-iac/
"""
import os

import cloud_function_ais_analysis
import cloud_run_images
import cloud_run_offset_tile
import git
import pulumi
import pulumi_gcp as gcp
import titiler_sentinel
from database import instance, sql_instance_url_with_asyncpg
from utils import construct_name

config = pulumi.Config()
stack = pulumi.get_stack()

repo = git.Repo(search_parent_directories=True)
git_sha = repo.head.object.hexsha
git_tag = next((tag.name for tag in repo.tags if tag.commit == repo.head.commit), None)


# Assign access to cloud SQL
cloud_function_service_account = gcp.serviceaccount.Account(
    construct_name("cloud-run-orchestrator"),
    account_id=f"{stack}-cloud-run-orch",
    display_name="Service Account for cloud run.",
)

cloud_function_service_account_iam = gcp.projects.IAMMember(
    construct_name("cloud-run-orchestrator-cloudTasksEnqueuer"),
    project=pulumi.Config("gcp").require("project"),
    role="roles/cloudtasks.enqueuer",
    member=cloud_function_service_account.email.apply(
        lambda email: f"serviceAccount:{email}"
    ),
)

cloud_function_service_account_iam = gcp.projects.IAMMember(
    construct_name("cloud-run-orchestrator-cloudSqlClient"),
    project=pulumi.Config("gcp").require("project"),
    role="roles/cloudsql.client",
    member=cloud_function_service_account.email.apply(
        lambda email: f"serviceAccount:{email}"
    ),
)

cloud_function_service_account_iam = gcp.projects.IAMMember(
    construct_name("cloud-run-orchestrator-secretmanagerSecretAccessor"),
    project=pulumi.Config("gcp").require("project"),
    role="roles/secretmanager.secretAccessor",
    member=cloud_function_service_account.email.apply(
        lambda email: f"serviceAccount:{email}"
    ),
)

# IAM Binding for Secret Manager access
secret_accessor_binding = gcp.secretmanager.SecretIamMember(
    construct_name("cloud-run-orchestrator-secret-accessor-binding"),
    secret_id=pulumi.Config("cerulean-cloud").require("keyname"),
    role="roles/secretmanager.secretAccessor",
    member=pulumi.Output.concat(
        "serviceAccount:", cloud_function_service_account.email
    ),
    opts=pulumi.ResourceOptions(depends_on=[cloud_function_service_account]),
)


service_name = construct_name("cloud-run-orchestrator")
default = gcp.cloudrun.Service(
    service_name,
    name=service_name,
    location=pulumi.Config("gcp").require("region"),
    template=gcp.cloudrun.ServiceTemplateArgs(
        spec=gcp.cloudrun.ServiceTemplateSpecArgs(
            service_account_name=cloud_function_service_account.email,
            containers=[
                gcp.cloudrun.ServiceTemplateSpecContainerArgs(
                    image=cloud_run_images.cloud_run_orchestrator_image.name,
                    envs=[
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="DB_URL",
                            value=sql_instance_url_with_asyncpg,
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TITILER_URL",
                            value=titiler_sentinel.lambda_api.api_endpoint.apply(
                                lambda api_endpoint: api_endpoint
                            ),
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="INFERENCE_URL",
                            value=cloud_run_offset_tile.default.statuses.apply(
                                lambda statuses: statuses[0].url
                            ),
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="GIT_HASH",
                            value=git_sha,
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="GIT_TAG",
                            value=git_tag,
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="MODEL",
                            value=os.getenv("MODEL"),
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="CLOUD_RUN_NAME",
                            value=service_name,
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="PROJECT_ID",
                            value=pulumi.Config("gcp").require("project"),
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="GCP_REGION",
                            value=pulumi.Config("gcp").require("region"),
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TITILER_API_KEY",
                            value_from=gcp.cloudrun.ServiceTemplateSpecContainerEnvValueFromArgs(
                                secret_key_ref=gcp.cloudrun.ServiceTemplateSpecContainerEnvValueFromSecretKeyRefArgs(
                                    name=pulumi.Config("cerulean-cloud").require(
                                        "titiler_keyname"
                                    ),
                                    key="latest",
                                )
                            ),
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
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="AAA_QUEUE",
                            value=cloud_function_ais_analysis.queue.name,
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="AIS_IS_DRY_RUN",
                            value=pulumi.Config("cerulean-cloud").require("dryrun_ais"),
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="FUNCTION_URL",
                            value=cloud_function_ais_analysis.fxn.https_trigger_url,
                        ),
                    ],
                    resources=dict(limits=dict(memory="4Gi", cpu="1000m")),
                ),
            ],
            timeout_seconds=3540,
            container_concurrency=1,
        ),
        metadata=dict(
            name=service_name + "-" + cloud_run_images.cloud_run_orchestrator_sha,
            annotations={
                "run.googleapis.com/cloudsql-instances": instance.connection_name,
                "autoscaling.knative.dev/maxScale": "45",
            },
        ),
    ),
    metadata=gcp.cloudrun.ServiceMetadataArgs(
        annotations={
            "run.googleapis.com/launch-stage": "BETA",
        },
    ),
    traffics=[
        gcp.cloudrun.ServiceTrafficArgs(
            percent=100,
            latest_revision=True,
        )
    ],
    opts=pulumi.ResourceOptions(
        depends_on=[
            titiler_sentinel.lambda_api,
            cloud_run_offset_tile.default,
        ]
    ),
)
noauth_iam_policy = gcp.cloudrun.IamPolicy(
    construct_name("cloud-run-noauth-iam-policy-orchestrator"),
    location=default.location,
    project=default.project,
    service=default.name,
    policy_data=cloud_run_offset_tile.noauth_iam_policy_data.policy_data,
)
