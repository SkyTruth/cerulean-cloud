"""infra for cloud run function for orchestration
Reference doc: https://www.pulumi.com/blog/build-publish-containers-iac/
"""

import cloud_run_images
import pulumi
import pulumi_gcp as gcp
from cloud_run_offset_tile import noauth_iam_policy_data
from database import instance, sql_instance_url
from utils import construct_name

config = pulumi.Config()
stack = pulumi.get_stack()

# Assign access to cloud secrets
cloud_function_service_account = gcp.serviceaccount.Account(
    construct_name("cr-tipg"),
    account_id=f"{stack}-cr-tipg",
    display_name="Service Account for cloud run.",
)

cloud_function_service_account_iam = gcp.projects.IAMMember(
    construct_name("cr-tipg-cloudSqlClient"),
    project=pulumi.Config("gcp").require("project"),
    role="roles/cloudsql.client",
    member=cloud_function_service_account.email.apply(
        lambda email: f"serviceAccount:{email}"
    ),
)

cloud_function_service_account_iam = gcp.projects.IAMMember(
    construct_name("cr-tipg-secretmanagerSecretAccessor"),
    project=pulumi.Config("gcp").require("project"),
    role="roles/secretmanager.secretAccessor",
    member=cloud_function_service_account.email.apply(
        lambda email: f"serviceAccount:{email}"
    ),
)

# IAM Binding for Secret Manager access
secret_accessor_binding = gcp.secretmanager.SecretIamMember(
    construct_name("cr-tipg-secret-accessor-binding"),
    secret_id=pulumi.Config("cerulean-cloud").require("keyname"),
    role="roles/secretmanager.secretAccessor",
    member=pulumi.Output.concat(
        "serviceAccount:", cloud_function_service_account.email
    ),
    opts=pulumi.ResourceOptions(depends_on=[cloud_function_service_account]),
)

service_name = construct_name("cr-tipg")
default = gcp.cloudrun.Service(
    service_name,
    opts=pulumi.ResourceOptions(depends_on=[secret_accessor_binding]),
    name=service_name,
    location=pulumi.Config("gcp").require("region"),
    autogenerate_revision_name=True,
    template=gcp.cloudrun.ServiceTemplateArgs(
        spec=gcp.cloudrun.ServiceTemplateSpecArgs(
            service_account_name=cloud_function_service_account.email,
            containers=[
                gcp.cloudrun.ServiceTemplateSpecContainerArgs(
                    image=cloud_run_images.cloud_run_tipg_image.name,
                    envs=[
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="UVICORN_PORT",
                            value="8080",
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="DATABASE_URL",
                            value=sql_instance_url,
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TIPG_NAME", value="Cerulean OGC API"
                        ),
                        *[
                            gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                                name=f"TIPG_TABLE_CONFIG__public_{geom_table}__GEOMCOL",
                                value="geometry",
                            )
                            for geom_table in [
                                "sentinel1_grd",
                                "orchestrator_run",
                                "slick",
                                "aoi",
                                "slick_plus",
                                "source_infra",
                                "slick_to_source",
                                "get_slicks_by_aoi_or_source",
                                "get_slicks_by_aoi",
                                "get_slicks_by_source",
                            ]
                        ],
                        *[
                            gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                                name=f"TIPG_TABLE_CONFIG__public_{datetime_table}__DATETIMECOL",
                                value="update_time",
                            )
                            for datetime_table in [
                                "layer",
                                "model",
                                "subscription",
                                "aoi_type",
                                "hitl_slick",
                            ]
                        ],
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TIPG_TABLE_CONFIG__public_sentinel1_grd__DATETIMECOL",
                            value="start_time",
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TIPG_TABLE_CONFIG__public_orchestrator_run__DATETIMECOL",
                            value="inference_start_time",
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TIPG_TABLE_CONFIG__public_slick__DATETIMECOL",
                            value="slick_timestamp",
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TIPG_TABLE_CONFIG__public_slick_plus__DATETIMECOL",
                            value="slick_timestamp",
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TIPG_MAX_FEATURES_PER_QUERY",
                            value=50000,
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="RESTRICTED_COLLECTIONS",
                            value='["public.aoi_user","public.filter", "public.frequency", "public.verification_token", "public.accounts", "public.sessions", "public.subscription", "public.users", "public.slick_to_source", "public.source", "public.source_infra", "public.source_type", "public.source_vessel", "public.source_dark", "public.source_natural", "public.source_to_tag", "public.tag", "public.hitl_slick", "public.permission"]',
                            # EditTheDatabase
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="SECRET_API_KEY",
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
                    resources=dict(limits=dict(memory="8Gi", cpu="6000m")),
                ),
            ],
            timeout_seconds=420,
        ),
        metadata=dict(
            annotations={
                "run.googleapis.com/cloudsql-instances": instance.connection_name,
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
)
noauth_iam_policy = gcp.cloudrun.IamPolicy(
    construct_name("cr-noauth-iam-policy-tipg"),
    location=default.location,
    project=default.project,
    service=default.name,
    policy_data=noauth_iam_policy_data.policy_data,
)
