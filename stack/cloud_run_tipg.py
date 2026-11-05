"""infra for cloud run function for orchestration
Reference doc: https://www.pulumi.com/blog/build-publish-containers-iac/
"""

import json

import cloud_run_images
import pulumi
import pulumi_gcp as gcp
from cloud_run_infer import noauth_iam_policy_data
from database import instance, sql_instance_url
from utils import construct_name

config = pulumi.Config()
stack = pulumi.get_stack()

# Service account for Cloud SQL access
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

service_name = construct_name("cr-tipg")
default = gcp.cloudrun.Service(
    service_name,
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
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TIPG_DB_SPATIAL_EXTENT", value="False"
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TIPG_DB_DATETIME_EXTENT", value="False"
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TIPG_DB_ONLY_SPATIAL_TABLES", value="False"
                        ),
                        # EditTheDatabase: explicitly opt collections into the public API.
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TIPG_DB_TABLES",
                            value=json.dumps(
                                [
                                    "public.aoi_eez",
                                    "public.aoi_iho",
                                    "public.aoi_mpa",
                                    "public.aoi_type_public",
                                    "public.cls",
                                    "public.layer",
                                    "public.model",
                                    "public.sentinel1_grd",
                                    "public.slick_plus",
                                    "public.slick_plus_2",
                                    "public.source_plus",
                                    "public.source_type",
                                ]
                            ),
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TIPG_DB_FUNCTIONS",
                            value=json.dumps(
                                [
                                    "public.get_slick_subclses",
                                    "public.get_slicks_by_aoi",
                                    "public.get_slicks_by_aoi_or_source",
                                    "public.get_slicks_by_source",
                                    "public.slick_history",
                                ]
                            ),
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TIPG_SORT_COLUMNS", value="False"
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
                                "aoi_type_public",
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
                    ],
                    resources=dict(limits=dict(memory="8Gi", cpu="6000m")),
                    startup_probe=gcp.cloudrun.ServiceTemplateSpecContainerStartupProbeArgs(
                        timeout_seconds=240,
                        period_seconds=240,
                        failure_threshold=3,
                        tcp_socket=gcp.cloudrun.ServiceTemplateSpecContainerStartupProbeTcpSocketArgs(
                            port=8080,
                        ),
                    ),
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
