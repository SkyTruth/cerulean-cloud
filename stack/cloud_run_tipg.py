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

service_name = construct_name("cloud-run-tipg")
default = gcp.cloudrun.Service(
    service_name,
    name=service_name,
    location=pulumi.Config("gcp").require("region"),
    template=gcp.cloudrun.ServiceTemplateArgs(
        spec=gcp.cloudrun.ServiceTemplateSpecArgs(
            containers=[
                gcp.cloudrun.ServiceTemplateSpecContainerArgs(
                    image=cloud_run_images.cloud_run_tipg_image.name,
                    envs=[
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
                                "magic_link",
                                "aoi_type",
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
                            value='["public.aoi_user","public.filter", "public.frequency", "public.magic_link", "public.subscription", "public.user", "public.slick_to_source", "public.source", "public.source_infra", "public.source_type", "public.source_vessel", "public.get_slicks_by_source", "public.get_slicks_by_aoi_or_source"]',
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="SECRET_API_KEY",
                            value=pulumi.Config("cerulean-cloud").require("apikey"),
                        ),
                    ],
                    resources=dict(limits=dict(memory="8Gi", cpu="6000m")),
                ),
            ],
            timeout_seconds=420,
        ),
        metadata=dict(
            name=service_name + "-" + cloud_run_images.cloud_run_tipg_sha,
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
    construct_name("cloud-run-noauth-iam-policy-tipg"),
    location=default.location,
    project=default.project,
    service=default.name,
    policy_data=noauth_iam_policy_data.policy_data,
)
