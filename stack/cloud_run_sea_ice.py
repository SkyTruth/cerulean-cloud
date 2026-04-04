"""Infra for a scheduled Cloud Run worker that syncs sea-ice extent to GCS."""

import pulumi
import pulumi_gcp as gcp
from utils import construct_name

config = pulumi.Config("sea-ice")
enabled = config.get_bool("enabled") or False

default = None
job = None

if enabled:
    import cloud_run_images

    stack = pulumi.get_stack()
    gcp_config = pulumi.Config("gcp")
    app_config = pulumi.Config("cerulean-cloud")

    region = gcp_config.require("region")
    sea_ice_mask_gcs_uri = app_config.require("sea_ice_mask_gcs_uri")
    bucket_name = sea_ice_mask_gcs_uri.removeprefix("gs://").split("/", 1)[0]

    worker_service_account = gcp.serviceaccount.Account(
        construct_name("cr-sea-ice"),
        account_id=f"{stack}-cr-sea-ice",
        display_name="Service Account for sea-ice sync Cloud Run.",
    )

    scheduler_service_account = gcp.serviceaccount.Account(
        construct_name("sch-sea-ice"),
        account_id=f"{stack}-sea-ice-sch",
        display_name="Service Account for sea-ice Cloud Scheduler.",
    )

    bucket_writer = gcp.storage.BucketIAMMember(
        construct_name("sea-ice-bucket-writer"),
        bucket=bucket_name,
        role="roles/storage.objectAdmin",
        member=worker_service_account.email.apply(
            lambda email: f"serviceAccount:{email}"
        ),
    )

    service_name = construct_name("cr-sea-ice")
    envs = [
        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
            name="UVICORN_PORT",
            value="8080",
        ),
        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
            name="SEA_ICE_MASK_GCS_URI",
            value=sea_ice_mask_gcs_uri,
        ),
        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
            name="SEA_ICE_REQUEST_TIMEOUT_SECONDS",
            value=config.get("request_timeout_seconds") or "600",
        ),
    ]

    default = gcp.cloudrun.Service(
        service_name,
        name=service_name,
        location=region,
        autogenerate_revision_name=True,
        template=gcp.cloudrun.ServiceTemplateArgs(
            spec=gcp.cloudrun.ServiceTemplateSpecArgs(
                service_account_name=worker_service_account.email,
                containers=[
                    gcp.cloudrun.ServiceTemplateSpecContainerArgs(
                        image=cloud_run_images.cloud_run_sea_ice_image.name,
                        envs=envs,
                        resources=dict(limits=dict(memory="2Gi", cpu="1000m")),
                    )
                ],
                timeout_seconds=1800,
                container_concurrency=1,
            ),
        ),
        metadata=gcp.cloudrun.ServiceMetadataArgs(
            annotations={"run.googleapis.com/launch-stage": "BETA"},
        ),
        traffics=[
            gcp.cloudrun.ServiceTrafficArgs(
                percent=100,
                latest_revision=True,
            )
        ],
        opts=pulumi.ResourceOptions(depends_on=[bucket_writer]),
    )

    invoker = gcp.cloudrun.IamMember(
        construct_name("cr-sea-ice-invoker"),
        service=default.name,
        location=default.location,
        role="roles/run.invoker",
        member=scheduler_service_account.email.apply(
            lambda email: f"serviceAccount:{email}"
        ),
    )

    http_target = gcp.cloudscheduler.JobHttpTargetArgs(
        http_method="GET",
        uri=default.statuses[0].url.apply(lambda url: f"{url}/sync"),
        oidc_token=gcp.cloudscheduler.JobHttpTargetOidcTokenArgs(
            service_account_email=scheduler_service_account.email,
        ),
    )

    job = gcp.cloudscheduler.Job(
        construct_name("sea-ice-scheduler"),
        description="Run the sea-ice sync worker on a daily schedule with cadence gating.",
        schedule=config.get("daily_schedule") or "0 8 * * *",
        time_zone=config.get("time_zone") or "UTC",
        http_target=http_target,
        opts=pulumi.ResourceOptions(depends_on=[invoker]),
    )
