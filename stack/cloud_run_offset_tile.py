"""infra for cloud run function to perform inference on offset tiles
Reference doc: https://www.pulumi.com/blog/build-publish-containers-iac/
"""
import cloud_run_images
import pulumi
import pulumi_gcp as gcp
from utils import construct_name

stack = pulumi.get_stack()

# Assign access to cloud secrets
cloud_function_service_account = gcp.serviceaccount.Account(
    construct_name("cloud-run-offset-tile"),
    account_id=f"{stack}-cr-offset-tile",
    display_name="Service Account for cloud run.",
)

cloud_function_service_account_iam = gcp.projects.IAMMember(
    construct_name("cloud-run-offset-tile-cloudSqlClient"),
    project=pulumi.Config("gcp").require("project"),
    role="roles/cloudsql.client",
    member=cloud_function_service_account.email.apply(
        lambda email: f"serviceAccount:{email}"
    ),
)

cloud_function_service_account_iam = gcp.projects.IAMMember(
    construct_name("cloud-run-offset-tile-secretmanagerSecretAccessor"),
    project=pulumi.Config("gcp").require("project"),
    role="roles/secretmanager.secretAccessor",
    member=cloud_function_service_account.email.apply(
        lambda email: f"serviceAccount:{email}"
    ),
)

# IAM Binding for Secret Manager access
secret_accessor_binding = gcp.secretmanager.SecretIamMember(
    construct_name("cloud-run-offset-tile-secret-accessor-binding"),
    secret_id=pulumi.Config("cerulean-cloud").require("keyname"),
    role="roles/secretmanager.secretAccessor",
    member=pulumi.Output.concat(
        "serviceAccount:", cloud_function_service_account.email
    ),
    opts=pulumi.ResourceOptions(depends_on=[cloud_function_service_account]),
)

service_name = construct_name("cloud-run-offset-tiles")
default = gcp.cloudrun.Service(
    service_name,
    opts=pulumi.ResourceOptions(depends_on=[secret_accessor_binding]),
    name=service_name,
    location=pulumi.Config("gcp").require("region"),
    template=gcp.cloudrun.ServiceTemplateArgs(
        spec=gcp.cloudrun.ServiceTemplateSpecArgs(
            service_account_name=cloud_function_service_account.email,
            containers=[
                gcp.cloudrun.ServiceTemplateSpecContainerArgs(
                    image=cloud_run_images.cloud_run_offset_tile_image.name,
                    envs=[
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="SOURCE",
                            value="remote",
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
            timeout_seconds=300,
            container_concurrency=1,
        ),
        metadata=dict(
            name=service_name + "-" + cloud_run_images.cloud_run_offset_tile_sha,
            annotations={
                "autoscaling.knative.dev/maxScale": "2000",
            },
        ),
    ),
    metadata=gcp.cloudrun.ServiceMetadataArgs(
        annotations={
            "run.googleapis.com/launch-stage": "BETA",
        }
    ),
    traffics=[
        gcp.cloudrun.ServiceTrafficArgs(
            percent=100,
            latest_revision=True,
        )
    ],
)
noauth_iam_policy_data = gcp.organizations.get_iam_policy(
    bindings=[
        gcp.organizations.GetIAMPolicyBindingArgs(
            role="roles/run.invoker",
            members=["allUsers"],
        )
    ]
)
noauth_iam_policy = gcp.cloudrun.IamPolicy(
    construct_name("cloud-run-noauth-iam-policy-offset"),
    location=default.location,
    project=default.project,
    service=default.name,
    policy_data=noauth_iam_policy_data.policy_data,
)
