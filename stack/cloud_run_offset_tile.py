"""infra for cloud run function to perform inference on offset tiles"""
import pulumi
import pulumi_gcp as gcp
from utils import construct_name

default = gcp.cloudrun.Service(
    construct_name("cloud-run-offset-tiles"),
    location=pulumi.Config("gcp").require("region"),
    template=gcp.cloudrun.ServiceTemplateArgs(
        spec=gcp.cloudrun.ServiceTemplateSpecArgs(
            containers=[
                gcp.cloudrun.ServiceTemplateSpecContainerArgs(
                    image="us-docker.pkg.dev/cloudrun/container/hello",
                    envs=[
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="SOURCE",
                            value="remote",
                        ),
                    ],
                )
            ],
        )
    ),
    metadata=gcp.cloudrun.ServiceMetadataArgs(
        annotations={
            "generated-by": "magic-modules",
        },
    ),
    traffics=[
        gcp.cloudrun.ServiceTrafficArgs(
            percent=100,
            latest_revision=True,
        )
    ],
    autogenerate_revision_name=True,
)
noauth_iam_policy = gcp.organizations.get_iam_policy(
    bindings=[
        gcp.organizations.GetIAMPolicyBindingArgs(
            role="roles/run.invoker",
            members=["allUsers"],
        )
    ]
)
noauth_iam_policy = gcp.cloudrun.IamPolicy(
    construct_name("cloud-run-noauth-iam-policy"),
    location=default.location,
    project=default.project,
    service=default.name,
    policy_data=noauth_iam_policy.policy_data,
)
