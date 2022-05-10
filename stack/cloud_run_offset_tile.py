"""infra for cloud run function to perform inference on offset tiles
Reference doc: https://www.pulumi.com/blog/build-publish-containers-iac/
"""
import pulumi
import pulumi_gcp as gcp
from utils import construct_name

# import pulumi_docker as docker
registry = gcp.container.Registry(construct_name("registry"), location="EU")
registry_url = registry.id.apply(
    lambda _: gcp.container.get_registry_repository().repository_url
)

image_name = registry_url.apply(lambda url: f"{url}/myapp")
registry_info = None  # use gcloud for authentication.

"""
image = docker.Image('my-image',
    build='app',
    image_name=image_name,
    registry=registry_info,
)
"""
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
