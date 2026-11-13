"""A changed image digest must change TiPG's Cloud Run revision template."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pulumi
import pytest


def _assert_tipg_revision_uses_image_digest(patch, digest):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from test.test_stack.conftest import MyMocks

    pulumi.runtime.set_mocks(MyMocks(), project="cerulean-cloud", stack="test")
    pulumi.runtime.set_all_config(
        {"gcp:project": "test-project", "gcp:region": "europe-west1"}
    )
    image_name = "europe-west1-docker.pkg.dev/test-project/images/tipg"
    pinned_image = f"{image_name}@sha256:{digest}"
    patch.setitem(
        sys.modules,
        "cloud_run_images",
        SimpleNamespace(
            cloud_run_tipg_image=SimpleNamespace(
                name=f"{image_name}:latest", repo_digest=pinned_image
            )
        ),
    )
    patch.setitem(
        sys.modules,
        "database",
        SimpleNamespace(
            instance=SimpleNamespace(
                connection_name="test-project:europe-west1:database"
            ),
            sql_instance_url="postgresql://test",
        ),
    )
    patch.setitem(
        sys.modules,
        "cloud_run_infer",
        SimpleNamespace(noauth_iam_policy_data=SimpleNamespace(policy_data="{}")),
    )
    patch.setitem(
        sys.modules,
        "utils",
        SimpleNamespace(construct_name=lambda name: f"cerulean-cloud-test-{name}"),
    )

    @pulumi.runtime.test
    def check_revision():
        path = root / "stack" / "cloud_run_tipg.py"
        spec = importlib.util.spec_from_file_location("tipg_deployment_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def check_template(template):
            assert template["spec"]["containers"][0]["image"] == pinned_image

        return module.default.template.apply(check_template)

    check_revision()


@pytest.mark.parametrize("digest", ["a" * 64, "b" * 64])
def test_tipg_revision_uses_image_digest(digest):
    # Pulumi caches event-loop-bound globals; do not share them with pytest-asyncio.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), digest],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as patch:
        _assert_tipg_revision_uses_image_digest(patch, sys.argv[1])
