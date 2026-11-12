"""A changed image digest must change TiPG's Cloud Run revision template."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pulumi
import pytest

from test.test_stack.conftest import MyMocks


@pytest.mark.parametrize("digest", ["a" * 64, "b" * 64])
@pulumi.runtime.test
def test_tipg_revision_uses_image_digest(monkeypatch, digest):
    pulumi.runtime.set_mocks(MyMocks(), project="cerulean-cloud", stack="test")
    pulumi.runtime.set_all_config(
        {"gcp:project": "test-project", "gcp:region": "europe-west1"}
    )
    image_name = "europe-west1-docker.pkg.dev/test-project/images/tipg"
    pinned_image = f"{image_name}@sha256:{digest}"
    monkeypatch.setitem(
        sys.modules,
        "cloud_run_images",
        SimpleNamespace(
            cloud_run_tipg_image=SimpleNamespace(
                name=f"{image_name}:latest", repo_digest=pinned_image
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "database",
        SimpleNamespace(
            instance=SimpleNamespace(
                connection_name="test-project:europe-west1:database"
            ),
            sql_instance_url="postgresql://test",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "cloud_run_infer",
        SimpleNamespace(noauth_iam_policy_data=SimpleNamespace(policy_data="{}")),
    )
    monkeypatch.setitem(
        sys.modules,
        "utils",
        SimpleNamespace(construct_name=lambda name: f"cerulean-cloud-test-{name}"),
    )
    path = Path(__file__).resolve().parents[2] / "stack" / "cloud_run_tipg.py"
    spec = importlib.util.spec_from_file_location("tipg_deployment_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def check_template(template):
        assert template["spec"]["containers"][0]["image"] == pinned_image

    return module.default.template.apply(check_template)
