from test.test_stack import conftest

import pulumi
import pytest

pulumi.runtime.set_mocks(conftest.MyMocks())

from stack import infra  # noqa: E402


@pytest.mark.skip(reason="Coroutine failure")
@pulumi.runtime.test
def test_bucket_has_labels():
    def check_labels(args):
        urn, tags = args
        assert "pulumi" in tags

    return pulumi.Output.all(infra.bucket.urn, infra.bucket.labels).apply(check_labels)
