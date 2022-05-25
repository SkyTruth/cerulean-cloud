from test.test_stack import conftest
from unittest.mock import patch

import pulumi
import pytest

pulumi.runtime.set_mocks(conftest.MyMocks())


@pytest.mark.skip(reason="intermittent failing test")
@pulumi.runtime.test
@patch("utils.create_package")
def test_bucket_has_name(mock_create_package):
    mock_create_package.return_value = "test/test_stack/fixtures/package.zip"

    from stack import titiler_sentinel  # noqa: E402

    def check_name(args):
        assert "titiler-lambda-archive" in args[0]

    return pulumi.Output.all(titiler_sentinel.s3_bucket.id).apply(check_name)
