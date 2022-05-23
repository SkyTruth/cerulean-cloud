import pytest

from cerulean_cloud.titiler_client import TitilerClient


@pytest.mark.skip
def test_create_fixture_tile(
    url="https://0xshe4bmk8.execute-api.eu-central-1.amazonaws.com/",
):
    ti = TitilerClient(url=url)
    print(ti)
