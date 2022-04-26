import pytest
import pulumi

class MyMocks(pulumi.runtime.Mocks):
    def new_resource(self, args: pulumi.runtime.MockResourceArgs):
        outputs = args.inputs
        return [args.name + '_id', outputs]
    def call(self, args: pulumi.runtime.MockCallArgs):
        return {}

pulumi.runtime.set_mocks(MyMocks())

import infra

@pulumi.runtime.test
def test_bucket_has_labels():
    def check_labels(args):
        urn, tags = args
        assert "pulumi" in tags
    
    return pulumi.Output.all(infra.bucket.urn, infra.bucket.labels).apply(check_labels)