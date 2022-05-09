import os
import sys

import pulumi

# add stack path to enable relative imports from stack
sys.path.append(os.path.join(os.path.abspath("."), "stack"))


class MyMocks(pulumi.runtime.Mocks):
    def new_resource(self, args: pulumi.runtime.MockResourceArgs):
        outputs = args.inputs
        return [args.name + "_id", outputs]

    def call(self, args: pulumi.runtime.MockCallArgs):
        return {}
