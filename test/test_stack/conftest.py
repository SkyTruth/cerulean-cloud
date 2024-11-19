"""
pytest configuration for the stack package.
"""

import os
import sys

import pulumi

# add stack path to enable relative imports from stack
sys.path.append(os.path.join(os.path.abspath("."), "stack"))


class MyMocks(pulumi.runtime.Mocks):
    """
    MyMocks class for pulumi.
    """

    def new_resource(self, args: pulumi.runtime.MockResourceArgs):
        """
        new_resource method for pulumi.
        """
        outputs = args.inputs
        return [args.name + "_id", outputs]

    def call(self, args: pulumi.runtime.MockCallArgs):
        """
        call method for pulumi.
        """
        return {}
