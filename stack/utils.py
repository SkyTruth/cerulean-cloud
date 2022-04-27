"""utils for stack building"""
import pulumi

project = pulumi.get_project()
stack = pulumi.get_stack()


def construct_name(resource_name: str) -> str:
    """construct resource names from project and stack"""
    return f"{project}-{stack}-{resource_name}"
