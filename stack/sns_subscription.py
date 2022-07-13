"""Subscription to SNS topic"""
import cloud_function_scene_relevancy
import pulumi_aws as aws
from utils import construct_name

sentinel1_sqs_target = aws.sns.TopicSubscription(
    construct_name("sentinel1-subscription"),
    endpoint=cloud_function_scene_relevancy.fxn.https_trigger_url,
    protocol="https",
    confirmation_timeout_in_minutes=5,
    topic="arn:aws:sns:eu-central-1:214830741341:SentinelS1L1C",
)
