# -*- coding: utf-8 -*-

import os
import boto3
from botocore.config import Config


def is_authorized_to_pull_image(account_id, repository_name):
    """Authorization check for Boathouse. You can set environment variables or use your local config"""
    try:
        # Retrieve the access keys and session token from environment variables
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        session_token = os.environ.get("AWS_SESSION_TOKEN")

        ecr_config = Config(
            region_name = 'us-east-1',
            signature_version = 'v4',
            retries = {
                'max_attempts': 10,
                'mode': 'standard'
            }
        )

        # Create a Boto3 ECR client
        if not access_key or not secret_key or not session_token:
            ecr_client = boto3.client("ecr", config=ecr_config)
        else:
            ecr_client = boto3.client(
                "ecr",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                aws_session_token=session_token,
                config=ecr_config
            )

        # Get the repository details
        response = ecr_client.describe_repositories(repositoryNames=[repository_name])
        repositories = response["repositories"]

        if not repositories:
            return False

        # Verify the repository is in the specified account
        repository = repositories[0]
        repository_arn = repository["repositoryArn"]
        repository_account_id = repository_arn.split(":")[4]

        if repository_account_id != account_id:
            return False

        # Get the authorization token
        response = ecr_client.get_authorization_token(registryIds=[account_id])
        authorization_data = response["authorizationData"]
        if not authorization_data:
            return False

        # The user is authorized to pull the image
        return True

    # This is an initalization check, so we want to catch anything here
    # pylint: disable=broad-exception-caught
    except Exception as _:
        # An error occurred, indicating the user is not authorized to pull the image
        return False


# check if you can pull this image from any of the verified accounts
if not (
    is_authorized_to_pull_image("199505301519", "important-sampling") # digital solutions production
    or is_authorized_to_pull_image("799488199792", "important-sampling") # digital solutions staging
    or is_authorized_to_pull_image("985109957555", "important-sampling") # digital solutions dev
    or is_authorized_to_pull_image("883935268206", "important-sampling") # lakewood
):
    raise RuntimeError("Unauthorized")
