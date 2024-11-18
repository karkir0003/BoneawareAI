import boto3
import configparser
import os
from botocore.exceptions import NoCredentialsError, ClientError
from constants import BONEAWAREAI_DATA_ACCESS_ROLE, BONEAWAREAI_DATA_ACCESS_SESSION, BONEAWAREAI_S3_BUCKET, DATASETS_FOLDER
from helpers.utils import zip_folder

def get_aws_credentials():
    config = configparser.ConfigParser()
    config.read("config.ini")
    aws_access_key_id=config['default']["aws_access_key_id"]
    aws_secret_access_key=config['default']["aws_secret_access_key"]
    return aws_access_key_id, aws_secret_access_key

def assume_role(role_arn, session_name, duration=3600):
    """
    Assume an IAM role and get temporary credentials.
    Note duration in seconds
    """
    aws_access_key_id, aws_secret_access_key = get_aws_credentials()
    sts_client = boto3.client(
        'sts', 
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    try:
        response = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=session_name,
            DurationSeconds=duration 
        )
        
        credentials = response['Credentials']
        return credentials
    except ClientError as e:
        print(f"Error assuming role: {e}")


def download_file_from_s3(bucket_name, file_key, output_folder):
    """
    Download a file from S3 using the provided temporary credentials.
    
    Params:
    - bucket name: name of S3 bucket
    - file_key: path within s3 bucket for specific file
    - output_folder: where to download the file to
    """
    # Get temporary credentials
    credentials = assume_role(BONEAWAREAI_DATA_ACCESS_ROLE, BONEAWAREAI_DATA_ACCESS_SESSION)
    
    # Use the temporary credentials to create an S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken']
    )
    
    if (not os.path.isdir(output_folder)):
        os.makedirs(output_folder)
    
    download_path = os.path.join(output_folder, file_key)

    try:
        # Download the file from S3
        s3_client.download_file(bucket_name, file_key, download_path)
        print(f"File downloaded successfully to {download_path}")
    except NoCredentialsError as nc:
        print("Error: No credentials provided for S3 operation")
    except ClientError as e:
        print(f"Error downloading file from S3: {e}")

def upload_file_to_s3(file_path, bucket_name):
    """
    Given path to a file (NOT a directory), upload it to S3
    
    file_path: path to file
    bucket_name: S3 Bucket
    """
    # Get temporary credentials
    credentials = assume_role(BONEAWAREAI_DATA_ACCESS_ROLE, BONEAWAREAI_DATA_ACCESS_SESSION)
    
    # Use the temporary credentials to create an S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken']
    )
    
    object_name = os.path.basename(file_path)
    
    try:
        # Upload file to S3
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"File uploaded successfully to {file_path}")
    except NoCredentialsError as nc:
        print("Error: No credentials provided for S3 operation")
    except ClientError as e:
        print(f"Error uploading file to S3: {e}")

def upload_folder_to_s3(folder, bucket_name):
    """
    Given path to a directory, upload a zip file in S3
    
    folder: directory
    bucket_name: S3 Bucket
    """
    file_name = os.path.basename(folder) + '.zip'
    zip_location = os.path.dirname(folder)
    print(f"zip location: {zip_location}")
    zip_folder(folder, file_name, zip_directory=zip_location)
    upload_file_to_s3(os.path.join(zip_location, file_name), bucket_name)