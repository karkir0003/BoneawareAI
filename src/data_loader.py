import zipfile
import os
from helpers.aws_utils import download_file_from_s3
from constants import BONEAWAREAI_S3_BUCKET
from helpers.utils import unzip_file


def download_dataset(file, output_folder):
    download_file_from_s3(BONEAWAREAI_S3_BUCKET, file, output_folder)
    zip_file_path = os.path.join(output_folder, file)

    if zipfile.is_zipfile(zip_file_path):
        unzip_file(zip_file_path)
