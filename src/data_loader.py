import zipfile
import os
from helpers.aws_utils import download_file_from_s3
from constants import BONEAWAREAI_S3_BUCKET
from helpers.utils import unzip_file

def download_dataset(file, output_folder):
    download_file_from_s3(BONEAWAREAI_S3_BUCKET, file, output_folder)
    
    if (zipfile.is_zipfile(file)):
        unzip_file(os.path.join(output_folder, file))
