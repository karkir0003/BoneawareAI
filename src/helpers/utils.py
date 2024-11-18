import zipfile
import os


def unzip_file(path):
    """
    Given path to .zip file, extract the contents and remove the original .zip file

    Params:
    - path: path to .zip file
    """
    curr_directory = os.path.dirname(path)
    with zipfile.ZipFile(path, "r") as zip_file:
        zip_file.extractall(curr_directory)
    print(f"successfully unzipped the file at path {path}")
