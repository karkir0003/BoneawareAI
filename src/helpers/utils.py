import zipfile
import os
import shutil 

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
    
def zip_folder(folder_path, zip_file_name, zip_directory=None):
    """
    Given directory, generate zip file. NOTE that the zip is done recursively

    Params:
    - path: directory
    - zip_file_name: Name of .zip file. It must have suffix .zip
    - zip_directory (optional): Directory to store the .zip file. Defaults to current working directory
    """
    if not zip_file_name.endswith('.zip'):
        raise ValueError(f"Error: The zip file name must end with '.zip'. Provided name: '{zip_file_name}'")
    
    if zip_directory:
      if not os.path.isdir(zip_directory):
        print(f"Directory '{zip_directory}' does not exist. Creating it...")
        os.makedirs(zip_directory)
    else:
      zip_directory = os.getcwd()
    
    # Combine zip_directory and zip_name to form the full path for the zip file
    zip_file_path = os.path.join(zip_directory, zip_file_name)

    # Check if the provided folder path is a valid directory
    if os.path.isdir(folder_path):
        # Create the zip file (the 'zip_name' is already included in the full path)
        shutil.make_archive(zip_file_path.replace('.zip', ''), 'zip', folder_path)
        print(f"Folder '{folder_path}' has been zipped at '{zip_file_path}'")
    else:
        print(f"The provided path '{folder_path}' is not a valid directory.")
