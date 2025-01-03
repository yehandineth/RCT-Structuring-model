from zipfile import ZipFile
from urllib.request import urlretrieve
import os
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets....", end="")

    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])
        os.remove(save_path)
        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)