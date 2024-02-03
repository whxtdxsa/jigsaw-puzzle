import os
import subprocess

def download_and_unzip():
    if not os.path.exists('open.zip'):
        # Download data from Google Drive
        subprocess.run(['pip', 'install', 'gdown'], check=True)
        subprocess.run(['gdown', 'https://drive.google.com/uc?id=13oGkm3Ao7fL2p51H62J68Gw630ABBR0g'], check=True)

    if not os.path.exists('./data/train'):
        # Unzip to the 'data' directory
        subprocess.run(['unzip', 'open.zip', '-d', 'data'], check=True)
    
def download_lib():
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,check=True)