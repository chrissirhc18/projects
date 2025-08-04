import os
import pandas as pd
import requests
from tqdm import tqdm

# Load your CSV
df = pd.read_csv('BirdsVoice.csv')

# Ensure the audio directory exists
os.makedirs('audio', exist_ok=True)

for xc_id in tqdm(df['xc_id'].unique()):
    num_id = xc_id.replace("XC", "")
    url = f"https://www.xeno-canto.org/{num_id}/download"
    file_path = os.path.join('audio', f"{xc_id}.mp3")

    if os.path.exists(file_path):
        continue

    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        else:
            print(f"Failed to download {xc_id}: Status {response.status_code}")
    except Exception as e:
        print(f"Error downloading {xc_id}: {e}")


