from urllib.request import urlretrieve
import os
from tqdm import tqdm

filename = {'40282102': '002.tiff',
 '41265615': 'MIDOGpp.json',
 '41265612': 'MIDOGpp.sqlite'}

for file in tqdm(filename):
    url = f'https://ndownloader.figshare.com/files/{file}'
    urlretrieve(url, 'images'+os.sep+filename[file])
