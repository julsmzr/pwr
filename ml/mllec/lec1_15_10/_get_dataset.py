import os

import requests
import zipfile
import io
import shutil

print("Downloading dataset...")

res = requests.get("https://www.kaggle.com/api/v1/datasets/download/alessiocorrado99/animals10")

print("Unzipping...")

with zipfile.ZipFile(io.BytesIO(res.content)) as z:
    z.extractall("datasets")

print("Formatting data...")

os.rename("datasets/raw-img", "datasets/animals-10")

shutil.rmtree("datasets/animals-10/ragno")
os.remove("datasets/translate.py")

translate = {
    "cane": "dog", 
    "cavallo": "horse", 
    "elefante": "elephant", 
    "farfalla": "butterfly", 
    "gallina": "chicken", 
    "gatto": "cat", 
    "mucca": "cow", 
    "pecora": "sheep", 
    "scoiattolo": "squirrel",
}

for src, tgt in translate.items():
    if os.path.exists("datasets/animals-10/%s" % src):
        print(src, "to", tgt)
        os.rename("datasets/animals-10/%s" % src, "datasets/animals-10/%s" % tgt)
    else:
        print("does not exists:", src, "to", tgt)

print("Done!")
