import os
import requests
import zipfile
import io
import shutil

# supress the http warning
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("Downloading dataset...")

res = requests.get("https://sci2s.ugr.es/keel/dataset/data/classification//full/All.zip", verify=False)

print("Unzipping...")

with zipfile.ZipFile(io.BytesIO(res.content)) as z:
    members = [m for m in z.namelist() if not m.startswith('__MACOSX/')]
    z.extractall("datasets", members=members)

os.rename("datasets/classification", "datasets/classification_raw")

print("Formatting data...")

dataset_target_path = "datasets/classification"
dataset_source_path = "datasets/classification_raw"

for item in os.listdir(dataset_source_path):
    dn_path = os.path.join(dataset_source_path, item)
    if not os.path.isdir(dn_path):
        continue
        
    os.makedirs(os.path.join(dataset_target_path, item), exist_ok=True)

    subsets = [f for f in os.listdir(dn_path) if f.endswith(".dat")]
    for ss in subsets:
        ss_srcpath = os.path.join(dataset_source_path, item, ss)
        ss_tgtpath = os.path.join(dataset_target_path, item, ss.replace(".dat", ".csv"))

        with open(ss_srcpath, 'r') as f:
            content = f.read()
        
        header, output_attr, data = None, None, None
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith('@inputs'):
                header = line.replace('@inputs', '').strip()
            elif line.startswith('@outputs'):
                output_attr = line.replace('@outputs', '').strip()
            elif line.startswith('@output'):
                output_attr = line.replace('@output', '').strip()
            elif line.startswith('@data'):
                data_index = lines.index(line) + 1
                data = '\n'.join(lines[data_index:])
                break

        if header is None or output_attr is None or data is None:
            # print("warning: dataset processing failed for", ss_srcpath, ": header or data could not be read.")
            continue

        with open(ss_tgtpath, 'w') as f:
            f.write(header  + ', ' + output_attr + '\n' + data)

shutil.rmtree("datasets/classification_raw")
for n in ["kddcup", "poker", "winequality-red", "winequality-white", "census", "adult"]:
    shutil.rmtree("datasets/classification/%s" % n)

print("Done!")