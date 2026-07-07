import os
from pathlib import Path
import base64
import json


metadata_dir = Path("/Users/poldrack/data_unsynced/BCBS/simple_workflow/wf_snakemake/.snakemake/metadata")

tags_to_print = [
    'rule',
    'code', 
    'input',
]

metadata_dict = {}
for encoded_name in metadata_dir.glob("*"):
    decoded = base64.b64decode(encoded_name.name).decode()

    with open(encoded_name, "r") as f:
        metadata_dict[decoded] = json.load(f)
        metadata_dict[decoded]['_metadata_path'] = str(encoded_name)

print("\n" + "-"*40)
for filename, metadata in metadata_dict.items():
    print(f"Metadata for file: {filename}")
    for tag in tags_to_print:
        md = metadata.get(tag)
        if type(md) is list:
            print(f"{tag}:")
            if len(md) == 0:
                print("  None")
            else:
                for item in md:
                    print(f"  {item.strip().replace(os.linesep, ' ')}") 
        else:
            print(f"{tag}: {metadata.get(tag).strip().replace(os.linesep, ' ')}")
    print('Complete:', not metadata.get('incomplete'))
    print("\n" + "-"*40 )