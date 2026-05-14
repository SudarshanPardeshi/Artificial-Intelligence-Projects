import time
import pandas as pd
import requests
from urllib.parse import quote


INPUT_FILE = "all_smiles_clean.csv"
OUTPUT_FILE = "all_smiles_with_names.csv"


def smiles_to_name(smiles):
    encoded_smiles = quote(smiles, safe="")

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{encoded_smiles}/property/IUPACName/JSON"
    )

    try:
        response = requests.get(url, timeout=15)

        if response.status_code != 200:
            return "Name Not Found"

        data = response.json()

        return data["PropertyTable"]["Properties"][0].get(
            "IUPACName",
            "Name Not Found"
        )

    except Exception:
        return "Name Not Found"


df = pd.read_csv(INPUT_FILE)

df = df.dropna(subset=["SMILES"]).copy()

names = []

total = len(df)

for i, smiles in enumerate(df["SMILES"], start=1):
    print(f"Processing {i}/{total}: {smiles}")

    name = smiles_to_name(smiles)
    names.append(name)

    time.sleep(0.2)

df["Molecule_Name"] = names

df.to_csv(OUTPUT_FILE, index=False)

print(f"Full file created: {OUTPUT_FILE}")
print("Total rows saved:", len(df))