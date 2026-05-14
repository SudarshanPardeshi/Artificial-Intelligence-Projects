import requests
from urllib.parse import quote


HEADERS = {
    "User-Agent": "MeltingPointAI/1.0"
}


def name_to_smiles(compound_name: str):
    try:
        encoded_name = quote(compound_name, safe="")

        cid_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{encoded_name}/cids/JSON"
        )

        cid_response = requests.get(
            cid_url,
            headers=HEADERS,
            timeout=15
        )

        if cid_response.status_code != 200:
            return None

        cid_data = cid_response.json()

        cid = cid_data["IdentifierList"]["CID"][0]

        smiles_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{cid}/property/CanonicalSMILES/JSON"
        )

        smiles_response = requests.get(
            smiles_url,
            headers=HEADERS,
            timeout=15
        )

        if smiles_response.status_code != 200:
            return None

        smiles_data = smiles_response.json()

        smiles = (
            smiles_data["PropertyTable"]["Properties"][0]
            ["CanonicalSMILES"]
        )

        return smiles

    except Exception:
        return None


def smiles_to_name(smiles: str):
    try:
        encoded_smiles = quote(smiles, safe="")

        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
            f"{encoded_smiles}/property/IUPACName/JSON"
        )

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        if response.status_code != 200:
            return None

        data = response.json()

        return data["PropertyTable"]["Properties"][0]["IUPACName"]

    except Exception:
        return None