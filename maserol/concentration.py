from pathlib import Path

import pandas as pd
import requests

UNIPROT_IDS = {
    "FcR2A": "P12318",
    "FcR2B": "P31994",
    "FcR3A": "P08637",
    "FcR3B": "O75015",
}

DATA_DIR = Path(__file__).parent / "data"
BIOTIN_MASS = 390
SAPE_MASS = 240000

AA_MASS = pd.read_csv(DATA_DIR / "amino-acid-mass.csv")
AA_MASS.set_index("AA", inplace=True)

DETECTION_MASS_PATH = DATA_DIR / "detection-mass.csv"


def fetch_uniprot_sequence(uniprot_id: str):
    """Fetches the amino acid sequence of a UniProt entry by its ID."""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        # Process the FASTA format
        fasta_data = response.text
        # Skip the first line with metadata and join the remaining lines for the sequence
        sequence = "".join(fasta_data.split("\n")[1:])
        return sequence
    else:
        return "Error: Could not retrieve data from UniProt."


def protein_mass(sequence: str):
    return sum(AA_MASS.loc[aa].values[0] for aa in sequence) - 18.015 * (
        len(sequence) - 1
    )


def tetramer_mass(uniprot_id: str):
    rcp_sequence = fetch_uniprot_sequence(uniprot_id)
    rcp_mass = protein_mass(rcp_sequence)
    return 4 * (rcp_mass + BIOTIN_MASS) + SAPE_MASS


def calculate_Fc_detection_molar_mass(FcR: str):
    return tetramer_mass(UNIPROT_IDS[FcR])


def get_Fc_detection_molar_mass_cached(FcR: str):
    return (
        pd.read_csv(DETECTION_MASS_PATH)
        .set_index("Receptor", drop=True)
        .loc[FcR]
        .values[0]
    )


def get_detection_masses():
    masses = {}
    for rcp in UNIPROT_IDS.keys():
        masses[rcp] = calculate_Fc_detection_molar_mass(rcp)
    masses = pd.DataFrame(list(masses.items()), columns=["Receptor", "Mass"]).set_index(
        "Receptor", drop=True
    )
    return masses


def mass_conc_to_molarity(conc_ug_ml, molar_mass_Da):
    return conc_ug_ml / molar_mass_Da * 1e-3


def get_Fc_detection_molarity(FcR: str, conc_ug_ml: float):
    return mass_conc_to_molarity(conc_ug_ml, get_Fc_detection_molar_mass_cached(FcR))


def cache_detection_masses():
    get_detection_masses().to_csv(DETECTION_MASS_PATH)


if __name__ == "__main__":
    cache_detection_masses()
