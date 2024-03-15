# AAscore
The Analog Accessibility score (AAscore) is a score that represents the accessibility of analog compounds for a target compound.

<img width="959" alt="README_image" src="https://github.com/U-T100/AAscore/assets/48980320/d946a839-3f1a-41ab-9df8-f6a958301496">

## Installation
To calculate the AAscore, a reaction prediction model (T5Chem) and rxnmapper are required. You can use AAscore_env.yml for setting up the environment, but if it doesn't work, please install the necessary libraries from the following link. If installing these via pip, Python 3.7 is highly recommended.
- [T5Chem](https://github.com/HelloJocelynLu/t5chem)
- [RXNMapper](https://github.com/rxn4chemistry/rxnmapper)

## Usage
Please adjust the arguments of the calculate_AAscore function in AAscore.py to suit your situation appropriately. You don't need to change the other arguments. **Please use aromatic (not kekule) SMILES that have been canonicalized with RDKit for compound data.**
~~~python
input_cpd       = 'COC1=CC=C(CNC(=O)[C@H]2CC(=O)N(C3=CC=C(S(N)(=O)=O)C=C3)[C@@H]2C2=CC=C(Cl)C=C2)C=C1' # Target compound for calculating AAscore
model_path      = 'path/to/your/model/' # Reaction prediction model (T5Chem) path
cpds_data_path  = 'path/to/your/compound_data.tsv' # You can use any compound data for searching candidate reactants
cpds_columnname = 'compounds' # Compounds column name in compound data
save_path       = 'path/to/save_directory/generated_analogs.tsv'
extracted_data  = pd.read_table(cpds_data_path)
caluculate_AAscore(input_cpd, model_path, extracted_data, cpds_columnname, used_reactants_num=7, save_analogs=True, save_path=save_path)
~~~
The template_extractor.py is a file for identifying the reaction center, and modifications have been made to the original [RDChiral](https://github.com/connorcoley/rdchiral). If you want to change parameters such as the radius of the reaction center, please modify the extract_from_reaction function within this file.

## Reference
Takato Ue and Tomoyuki Miyao., Analog Accessibility Score (AAscore) for Rational Compound Selection, URL_to_paper
