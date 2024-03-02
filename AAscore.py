import os
import re
import pandas as pd
import itertools
import time
from collections import Counter
from transformers import T5ForConditionalGeneration
from t5chem import SimpleTokenizer
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdFMCS
from rxnmapper import RXNMapper
from template_extractor import extract_from_reaction

def get_elements_from_list(input : list, wanted_num : int) -> list:
    '''Return the first wanted_num elements from a list.

    Example:
    >>> get_elements_from_list([1, 2, 3, 4, 5], 3)
    [1, 2, 3]
    '''
    if len(input) < wanted_num:
        return input
    else:
        return input[:wanted_num]

def get_element_with_max_length_in_values(dictionary):
    '''Return the key-value pair with the longest value in a dictionary.

    Example:
    >>> get_element_with_max_length_in_values({'a': [1, 2, 3], 'b': ['wifi', 'oxygen']})
    {'a': [1, 2, 3]}
    '''
    if len(dictionary) == 1:
        return dictionary
    max_key = max(dictionary, key=lambda k: len(dictionary[k]))
    return {max_key: dictionary[max_key]}

def remove_original_reactant_from_reac_with_rr(reac_with_rr : dict) -> dict:
    '''
    Inputed reac_with_rr looks like {'[CH3:1][O:2][C:3](=[O:4])[c:5]1[s:6][cH:7][cH:8][c:9]1[NH2:10]': ['CC(=O)c1sccc1N', 'CCOC(=O)c1sccc1N', 'COC(=O)c1sccc1N']}
    ex reac_with_rr = {'compoundX' : ['compoundA', 'compoundX', 'compoundB']}
       return       = {'compoundX' : ['compoundA', 'compoundB']}
    '''
    wo_mapping       = {Chem.MolToSmiles(remove_atommapping(Chem.MolFromSmiles(key))): [Chem.MolToSmiles(Chem.MolFromSmiles(value)) for value in values ] for key, values in reac_with_rr.items()}
    no_key_in_values = {key: list(set(values) - set(key.split('.'))) for key, values in wo_mapping.items()}
    return no_key_in_values

def remove_atommapping(mol):
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return mol

def seperate_reaction_center_by_round_bracket(string) -> list:
    '''
    example
    string = '(ab.zyy).(wifis.u4[CH]2).(a.cxyz)'
    return = ['ab.zyy', 'wifis.u4[CH]2', 'a.cxyz']
    '''
    result     = string.split(').(')
    # Remove '(' from the first element and ')' from the last element.
    result[0]  = result[0][1:]
    result[-1] = result[-1][:-1]
    return result

def get_elements_of_smallest_molecule(dictionary) -> dict:
    '''May return the multiple compounds with the lowest number of heavy atoms among the keys in the dictionary and the corresponding value.
    This function is a multiple version of get_element_of_smallest_molecule function.
    Input keys of dictionary must be SMILES.
    '''
    min_heavy_atom_num = min(Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() for smiles in dictionary)
    return {smiles: rc for smiles, rc in dictionary.items() if Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() == min_heavy_atom_num}

def replace_nodot_with_none(value):
    if isinstance(value, str) and '.' in value:
        return value
    return None

def replace_deuterated(smi):
    return re.sub('\[2H\]', r'[H]', smi)

def sort_smiles_byMW(smiles_list):
    # Find the molecular weight of the SMILES in the list and sort them in ascending order.
    if type(smiles_list) is not list:
        return smiles_list
    return sorted(smiles_list, key=lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))

def add_leaving_group_to_product(atommapped_rxn):
    # This function adds atomic groups without atom mapping in the reactant to the product.
    reactants_and_pro = atommapped_rxn.split('>>')
    reactants = reactants_and_pro[0].split('.')
    products  = reactants_and_pro[1].split('.')
    reactant_rw_mols = [Chem.RWMol(Chem.MolFromSmiles(reactant)) for reactant in reactants]
    for reactant, reactant_rw_mol in zip(reactants, reactant_rw_mols):
        mapped_atom_indexes = [atom.GetIdx() for atom in reactant_rw_mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        if len(mapped_atom_indexes) == 0:
            products.append(reactant)
            continue
        if len(mapped_atom_indexes) == reactant_rw_mol.GetNumAtoms():
            continue
        mapped_atom_indexes.sort(reverse=True)
        for index in mapped_atom_indexes:
            reactant_rw_mol.RemoveAtom(index)
        wo_mapped_atomic_group = Chem.MolToSmiles(reactant_rw_mol.GetMol())
        products.append(wo_mapped_atomic_group)
    return reactants_and_pro[0] + '>>' + '.'.join(products)

class MultiTaskModel:
    def __init__(self, pretrain_path):
        self.model = T5ForConditionalGeneration.from_pretrained(pretrain_path) # for seq2seq tasks
        self.tokenizer = SimpleTokenizer(vocab_file=os.path.join(pretrain_path, 'vocab.pt'))

    def predict(self, input_cpd, task_type, num_preds, num_beam):
        task2prefix = {
            'product'   : 'Product:',
            'reactants' : 'Reactants:',
            'reagents'  : 'Reagents:'
        }

        self.product  = input_cpd
        input_cpd_mol = Chem.MolFromSmiles(input_cpd)
        if input_cpd_mol is None:
            return None
        input_cpd      = Chem.MolToSmiles(input_cpd_mol) # Convert smiles to canonical smiles with rdkit.
        if task_type   == 'reactants':
            input_seq  = task2prefix[task_type] + input_cpd
        elif task_type == 'product':
            input_seq  = task2prefix[task_type] + input_cpd + '>>'

        inputs = self.tokenizer.encode(input_seq, return_tensors='pt')
        output = self.model.generate(input_ids=inputs, max_length=300, early_stopping=True, num_beams=num_beam, num_return_sequences=num_preds)
        if task_type == 'reactants':
            self.predicted_list = list()
            for pred in output:
                if Chem.MolFromSmiles(self.tokenizer.decode(pred, skip_special_tokens=True)) is not None:
                    self.predicted_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(self.tokenizer.decode(pred, skip_special_tokens=True))))
            # Remove duplicates while maintaining the order.
            self.predicted_list = sorted(set(self.predicted_list), key=self.predicted_list.index)
            return input_cpd, self.predicted_list
        elif task_type == 'product':
            for pred in output:
                if Chem.MolFromSmiles(self.tokenizer.decode(pred, skip_special_tokens=True)) is not None:
                    return Chem.MolToSmiles(Chem.MolFromSmiles(self.tokenizer.decode(pred, skip_special_tokens=True)))
                else:
                    return None

def predicted_reactants_curation(canon_cpd, predicted_reactants):
    # Removing predicted reactants consisting of only one compound.
    predicted_reactants = [reactant for reactant in predicted_reactants if replace_nodot_with_none(reactant) is not None]
    # Unique components of reactants. ex 'Cl.Cl.Oc1ccccc1' -> 'Cl.Oc1ccccc1'
    predicted_reactants = [replace_deuterated('.'.join(set(reactant.split('.')))) for reactant in predicted_reactants]
    # Sort components of reactants by molecular weight in ascending order.
    predicted_reactants = ['.'.join(sort_smiles_byMW(reactant.split('.'))) for reactant in predicted_reactants]
    # Remove duplicates while maintaining the order.
    predicted_reactants = sorted(set(predicted_reactants), key=predicted_reactants.index) 
    predicted_reactants = [Chem.MolToSmiles(Chem.MolFromSmiles(reactant)) for reactant in predicted_reactants if Chem.MolFromSmiles(reactant) is not None]
    # Remove invalid reactants(product is in reactants).
    predicted_reactants = [reactant for reactant in predicted_reactants if canon_cpd not in reactant]
    # Remove predicted reactants with formal charge.
    predicted_reactants = [reactants for reactants in predicted_reactants if all(Chem.GetFormalCharge(Chem.MolFromSmiles(reactant)) == 0 for reactant in reactants.split('.'))]
    return predicted_reactants

def extract_candidate_reactants_from_cpds(extracted_data, cpds_columnname, smallest_reactant_with_rc, only_rc=False, only_similarity=False, rc_and_similarity=True, similarity_threshold=0.4, match_rc_num_in_reactant=True, narrow_down_database=True):
    '''This function retrieves compounds with all reaction center of the reactant under consideration.
    RC = Reaction Center
    RR = Replaced Reactant (smallest predicted reactant)
    CR = Candidate Reactant
    only_rc : If True, consider only the RC of the RR when searching for candidate reactants.
    only_similarity : If True, consider only the similarity of the RR when searching for candidate reactants.
    rc_and_similarity : If True, consider both the RC and the similarity of the RR when searching for candidate reactants (recommended). 
    similarity_threshold : The similarity between the RR and the CR must be greater than or equal to this value.
    match_rc_num_in_reactant : If True, the number of reaction centers in the RR and the number of reaction centers in the candidate reactant must be the same.
    narrow_down_database : If True, narrow down the compound database to be searched by the number of heavy atoms of the RR. When the size of the compound database is large, it can significantly reduce the time required.
    ex result = [{'reacA': ['rr1', 'rr2'], 'reacB': ['rr3']}, {'reacC': ['rr4'], 'reacD': ['rr5', 'rr6', 'rr7']}, ... ,{'reacX': ['rrN']}]
    '''
    cpds = extracted_data[cpds_columnname]
    extracted_data['compound_mols'] = cpds.map(Chem.MolFromSmiles)
    extracted_data.dropna(subset=['compound_mols'], inplace=True)
    extracted_data.reset_index(drop=True, inplace=True)
    mols = extracted_data['compound_mols']
    if narrow_down_database:
        extracted_data['heavy_atoms_num'] = mols.map(lambda mol: mol.GetNumHeavyAtoms())
    result = list()
    unexpected_rc_num = 0
    unexpected_reactant_with_rc = list()
    for reactant_with_rc in smallest_reactant_with_rc:
        temp_reactant = list()
        temp_candidate_reactants = list()
        for reactant, reaction_centers in reactant_with_rc.items():
            seperated_reaction_centers = reaction_centers.split('.')
            if len(seperated_reaction_centers) == 0:
                continue
            temp_reactant.append(reactant)
            if narrow_down_database:
                reactant_heavy_atoms_num = Chem.MolFromSmiles(reactant).GetNumHeavyAtoms()
                if reactant_heavy_atoms_num <= 27:
                    selected_data = extracted_data[(extracted_data['heavy_atoms_num'] <= reactant_heavy_atoms_num+9) & (extracted_data['heavy_atoms_num'] >= reactant_heavy_atoms_num-3)]
                else:
                    selected_data = extracted_data[(extracted_data['heavy_atoms_num'] <= reactant_heavy_atoms_num-7) & (extracted_data['heavy_atoms_num'] >= reactant_heavy_atoms_num-17)]
                cpds = selected_data[cpds_columnname]
                mols = selected_data['compound_mols']
            rc_mols  = [Chem.MolFromSmarts(rc) for rc in seperated_reaction_centers]
            if match_rc_num_in_reactant:
                wo_mapping_rc = [Chem.MolToSmarts(remove_atommapping(rc_mol)) for rc_mol in rc_mols]
                counter       = Counter(wo_mapping_rc)
                duplicated_rc_dict = {duplicated_rc : count for duplicated_rc, count in counter.items() if count > 1}
                if len(duplicated_rc_dict) > 1:
                    unexpected_rc_num += 1
                    unexpected_reactant_with_rc.append(reactant_with_rc)
                    continue

            # Obtain compounds with all reaction centers of the reactant under consideration.
            if (only_rc) and (not match_rc_num_in_reactant):
                candidate_reactants = [cpd for cpd, cpd_mol in zip(cpds, mols) if all(cpd_mol.HasSubstructMatch(rc_mol) for rc_mol in rc_mols)]
            if (only_rc) and (match_rc_num_in_reactant):
                if (len(rc_mols) == 1) or (len(duplicated_rc_dict) == 0):
                    candidate_reactants = [cpd for cpd, cpd_mol in zip(cpds, mols) if all(cpd_mol.HasSubstructMatch(rc_mol) for rc_mol in rc_mols)]
                elif len(duplicated_rc_dict) == 1:
                    duplicated_rc      = list(duplicated_rc_dict.keys())[0]
                    candidate_reactants = [cpd for cpd, cpd_mol in zip(cpds, mols) if (all(cpd_mol.HasSubstructMatch(rc_mol) for rc_mol in rc_mols)) and (duplicated_rc_dict[duplicated_rc] == len(cpd_mol.GetSubstructMatches(Chem.MolFromSmarts(duplicated_rc))))]
            
            if (only_similarity) and (not match_rc_num_in_reactant):
                reactant_ecfp      = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactant), 2, 8192)
                candidate_reactants = [cpd for cpd, cpd_mol in zip(cpds, mols) if DataStructs.TanimotoSimilarity(reactant_ecfp, AllChem.GetMorganFingerprintAsBitVect(cpd_mol, 2, 8192)) >= similarity_threshold]
            if (only_similarity) and (match_rc_num_in_reactant):
                if (len(rc_mols) == 1) or (len(duplicated_rc_dict) == 0):
                    reactant_ecfp      = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactant), 2, 8192)
                    candidate_reactants = [cpd for cpd, cpd_mol in zip(cpds, mols) if DataStructs.TanimotoSimilarity(reactant_ecfp, AllChem.GetMorganFingerprintAsBitVect(cpd_mol, 2, 8192)) >= similarity_threshold]
                elif len(duplicated_rc_dict) == 1:
                    duplicated_rc      = list(duplicated_rc_dict.keys())[0]
                    reactant_ecfp      = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactant), 2, 8192)
                    candidate_reactants = [cpd for cpd, cpd_mol in zip(cpds, mols) if (DataStructs.TanimotoSimilarity(reactant_ecfp, AllChem.GetMorganFingerprintAsBitVect(cpd_mol, 2, 8192)) >= similarity_threshold) and (duplicated_rc_dict[duplicated_rc] == len(cpd_mol.GetSubstructMatches(Chem.MolFromSmarts(duplicated_rc))))]
            
            if (rc_and_similarity) and (not match_rc_num_in_reactant):
                reactant_ecfp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactant), 2, 8192)
                candidate_reactants = [cpd for cpd, cpd_mol in zip(cpds, mols) if (all(cpd_mol.HasSubstructMatch(rc_mol) for rc_mol in rc_mols)) and (DataStructs.TanimotoSimilarity(reactant_ecfp, AllChem.GetMorganFingerprintAsBitVect(cpd_mol, 2, 8192)) >= similarity_threshold)]
            if (rc_and_similarity) and (match_rc_num_in_reactant):
                if (len(rc_mols) == 1) or (len(duplicated_rc_dict) == 0):
                    reactant_ecfp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactant), 2, 8192)
                    candidate_reactants = [cpd for cpd, cpd_mol in zip(cpds, mols) if (all(cpd_mol.HasSubstructMatch(rc_mol) for rc_mol in rc_mols)) and (DataStructs.TanimotoSimilarity(reactant_ecfp, AllChem.GetMorganFingerprintAsBitVect(cpd_mol, 2, 8192)) >= similarity_threshold)]
                elif len(duplicated_rc_dict) == 1:
                    duplicated_rc      = list(duplicated_rc_dict.keys())[0]
                    reactant_ecfp      = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactant), 2, 8192)
                    candidate_reactants = [cpd for cpd, cpd_mol in zip(cpds, mols) if (all(cpd_mol.HasSubstructMatch(rc_mol) for rc_mol in rc_mols)) and (DataStructs.TanimotoSimilarity(reactant_ecfp, AllChem.GetMorganFingerprintAsBitVect(cpd_mol, 2, 8192)) >= similarity_threshold) and (duplicated_rc_dict[duplicated_rc] == len(cpd_mol.GetSubstructMatches(Chem.MolFromSmarts(duplicated_rc))))]

            temp_candidate_reactants.append(candidate_reactants)
        reactant_with_cr = dict(zip(temp_reactant, temp_candidate_reactants))
        result.append(reactant_with_cr)
    print('Unexpected number of reaction center: ', unexpected_rc_num)
    print('Unexpected reactant with reaction center: ', unexpected_reactant_with_rc)
    return result

def extract_candidate_reactants_from_mols(extracted_data, cpds_columnname, ecfp_columnname, mols_columnname, smallest_reactant_with_rc, only_rc=False, only_similarity=False, rc_and_similarity=True, similarity_threshold=0.4, match_rc_num_in_reactant=True, narrow_down_database=True):
    # This function can be used when the compound database is already converted to mols and ecfps.
    result = list()
    unexpected_rc_num = 0
    unexpected_reactant_with_rc = list()
    if not narrow_down_database:
        zinc_cpds  = extracted_data[cpds_columnname]
        zinc_ecfps = extracted_data[ecfp_columnname]
        zinc_mols  = extracted_data[mols_columnname]
    for reactant_with_rc in smallest_reactant_with_rc:
        temp_reactant = list()
        temp_candidate_reactants = list()
        for reactant, reaction_centers in reactant_with_rc.items():
            seperated_reaction_centers = reaction_centers.split('.')
            if len(seperated_reaction_centers) == 0:
                continue
            temp_reactant.append(reactant)
            if narrow_down_database:
                reactant_heavy_atoms_num = Chem.MolFromSmiles(reactant).GetNumHeavyAtoms()
                if reactant_heavy_atoms_num <= 27:
                    selected_data = extracted_data[(extracted_data['heavy_atoms_num'] <= reactant_heavy_atoms_num+9) & (extracted_data['heavy_atoms_num'] >= reactant_heavy_atoms_num-3)]
                else:
                    selected_data = extracted_data[(extracted_data['heavy_atoms_num'] <= reactant_heavy_atoms_num-7) & (extracted_data['heavy_atoms_num'] >= reactant_heavy_atoms_num-17)]
                zinc_cpds  = selected_data[cpds_columnname]
                zinc_ecfps = selected_data[ecfp_columnname]
                zinc_mols  = selected_data[mols_columnname]
            rc_mols = [Chem.MolFromSmarts(rc) for rc in seperated_reaction_centers]
            if match_rc_num_in_reactant:
                wo_mapping_rc = [Chem.MolToSmarts(remove_atommapping(rc_mol)) for rc_mol in rc_mols]
                counter       = Counter(wo_mapping_rc)
                duplicated_rc_dict = {duplicated_rc : count for duplicated_rc, count in counter.items() if count > 1}
                if len(duplicated_rc_dict) > 1:
                    unexpected_rc_num += 1
                    unexpected_reactant_with_rc.append(reactant_with_rc)
                    continue

            # Obtain compounds with all reaction centers of the reactant under consideration.
            if (only_rc) and (not match_rc_num_in_reactant):
                candidate_reactants = [cpd for cpd, cpd_mol in zip(zinc_cpds, zinc_mols) if all(cpd_mol.HasSubstructMatch(rc_mol) for rc_mol in rc_mols)]
            if (only_rc) and (match_rc_num_in_reactant):
                if (len(rc_mols) == 1) or (len(duplicated_rc_dict) == 0):
                    candidate_reactants = [cpd for cpd, cpd_mol in zip(zinc_cpds, zinc_mols) if all(cpd_mol.HasSubstructMatch(rc_mol) for rc_mol in rc_mols)]
                elif len(duplicated_rc_dict) == 1:
                    duplicated_rc      = list(duplicated_rc_dict.keys())[0]
                    candidate_reactants = [cpd for cpd, cpd_mol in zip(zinc_cpds, zinc_mols) if (all(cpd_mol.HasSubstructMatch(rc_mol) for rc_mol in rc_mols)) and (duplicated_rc_dict[duplicated_rc] == len(cpd_mol.GetSubstructMatches(Chem.MolFromSmarts(duplicated_rc))))]
            
            if (only_similarity) and (not match_rc_num_in_reactant):
                reactant_ecfp      = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactant), 2, 8192)
                candidate_reactants = [cpd for cpd, cpd_ecfp in zip(zinc_cpds, zinc_ecfps) if DataStructs.TanimotoSimilarity(reactant_ecfp, cpd_ecfp) >= similarity_threshold]
            if (only_similarity) and (match_rc_num_in_reactant):
                if (len(rc_mols) == 1) or (len(duplicated_rc_dict) == 0):
                    reactant_ecfp      = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactant), 2, 8192)
                    candidate_reactants = [cpd for cpd, cpd_ecfp in zip(zinc_cpds, zinc_ecfps) if DataStructs.TanimotoSimilarity(reactant_ecfp, cpd_ecfp) >= similarity_threshold]
                elif len(duplicated_rc_dict) == 1:
                    duplicated_rc      = list(duplicated_rc_dict.keys())[0]
                    reactant_ecfp      = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactant), 2, 8192)
                    candidate_reactants = [cpd for cpd, cpd_mol, cpd_ecfp in zip(zinc_cpds, zinc_mols, zinc_ecfps) if (DataStructs.TanimotoSimilarity(reactant_ecfp, cpd_ecfp) >= similarity_threshold) and (duplicated_rc_dict[duplicated_rc] == len(cpd_mol.GetSubstructMatches(Chem.MolFromSmarts(duplicated_rc))))]
            
            if (rc_and_similarity) and (not match_rc_num_in_reactant):
                reactant_ecfp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactant), 2, 8192)
                candidate_reactants = [cpd for cpd, cpd_mol, cpd_ecfp in zip(zinc_cpds, zinc_mols, zinc_ecfps) if (all(cpd_mol.HasSubstructMatch(rc_mol) for rc_mol in rc_mols)) and (DataStructs.TanimotoSimilarity(reactant_ecfp, cpd_ecfp) >= similarity_threshold)]
            if (rc_and_similarity) and (match_rc_num_in_reactant):
                if (len(rc_mols) == 1) or (len(duplicated_rc_dict) == 0):
                    reactant_ecfp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactant), 2, 8192)
                    candidate_reactants = [cpd for i, (cpd, cpd_mol, cpd_ecfp) in enumerate(zip(zinc_cpds, zinc_mols, zinc_ecfps)) if (all(cpd_mol.HasSubstructMatch(rc_mol) for rc_mol in rc_mols)) and (DataStructs.TanimotoSimilarity(reactant_ecfp, cpd_ecfp) >= similarity_threshold)]
                elif len(duplicated_rc_dict) == 1:
                    duplicated_rc      = list(duplicated_rc_dict.keys())[0]
                    reactant_ecfp      = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactant), 2, 8192)
                    candidate_reactants = [cpd for i, (cpd, cpd_mol, cpd_ecfp) in enumerate(zip(zinc_cpds, zinc_mols, zinc_ecfps)) if (all(cpd_mol.HasSubstructMatch(rc_mol) for rc_mol in rc_mols)) and (DataStructs.TanimotoSimilarity(reactant_ecfp, cpd_ecfp) >= similarity_threshold) and (duplicated_rc_dict[duplicated_rc] == len(cpd_mol.GetSubstructMatches(Chem.MolFromSmarts(duplicated_rc))))]

            temp_candidate_reactants.append(candidate_reactants)
        reactant_with_cr = dict(zip(temp_reactant, temp_candidate_reactants))
        result.append(reactant_with_cr)
    print('Unexpected number of reaction center: ', unexpected_rc_num)
    print('Unexpected reactant with reaction center: ', unexpected_reactant_with_rc)
    return result

def replace_original_reac_to_virtual_reac(multi_reactants : list, reactants_with_rr):
    all_virtual_reactants_combi = list()
    for reactants, reac_with_rr in zip(multi_reactants, reactants_with_rr):
        virtual_reactants_combi = [reactants.replace(key, value) for key in reac_with_rr for value in reac_with_rr[key] if key in reactants]
        all_virtual_reactants_combi.append(virtual_reactants_combi)
    return all_virtual_reactants_combi

def extract_mcs_analogs_from_cpds(input_cpd, cpds, mcs_ratio=0.67):
    if len(cpds) == 0:
        return list()
    input_cpd_mol = Chem.MolFromSmiles(input_cpd)
    cpd_mols      = [Chem.MolFromSmiles(cpd) for cpd in cpds]
    mcs_list      = [rdFMCS.FindMCS([input_cpd_mol, cpd_mol]) for cpd_mol in cpd_mols]
    analogs_list  = [Chem.MolToSmiles(cpd_mol) for mcs, cpd_mol in zip(mcs_list, cpd_mols) if (mcs.numAtoms/cpd_mol.GetNumHeavyAtoms() >= mcs_ratio) and (mcs.numAtoms/input_cpd_mol.GetNumHeavyAtoms() >= mcs_ratio)]
    input_cpd = Chem.MolToSmiles(input_cpd_mol) # convert input cpd to be canonical with rdkit.
    # remove input cpd from generated analogs.
    while input_cpd in analogs_list:
        analogs_list.remove(input_cpd)
    return analogs_list

def remove_reactants_with_same_rc(smallest_reactant_with_rc, reactant_with_rc_flag=True, rc_flag=False):
    wo_mapping_smallest_reactant_with_rc = [{Chem.MolToSmiles(remove_atommapping(Chem.MolFromSmiles(key))) : Chem.MolToSmarts(remove_atommapping(Chem.MolFromSmarts(value)))} for smallest_reac_with_rc in smallest_reactant_with_rc for key, value in smallest_reac_with_rc.items()]
    same_reac_with_rc_index = list()
    same_rc_index           = list()
    seen_reactant_with_rc   = list()
    seen_rc                 = list()
    for i, smallest_reac_with_rc in enumerate(wo_mapping_smallest_reactant_with_rc):
        reac_with_rc = set(smallest_reac_with_rc.items())
        rc           = set(smallest_reac_with_rc.values())
        if (reac_with_rc in seen_reactant_with_rc) and (reactant_with_rc_flag):
            same_reac_with_rc_index.append(i)
        elif (reac_with_rc not in seen_reactant_with_rc) and (reactant_with_rc_flag):
            seen_reactant_with_rc.append(reac_with_rc)
        elif (rc in seen_rc) and (rc_flag):
            same_rc_index.append(i)
        elif (rc not in seen_rc) and (rc_flag):
            seen_rc.append(rc)
    if reactant_with_rc_flag:
        smallest_reactant_with_rc = [reac_with_rc for i, reac_with_rc in enumerate(smallest_reactant_with_rc) if i not in same_reac_with_rc_index]
    elif rc_flag:
        smallest_reactant_with_rc = [reac_with_rc for i, reac_with_rc in enumerate(smallest_reactant_with_rc) if i not in same_rc_index]
    return smallest_reactant_with_rc

def caluculate_AAscore(input_cpd, model_path, extracted_data, cpds_columnname, used_reactants_num=7, save_analogs=True, save_path=None):
    '''
    used_reactants_num : The maximum number of predicted reactants to be used.
    save_analogs       : If True, save the generated analogs with corresponding reactants as dataframe.
    '''
    multitask   = MultiTaskModel(pretrain_path=model_path)
    retro_start = time.time()
    # Input product must consist of a single compound.
    canon_cpd, predicted_reactants = multitask.predict(input_cpd=input_cpd, task_type='reactants', num_preds=30, num_beam=30)
    curated_predicted_reactants    = predicted_reactants_curation(canon_cpd, predicted_reactants)
    # Check if the product can be reproduced from the predicted reactants (round-trip).
    round_trip_reactants = list()        
    for reactant in curated_predicted_reactants:
        predicted_product = multitask.predict(input_cpd=reactant, task_type='product', num_preds=1, num_beam=5)
        if canon_cpd in predicted_product:
            round_trip_reactants.append(reactant)
    if len(round_trip_reactants) == 0:
        print('Since there were no valid predicted reactants after curation, the score cannot be calculated.')
        return
    print('Reactants prediction and round-trip time: ', time.time() - retro_start)

    rxns = [f'{reactant}>>{canon_cpd}' for reactant in round_trip_reactants]
    rxn_mapper = RXNMapper()
    mapped_rxns_with_confi = rxn_mapper.get_attention_guided_atom_maps(rxns, canonicalize_rxns=False)
    mapped_rxns_wo_lg      = [rxn_with_confi['mapped_rxn'] for rxn_with_confi in mapped_rxns_with_confi] # lg means leaving group.
    mapped_rxns_with_lg    = [add_leaving_group_to_product(rxn) for rxn in mapped_rxns_wo_lg]
    # Atom mapping is also done on the leaving group.
    mapped_rxns_with_confi = rxn_mapper.get_attention_guided_atom_maps(mapped_rxns_with_lg, canonicalize_rxns=False)
    atom_mapped_rxns       = [rxn_with_confi['mapped_rxn'] for rxn_with_confi in mapped_rxns_with_confi]
    atom_mapped_reactants = [rxn.split('>>')[0] for rxn in atom_mapped_rxns]
    atom_mapped_products  = [rxn.split('>>')[1] for rxn in atom_mapped_rxns]
    temp_list      = list()
    error_index    = list()
    error_reaction = list()
    reactants_with_rc = list()
    smallest_reactant_with_rc = list()
    for i, (reactants, product) in enumerate(zip(atom_mapped_reactants, atom_mapped_products)):
        reaction = {'reactants': reactants, 'products': product, '_id': i}
        temp     = extract_from_reaction(reaction)
        if len(temp) == 1:
            error_index.append(i)
            error_reaction.append(reaction)
            continue
        necesary_temp = {
        'reactants' : reactants,
        'product' : product,
        'product_rc' : temp['products'],
        'reactants_rc' : temp['reactants'],
        'reaction_smarts' : temp['reaction_smarts'],
        }
        temp_list.append(necesary_temp)
        seperated_reactants = reactants.split('.')
        seperated_rc = seperate_reaction_center_by_round_bracket(necesary_temp['reactants_rc'])
        if len(seperated_reactants) == len(seperated_rc):
            temp_reactants_with_rc = dict(zip(seperated_reactants, seperated_rc))
        else:
            reac_combi    = list(itertools.combinations(seperated_reactants, len(seperated_rc)))
            reac_rc_combi = list()
            for combi in reac_combi:
                temp_reac_rc = dict()
                for reac, rc in zip(combi, seperated_rc):
                    temp_reac_rc[reac] = rc
                reac_rc_combi.append(temp_reac_rc)
            temp_reactants_with_rc = [reac_rc for reac_rc in reac_rc_combi if all(Chem.MolFromSmiles(reac).HasSubstructMatch(Chem.MolFromSmarts(rc)) for reac, rc in reac_rc.items())]
            # If all reaction centers are correctly assigned to reactants, the number of elements in temp_reactants_with_rc will be 1.
            if len(temp_reactants_with_rc) == 1:
                temp_reactants_with_rc = temp_reactants_with_rc[0]
            elif len(temp_reactants_with_rc) != 1:
                temp_reactants_with_rc = dict()
                reactants_with_rc.append(temp_reactants_with_rc)
                smallest_reactant_with_rc.append(temp_reactants_with_rc)
                continue
        reactants_with_rc.append(temp_reactants_with_rc)
        smallest_reactant_with_rc.append(get_elements_of_smallest_molecule(temp_reactants_with_rc))
    curated_smallest_reactant_with_rc = remove_reactants_with_same_rc(smallest_reactant_with_rc, reactant_with_rc_flag=True, rc_flag=False)
    curated_smallest_reactant_with_rc = get_elements_from_list(curated_smallest_reactant_with_rc, used_reactants_num)
    smallest_reactant_with_rc_indices = [smallest_reactant_with_rc.index(item) for item in curated_smallest_reactant_with_rc]
    selected_round_trip_reactants     = [round_trip_reactants[i] for i in smallest_reactant_with_rc_indices]
    
    extract_start    = time.time()
    reactant_with_cr = extract_candidate_reactants_from_cpds(extracted_data,
                                                             cpds_columnname=cpds_columnname,
                                                             smallest_reactant_with_rc=curated_smallest_reactant_with_rc,
                                                             only_rc=False, only_similarity=False, rc_and_similarity=True, similarity_threshold=0.4, match_rc_num_in_reactant=True, narrow_down_database=True)
    print('Extraction time: ', time.time() - extract_start)
    reactant_with_cr = [remove_original_reactant_from_reac_with_rr(reac_with_rr) for reac_with_rr in reactant_with_cr]
    # If there are multiple reactants with the lowest number of heavy atoms, use the one with the highest number of extracted compounds.
    reactant_with_cr = [get_element_with_max_length_in_values(reac_with_rr) for reac_with_rr in reactant_with_cr]
    vr_combinations  = replace_original_reac_to_virtual_reac(selected_round_trip_reactants, reactant_with_cr)
    flattened_vr     = [vr for vr_combi in vr_combinations for vr in vr_combi]
    predicted_products = list()
    vr_num = 0

    forward_start = time.time()
    for reactants, vr_combi in zip(selected_round_trip_reactants, vr_combinations):
        temp_predicted_products = list()
        for vr in vr_combi:
            vr_num += 1
            predicted_product = multitask.predict(input_cpd=vr, task_type='product', num_preds=1, num_beam=5)
            if (predicted_product is None) or (predicted_product in vr) or ('.' in predicted_product):
                # To correspond to flattened_vr and predicted_products.
                temp_predicted_products.append(None)
                continue
            temp_predicted_products.append(predicted_product)
        predicted_products.append(temp_predicted_products)
    flattened_predicted_products = [predicted_product for predicted_pros in predicted_products for predicted_product in predicted_pros]
    print('Product prediction time: ', time.time() - forward_start)

    predicted_analogs_num_per_reactant = list()
    for predicted_pros in predicted_products:
        predicted_pros_wo_none = [predicted_product for predicted_product in predicted_pros if predicted_product is not None]
        predicted_analogs_for_ratio = extract_mcs_analogs_from_cpds(input_cpd, set(predicted_pros_wo_none))
        predicted_analogs_num_per_reactant.append(len(predicted_analogs_for_ratio))
    flattened_predicted_products_wo_none = [flattened_predicted_product for flattened_predicted_product in flattened_predicted_products if flattened_predicted_product is not None]
    predicted_analogs = extract_mcs_analogs_from_cpds(input_cpd, flattened_predicted_products_wo_none)
    unique_predicted_analogs = list(dict.fromkeys(predicted_analogs))
    if save_analogs:
        vr_and_predicted_analogs = [(vr, predicted_product) for vr, predicted_product in zip(flattened_vr, flattened_predicted_products) if predicted_product in unique_predicted_analogs]            
        df = pd.DataFrame(vr_and_predicted_analogs, columns=['reactants', 'generated_analogs'])
        df.drop_duplicates(subset=['generated_analogs'], keep='first', inplace=True)
        df.to_csv(save_path, sep='\t', index=False)

    # Calculate the AAscore.
    ratio = sum(predicted_analogs_num_per_reactant)/vr_num
    predicted_analogs_num = len(unique_predicted_analogs)
    print('Conversion rate to analogs: ', ratio)
    print('Analog Accessibility score (AAscore): ', predicted_analogs_num)
    
if __name__ == '__main__':
    if 1:
        input_cpd       = 'COC1=CC=C(CNC(=O)[C@H]2CC(=O)N(C3=CC=C(S(N)(=O)=O)C=C3)[C@@H]2C2=CC=C(Cl)C=C2)C=C1'
        model_path      = 'finetuned_model/'
        cpds_data_path  = 'data/curated_zinc15_subset.tsv'
        cpds_columnname = 'washed_isomeric_aromatic_smiles'
        save_path       = 'data/sample_generated_analogs.tsv'
        extracted_data  = pd.read_table(cpds_data_path)
        caluculate_AAscore(input_cpd, model_path, extracted_data, cpds_columnname, used_reactants_num=7, save_analogs=True, save_path=save_path)