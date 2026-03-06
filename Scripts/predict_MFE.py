import subprocess
import pandas as pd
import os

from absl.flags import flag_dict_to_args

from fasta_to_csv import fasta_to_csv_in_folder
RNAFOLD_PATH = "/gs/bs/tga-satolab-gtex/dai/apps/vienna-rna/bin/RNAfold"

def get_mfe(sequence):
    result = subprocess.run([RNAFOLD_PATH], input=sequence.encode(), stdout=subprocess.PIPE)
    lines = result.stdout.decode().split('\n')
    mfe_line = lines[1]
    mfe = float(mfe_line.split('(')[-1].split(')')[0])
    return mfe

# print(get_mfe('CCGGCTATAGCGCGCAGTGCTCGGATGGCAAGGCGTTCAACCGTGACCAC'))
# #CCGGCTATAGCGCGCAGTGCTCGGATGGCAAGGCGTTCAACCGTGACCAC

def get_mfe_batch(sequences):
    input = "\n".join(sequences).encode()
    result = subprocess.run([RNAFOLD_PATH], input=input, stdout=subprocess.PIPE)
    lines = result.stdout.decode().strip().split('\n')
    mfe_batch = []
    for i in range(1, len(lines), 2):
        mfe_line = lines[i]
        mfe = float(mfe_line.split('(')[-1].split(')')[0])
        mfe_batch.append(mfe)
    return mfe_batch

def process_batch_in_folder(root):
    file_list = [f for f in os.listdir(root) if f.endswith('.csv')]
    for file in file_list:
        # read_csv_and_wrire_MFE(os.path.join(root, file), overwrite=True)
        read_csv_and_wrire_MFE_batch(os.path.join(root, file), overwrite=True, batch_size=1000)

def read_csv_and_wrire_MFE(data_path, overwrite=False):
    print(f'[Processing] {data_path}')
    data = pd.read_csv(data_path)
    mfe_list = []
    for idx, seq in enumerate(data['Sequence']):
        mfe = get_mfe(seq)
        mfe_list.append(mfe)
        print(f'\r{idx}/{len(data)}', end='')

    data['MFE'] = mfe_list
    save_path = data_path if overwrite else data_path.replace('.csv', '_with_MFE.csv')
    data.to_csv(save_path, index=False, float_format="%.3f")
    print(f"[Success] Save with MFE: {save_path}")


def read_csv_and_wrire_MFE_batch(data_path, overwrite=False, batch_size:int=5):
    print(f'[Processing] {data_path} with batch size {batch_size}')
    data = pd.read_csv(data_path)
    sequences = data['Sequence'].tolist()
    mfe_list = []

    for i in range(0, len(sequences), batch_size):
        seq_batch = sequences[i : i + batch_size]
        mfe_batch = get_mfe_batch(seq_batch)
        mfe_list.extend(mfe_batch)
        print(f'\r{i}/{len(sequences)}', end='')

    data['MFE'] = mfe_list
    save_path = data_path if overwrite else data_path.replace('.csv', '_with_MFE.csv')
    data.to_csv(save_path, index=False, float_format="%.3f")
    print(f"[Success] Save with MFE: {save_path}")

if __name__ == "__main__":
    folder_path = "/gs/bs/tga-satolab-gtex/dai/myscript/utr-diffusion/save/MFE_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_2000epoch_sample_class_3_[-15, -10, -5]"
    fasta_to_csv_in_folder(folder_path)
    process_batch_in_folder(root=folder_path)