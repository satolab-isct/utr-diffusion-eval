import os
from Bio import SeqIO
import pandas as pd

def fasta_to_csv_in_folder(root):
    fasta_files = [f for f in os.listdir(root) if f.endswith('.fasta')]
    if not fasta_files:
        print(f"[Warning] No fasta files found in {root}")
        return

    for fasta_file in fasta_files:
        fasta_path = os.path.join(root, fasta_file)
        csv_path = fasta_path.replace('.fasta', '.csv')

        if not os.path.exists(csv_path):
            print(f"[Converting] {fasta_file} → {os.path.basename(csv_path)}")
            fasta_to_csv_file(fasta_path)
        else:
            print(f"[Skip] {csv_path} already exists")


def fasta_to_csv_file(fasta_path):
    ids, seqs = [], []
    records = SeqIO.parse(fasta_path, "fasta")
    for record in records:
        ids.append(record.id)
        seqs.append(record.seq)
    data = {'ID': ids, 'Sequence': seqs}
    csv_path = fasta_path.replace('.fasta', '.csv')
    pd.DataFrame(data).to_csv(csv_path, index=False)
    print(f'[Saved] {csv_path}')


if __name__ == '__main__':
    fasta_to_csv_in_folder(root='/gs/bs/tga-satolab-gtex/dai/myscript/dnadiffusion/save/MRL_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4')


