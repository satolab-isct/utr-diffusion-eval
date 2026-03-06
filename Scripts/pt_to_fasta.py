import os
import torch

nucleotides = ['A', 'C', 'G', 'T']

def pt_to_fasta(root: str, filename: str):
    file_path = os.path.join(root, filename)
    print(f"Loading file: {file_path}")
    x = torch.load(file_path)
    lines = []
    for key in x.keys(): # x: {dict:3},
        for step, seqs in enumerate(x[key]):  # x[key]: [201, 1000, 4, 50]
            for num, seq in enumerate(seqs):  # seq: [4, 50]
                # Convert one-hot to string
                sequence = ''.join([nucleotides[n] for n in seq.argmax(axis=0)])
                header = f">_{key}_{step}_{num}"
                lines.append(header)
                lines.append(sequence)

    with open(os.path.join(root, 'all_sequences_denoising_process.fasta'), 'w') as f:
        f.write('\n'.join(lines))
    print("fasta saved")

if __name__ == '__main__':
    # root = "/gs/bs/tga-satolab-gtex/dai/myscript/dnadiffusion/save/MRL_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4"
    # pt_to_fasta(root, filename="all_images_denoising_process.pt")
    root = "/gs/bs/tga-satolab-gtex/dai/myscript/dnadiffusion/repaint_save/MFE_100k_class_3_[-15, -10, -5]_ep_2k_ts_200_beta_0.01_con_1_uncon_0.2_drop_0.2_at_2000epoch"
    pt_to_fasta(root, filename="pattern_single_amino.pt")