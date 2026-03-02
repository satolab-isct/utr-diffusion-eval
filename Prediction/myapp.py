import streamlit as st
from Bio import SeqIO
import torch
import torch.nn as nn
import pandas as pd

import esm
from esm.data import *
from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS

from esm import Alphabet, FastaBatchedDataset

from io import StringIO

seed = 1337
torch.manual_seed(seed)

global modelfile, layers, heads, embed_dim, batch_toks, inp_len, device
modelfile = 'model.pt'
# modelfile = '/gs/bs/tga-satolab-gtex/dai/myscript/UTR_LM_prediction/Prediction_Modal/model.pt'


layers = 6
heads = 16
embed_dim = 128
# batch_toks = 4096
batch_toks = 4096 * 8 * 8

inp_len = 128 #50
device = 'cuda'

alphabet = Alphabet(mask_prob=0.0, standard_toks='AGCT')
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7,
                               '<mask>': 8, '<sep>': 9}


class CNN_linear(nn.Module):
    def __init__(self,
                 border_mode='same', filter_len=8, nbr_filters=120,
                 dropout1=0, dropout2=0):
        super(CNN_linear, self).__init__()

        self.embedding_size = embed_dim
        self.border_mode = border_mode
        self.inp_len = inp_len
        self.nodes = 40
        self.cnn_layers = 0
        self.filter_len = filter_len
        self.nbr_filters = nbr_filters
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = 0.5

        self.esm2 = ESM2_SISS(num_layers=layers,
                              embed_dim=embed_dim,
                              attention_heads=heads,
                              alphabet=alphabet)

        self.conv1 = nn.Conv1d(in_channels=self.embedding_size,
                               out_channels=self.nbr_filters, kernel_size=self.filter_len, padding=self.border_mode)
        self.conv2 = nn.Conv1d(in_channels=self.nbr_filters,
                               out_channels=self.nbr_filters, kernel_size=self.filter_len, padding=self.border_mode)

        self.dropout1 = nn.Dropout(self.dropout1)
        self.dropout2 = nn.Dropout(self.dropout2)
        self.dropout3 = nn.Dropout(self.dropout3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=embed_dim, out_features=self.nodes)
        self.linear = nn.Linear(in_features=self.nbr_filters, out_features=self.nodes)
        self.output = nn.Linear(in_features=self.nodes, out_features=1)

    def forward(self, tokens, need_head_weights=True, return_contacts=False, return_representation=True):
        x = self.esm2(tokens, [layers], need_head_weights, return_contacts, return_representation)
        # x = self.esm2(tokens, [layers])


        x = x["representations"][layers][:, 0]
        x_o = x.unsqueeze(2)

        x = self.flatten(x_o)
        o_linear = self.fc(x)
        o_relu = self.relu(o_linear)
        o_dropout = self.dropout3(o_relu)
        o = self.output(o_dropout)
        return o


def eval_step(dataloader, model, threshold=0.5):
    model.eval()
    logits_list = []
    # y_pred_list, y_prob_list = [], []
    ids_list, strs_list = [], []
    my_bar = st.progress(0, text="Running UTR_LM")
    with torch.no_grad():
        for i, (ids, strs, _, toks, _, _) in enumerate(dataloader):
            ids_list.extend(ids)
            strs_list.extend(strs)
            # toks = toks.to(device)
            my_bar.progress((i + 1) / len(dataloader), text="Running UTR_LM")
            # print(toks)
            logits = model(toks, return_representation=True, return_contacts=True)

            logits = logits.reshape(-1)
            # y_prob = torch.sigmoid(logits)
            # y_pred = (y_prob > threshold).long()

            logits_list.extend(logits.tolist())
            # y_prob_list.extend(y_prob.tolist())
            # y_pred_list.extend(y_pred.tolist())

    st.success('Done', icon="✅")
    # data_pred = pd.DataFrame({'ID':ids_list, 'Sequence':strs_list, "MRL":logits_list, "prob":y_prob_list, "pred":y_pred_list})
    data_pred = pd.DataFrame({'ID': ids_list, 'Sequence': strs_list, "MRL": logits_list})
    return data_pred


def read_raw(raw_input):
    ids = []
    sequences = []

    file = StringIO(raw_input)
    for record in SeqIO.parse(file, "fasta"):

        # 检查序列是否只包含A, G, C, T
        sequence = str(record.seq.back_transcribe()).upper()[-inp_len:]
        if not set(sequence).issubset(set("AGCT")):
            st.write(
                f"Record '{record.description}' was skipped for containing invalid characters. Only A, G, C, T(U) are allowed.")
            continue

        # 将符合条件的序列添加到列表中
        ids.append(record.id)
        sequences.append(sequence)

    return ids, sequences


def generate_dataset_dataloader(ids, seqs):
    dataset = FastaBatchedDataset(ids, seqs, mask_prob=0.0)

    # dataset = FastaBatchedDataset(ids, seqs)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             collate_fn=alphabet.get_batch_converter(),
                                             batch_sampler=batches,
                                             shuffle=False)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batches, shuffle = False)
    # st.write(f"{len(dataset)} sequences")
    return dataset, dataloader


def predict_raw(raw_input, file_name):
    # st.write('====Parse Input====')
    ids, seqs = read_raw(raw_input)
    _, dataloader = generate_dataset_dataloader(ids, seqs)

    model = CNN_linear()
    # st.write(model.state_dict().keys())
    # st.write({k.replace('module.', ''):v for k,v in torch.load(modelfile, map_location=torch.device('cpu')).items()}.keys())
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(modelfile, map_location=torch.device('cuda')).items()},
        strict=True)
    # model.load_state_dict(torch.load(modelfile, map_location=torch.device('cpu')), strict = False)

    # st.write('====Predict====')
    pred = eval_step(dataloader, model)
    st.write(f'{file_name}')
    # print(pred)
    return pred


st.title("5' UTR prediction- Batch Processing")

st.subheader("Select fasta files")
uploaded_files = st.file_uploader("Sequence file in FASTA format",type=["fasta"], accept_multiple_files=True)


st.subheader("Download Prediction Results")
save_dir = st.text_input("Enter output directory path (absolute or relative):",
                         value="/gs/bs/tga-satolab-gtex/dai/myscript/dna_LDM/save/")

if "results" not in st.session_state:
    st.session_state.results = []
    st.session_state.save_names = []

if st.button("Predict All"):
    if uploaded_files:
        st.session_state.results = []
        st.session_state.save_names = []
        for file in uploaded_files:
            fasta_content = file.getvalue().decode('utf-8')
            result = predict_raw(fasta_content, file.name.replace('.fasta', ''))
            st.session_state.results.append(result)
            st.session_state.save_names.append(file.name.replace('.fasta','.csv'))

    if st.session_state.results:
        st.success(f"Processed {len(st.session_state.results)} files. Ready to save.")

if st.button("Download All"):
    if not st.session_state.results:
        st.warning("Please run prediction first")
    else:
        try:
            for result, save_name in zip(st.session_state.results, st.session_state.save_names):
                result.to_csv(os.path.join(save_dir,save_name))
                st.success(f"All files saved to `{save_dir}`")
        except Exception as e:
            st.error(f"Failed to save prediction results: {e}")


