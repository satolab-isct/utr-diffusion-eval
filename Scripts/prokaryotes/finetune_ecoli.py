# CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 5001 MJ3_Finetune_extract_append_predictor_Sample_10fold-lr-huber-DDP.py --device_ids 0,1,2,3 --label_type rl --epochs 1 --huber_loss --train_file 4.1_train_data_GSM3130435_egfp_unmod_1.csv --prefix ESM2SISS_FS4.1.ep93.1e-2.dr5 --lr 1e-2 --dropout3 0.5 --modelfile /scratch/users/yanyichu/UTR-LM/Model/ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl --finetune --bos_emb --test1fold
# {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}
import os
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", ""))
import argparse

import torch
import torch.nn as nn
from accelerate import Accelerator

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split

from esm.data import *
from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS
from esm.model.esm2_supervised import ESM2
from esm.model.esm2_only_secondarystructure import ESM2 as ESM2_SS
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

import scipy.stats as stats
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', type=str, default='0,1,2', help="Training Devices")
parser.add_argument('--local_rank', type=int, default=-1, help="DDP parameter, do not modify")
parser.add_argument('--log_interval', type=int, default=50, help="Log Interval")
parser.add_argument('--seed', type=int, default=1337)

parser.add_argument('--prefix', type=str, default='ESM2SISS_FS4.1.ep93.1e-2.dr5')
parser.add_argument('--label_type', type=str, default='rl')
parser.add_argument('--seq_type', type=str, default='utr')

parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--cnn_layers', type=int, default=0)
parser.add_argument('--nodes', type=int, default=40)
parser.add_argument('--dropout3', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--patience', type=int, default=0)
parser.add_argument('--test1fold', action='store_true')
parser.add_argument('--huber_loss', action='store_true')

parser.add_argument('--train_file', type=str, default='4.1_train_data_GSM3130435_egfp_unmod_1.csv')

parser.add_argument('--load_wholemodel', action='store_true')  ## if --finetune: False
parser.add_argument('--init_epochs', type=int, default=0)
parser.add_argument('--modelfile', type=str,
                    default='/scratch/users/yanyichu/UTR-LM/saved_models/ESM2_1.4_five_species_TrainLossMin_6layers_16heads_128embedsize_4096batchToks.pkl')

parser.add_argument('--finetune_modelfile', type=str,
                    default='/scratch/users/yanyichu/UTR-LM/saved_models/CVESM2lr1e-5_DDP9.1_unmod_1_10folds_rl_LabelScalerTrue_LabelLog2False_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_nodes40_dropout30.2_finetuneTrue_huberlossTrue_magicFalse_fold0_epoch19_lr0.1.pt')

parser.add_argument('--finetune', action='store_true')  ## if --finetune: False
parser.add_argument('--scaler', action='store_true')  ## if --finetune: False
parser.add_argument('--log2', action='store_true')  ## if --finetune: False
parser.add_argument('--avg_emb', action='store_true')  ## if --finetune: False
parser.add_argument('--bos_emb', action='store_true')  ## if --finetune: False
parser.add_argument('--train_atg', action='store_true')  ## if --finetune: False
parser.add_argument('--train_n_atg', action='store_true')  ## if --finetune: False
parser.add_argument('--magic', action='store_true')  ## if --finetune: False

args = parser.parse_args()
print(args)

global layers, heads, embed_dim, batch_toks, inp_len, device_ids, device, train_obj_col, epoch
model_info = args.modelfile.split('/')[-1].split('_')
inp_len = 96
for item in model_info:
    if 'layers' in item:
        layers = int(item[0])
    elif 'heads' in item:
        heads = int(item[:-5])
    elif 'embedsize' in item:
        embed_dim = int(item[:-9])
    elif 'batchToks' in item:
        print(item)
        batch_toks = (inp_len + 2) * 4096


class CNN_linear(nn.Module):
    def __init__(self,
                 border_mode='same', filter_len=8, nbr_filters=120,
                 dropout1=0, dropout2=0):

        super(CNN_linear, self).__init__()

        self.embedding_size = embed_dim
        self.border_mode = border_mode
        self.inp_len = inp_len
        self.nodes = args.nodes
        self.cnn_layers = args.cnn_layers
        self.filter_len = filter_len
        self.nbr_filters = nbr_filters
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = args.dropout3

        if 'SISS' in args.modelfile:
            self.esm2 = ESM2_SISS(num_layers=layers,
                                  embed_dim=embed_dim,
                                  attention_heads=heads,
                                  alphabet=alphabet)
        elif 'SS' in args.modelfile:
            self.esm2 = ESM2_SS(num_layers=layers,
                                embed_dim=embed_dim,
                                attention_heads=heads,
                                alphabet=alphabet)
        else:
            self.esm2 = ESM2(num_layers=layers,
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
        if args.avg_emb or args.bos_emb:
            self.fc = nn.Linear(in_features=embed_dim, out_features=self.nodes)
        else:
            self.fc = nn.Linear(in_features=inp_len * embed_dim, out_features=self.nodes)
        if args.avg_emb or args.bos_emb:
            self.linear = nn.Linear(in_features=self.nbr_filters, out_features=self.nodes)
        else:
            self.linear = nn.Linear(in_features=inp_len * self.nbr_filters, out_features=self.nodes)
        self.output = nn.Linear(in_features=self.nodes, out_features=1)
        if self.cnn_layers == -1: self.direct_output = nn.Linear(in_features=embed_dim, out_features=1)
        if args.magic: self.magic_output = nn.Linear(in_features=1, out_features=1)

    def forward(self, tokens, need_head_weights=True, return_contacts=True, return_representation=True):

        x = self.esm2(tokens, [layers], need_head_weights, return_contacts, return_representation)
        if args.avg_emb:
            x = x["representations"][layers][:, 1: inp_len + 1].mean(1)
            x_o = x.unsqueeze(2)
        elif args.bos_emb:
            x = x["representations"][layers][:, 0]
            x_o = x.unsqueeze(2)
        else:
            x_o = x["representations"][layers][:, 1: inp_len + 1]
            x_o = x_o.permute(0, 2, 1)

        if self.cnn_layers >= 1:
            x_cnn1 = self.conv1(x_o)
            x_o = self.relu(x_cnn1)
        if self.cnn_layers >= 2:
            x_cnn2 = self.conv2(x_o)
            x_relu2 = self.relu(x_cnn2)
            x_o = self.dropout1(x_relu2)
        if self.cnn_layers >= 3:
            x_cnn3 = self.conv2(x_o)
            x_relu3 = self.relu(x_cnn3)
            x_o = self.dropout2(x_relu3)

        #         if self.cnn_layers >= 1:
        x = self.flatten(x_o)
        if self.cnn_layers != -1:
            if self.cnn_layers != 0:
                o_linear = self.linear(x)
            else:
                o_linear = self.fc(x)
            o_relu = self.relu(o_linear)
            o_dropout = self.dropout3(o_relu)
            o = self.output(o_dropout)
        else:
            o = self.direct_output(x)
        #         print(o.shape)
        if args.magic:
            o = self.magic_output(o)
        return o


def train_step_batch(batch, model, criterion, device):
    """
    batch: (labels, strs, masked_strs, toks, masked_toks, extra)
    return: loss (tensor)
    """
    labels, strs, masked_strs, toks, masked_toks, _ = batch
    toks = toks.to(device)
    labels = torch.as_tensor(labels, dtype=torch.float32, device=device).view(-1, 1)

    outputs = model(toks, return_representation=True, return_contacts=True)
    loss = criterion(outputs, labels)
    return loss


@torch.no_grad()
def eval_step_batch(batch, model, criterion, device):
    labels, strs, masked_strs, toks, masked_toks, _ = batch
    toks = toks.to(device)
    labels = torch.as_tensor(labels, dtype=torch.float32, device=device).view(-1, 1)

    outputs = model(toks, return_representation=True, return_contacts=True)
    loss = criterion(outputs, labels)
    return loss

def test_step_batch(batch, model, criterion, device):
    labels, strs, masked_strs, toks, masked_toks, _ = batch
    toks = toks.to(device)
    labels = torch.as_tensor(labels, dtype=torch.float32, device=device).view(-1, 1)

    outputs = model(toks, return_representation=True, return_contacts=True)
    loss = criterion(outputs, labels)
    return loss, outputs, labels

def plot_test(pred_all, true_all, label, epoch):
    pearson_r = pearsonr(true_all, pred_all)[0]
    spearman_r = spearmanr(true_all, pred_all)[0]
    n = len(true_all)

    plt.figure(figsize=(8, 6))
    plt.scatter(true_all, pred_all, s=6, label=f"Pearson={pearson_r:.3f}, Spearman={spearman_r:.3f}")
    plt.xlabel(f"True {label}", fontsize=16)
    plt.ylabel(f"Predicted {label}", fontsize=16)
    plt.tick_params(labelsize=14)
    plt.title(f"Test set (n={n}) on {label} at best epoch {epoch}", fontsize=16)
    plt.legend(loc="best", frameon=True, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"test_{label}.jpg", dpi=300)
    plt.close()

    return pearson_r, spearman_r


def r2(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2


def performances(label, pred):
    label, pred = list(label), list(pred)

    r = r2(label, pred)
    R2 = r2_score(label, pred)

    rmse = np.sqrt(mean_squared_error(label, pred))
    mae = mean_absolute_error(label, pred)

    try:
        pearson_r = pearsonr(label, pred)[0]
    except:
        pearson_r = -1e-9
    try:
        sp_cor = spearmanr(label, pred)[0]
    except:
        sp_cor = -1e-9

    print(
        f'r-squared = {r:.4f} | pearson r = {pearson_r:.4f} | spearman R = {sp_cor:.4f} | R-squared = {R2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}')

    return [r, pearson_r, sp_cor, R2, rmse, mae]


def generate_dataset_dataloader(e_data, seq_col, label_col):
    seqs   = e_data[seq_col].astype(str).tolist()
    labels = e_data[label_col].tolist()

    dataset = FastaBatchedDataset(labels, seqs, mask_prob=0.0)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    dataloader = DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches, shuffle=False)
    print(f"{len(dataset)} sequences")
    return dataset, dataloader

def wandb_log(epoch=0, iter=0, loss=None, mode='train', pearson_r=None, spearman_r=None):
    if mode == 'init':
        accelerator.log({
            'input length': inp_len,
            'layers': layers,
            'heads': heads,
            'embedding dim': embed_dim,
            'batch tokens': batch_toks
        })
    if mode == 'train':
        accelerator.log({"epoch": epoch, "training loss": loss,}, step=iter)
    if mode == 'valid':
        accelerator.log({"epoch": epoch, "validation loss": loss}, step=iter)
    if mode == 'test':
        accelerator.log({
            'test loss': loss,
            'pearson_r': pearson_r,
            'spearman_r': spearman_r
        })

#######
alphabet = Alphabet(mask_prob=0.0, standard_toks='AGCT')
print(alphabet.tok_to_idx)
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
EPOCHS = 50  # debug 用 5 个 epoch 就够
eval_interval = 2
LR = 5e-4
torch.manual_seed(SEED)
np.random.seed(SEED)

############################################
# DEBUG VERSION (Single Split, No DDP)
############################################

if __name__ == "__main__":
    ############################################
    # 1️⃣ Load Data
    ############################################
    train_file = "../data/Supplementary_Data_15_Integrated_Phenotypic_Measurements.csv"
    data = pd.read_csv(train_file)
    seq_col = 'gs.sequence'
    #label_col = "polysome.mean.load"   # 修改成你的 label
    #abel_col = "clean.lin.prot.mean"
    label_col = "halflife.rna.dna.mean"

    data = data[[seq_col, label_col]].copy()
    data[seq_col] = data[seq_col].astype(str).str.upper()
    print(f'Data number Before drop NaN: {len(data)}')
    data = data.dropna(subset=[label_col]).reset_index(drop=True)
    print(f'Data number After drop NaN: {len(data)}')
    seq_lens = data[seq_col].astype(str).str.len()
    print(f"Data sequence length\n Average: {seq_lens.mean():.2f}\n max: {seq_lens.max()}\n min: {seq_lens.min()}")

    def normalize_y(df, y_mean, y_std, eps = 1e-12):
        df = df.copy()
        df[label_col] = (df[label_col] - y_mean) / (y_std + eps)
        return df

    def reverse_y(y_norm, y_mean, y_std, eps = 1e-12):
        # y_norm: numpy array or torch tensor
        return y_norm * (y_std + eps) + y_mean

    ############################################
    # 2️⃣ Data splitting
    ############################################
    e_train, e_tmp = train_test_split(data, train_size=TRAIN_RATIO, random_state=SEED, shuffle=True)
    test_size_in_tmp = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    e_val, e_test = train_test_split(e_tmp, test_size=test_size_in_tmp, random_state=SEED, shuffle=True)
    print(f"Train: { e_train.shape[0]}\n, Valid: {e_val.shape[0]}\n, Test : { e_test.shape[0]}")

    # normalization
    y_mean, y_std = e_train[label_col].mean(), e_train[label_col].std()
    print(f"[Normalization] {label_col}: train mean={y_mean:.6f}, train std={y_std:.6f}")
    e_train = normalize_y(e_train, y_mean, y_std)
    e_val   = normalize_y(e_val, y_mean, y_std)
    e_test  = normalize_y(e_test, y_mean, y_std)

    # Data Loader
    train_dataset, train_loader = generate_dataset_dataloader(e_train, seq_col, label_col)
    val_dataset, val_loader = generate_dataset_dataloader(e_val, seq_col, label_col)
    test_dataset, test_loader = generate_dataset_dataloader(e_test, seq_col, label_col)


    ############################################
    # 3️⃣ Load Model
    ############################################
    accelerator =Accelerator(log_with=['wandb'], mixed_precision='fp16')
    accelerator.init_trackers(project_name="UTRLM_ecoli", init_kwargs={"wandb": {"notes": label_col}})

    model = CNN_linear().to(accelerator.device)
    state_dict = torch.load("../Model/Pretrained/ESM2_1.4_five_species_TrainLossMin_6layers_16heads_128embedsize_4096batchToks.pkl", map_location=accelerator.device)
    model.esm2.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()},strict=False)

    ############################################
    # 4 Froze params in ESM2 and deploy optimizer
    ############################################

    for name, param in model.named_parameters():
        if "esm2" in name:
            param.requires_grad = False

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    ############################################
    # 5️⃣ Training Loop（极简）
    ############################################

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader)
    wandb_log(mode='init')

    best_val, best_epoch, iter = np.inf, 0, 0
    best_ckpt = None
    for epoch in tqdm(range(EPOCHS), total=EPOCHS):
        model.train()
        for batch in train_loader:
            with accelerator.autocast():
                loss = train_step_batch(batch, model, criterion, accelerator.device)
            iter += 1
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            wandb_log(epoch, iter, loss.item(), mode='train')

        # Validation
        if (epoch + 1) % eval_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                with accelerator.autocast():
                    for batch in val_loader:
                        loss = eval_step_batch(batch, model, criterion, accelerator.device)
                        val_loss += loss.item()
            val_loss /= len(val_loader)
            wandb_log(epoch, iter, val_loss, mode='valid')
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_ckpt = deepcopy(accelerator.unwrap_model(model).state_dict())

                
    ############################################
    # Final Test
    ############################################

    print("==== Final Test ====")
    accelerator.unwrap_model(model).load_state_dict(best_ckpt, strict=False)
    model.eval()
    test_loss = 0.0
    pred_labels, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            loss, pred_label, true_label = test_step_batch(batch, model, criterion, accelerator.device)
            test_loss += loss.item()
            pred_labels.extend(pred_label.detach().cpu().numpy().ravel().tolist())
            true_labels.extend(true_label.detach().cpu().numpy().ravel().tolist())

    true_labels = reverse_y(np.array(true_labels), y_mean, y_std)
    pred_labels = reverse_y(np.array(pred_labels), y_mean, y_std)
    pearson_r, spearman_r = plot_test(true_labels, pred_labels, label_col.replace('.', ' '), epoch=best_epoch)
    wandb_log(epoch, iter, test_loss / len(test_loader), mode='test', pearson_r=pearson_r, spearman_r=spearman_r)
    accelerator.save(best_ckpt, f'../Model/polysome_best_ckpt_{best_epoch}.ckpt')
