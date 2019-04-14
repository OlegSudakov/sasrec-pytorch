''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import sasrec.Constants as Constants
from sasrec.Layers import SelfAttentionBlock, MatrixFactorizationLayer


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class SASRec(nn.Module):

    def __init__(self, n_src_items, len_max_seq, d=40, n_layers=2, n_head=1, dropout=0.2):
        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_items, d, padding_idx=Constants.PAD)
        self.src_word_emb_weight = self.src_word_emb.weight
        self.emb_dropout = nn.Dropout(dropout)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d, padding_idx=0))

        self.layer_stack = nn.ModuleList([
            SelfAttentionBlock(d, d, n_head, d, d, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.mfl = MatrixFactorizationLayer(self.src_word_emb_weight)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_pos):
        # -- Prepare masks
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        slf_attn_mask_subseq = get_subsequent_mask(src_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.emb_dropout(self.src_word_emb(src_seq) + self.position_enc(src_pos))
        item_emb = enc_output

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output += item_emb

        enc_output = self.mfl(enc_output)
        enc_output = self.softmax(enc_output)

        return enc_output,
