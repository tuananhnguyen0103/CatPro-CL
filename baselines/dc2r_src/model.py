#!/usr/bin/env python36
# -*- coding: utf-8 -*-


import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module
from recbole.model.layers import TransformerEncoder


class TransNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.n_layers = opt.n_layers
        self.n_heads = opt.n_heads
        self.hidden_size = opt.hiddenSize
        self.inner_size = opt.inner_size
        self.hidden_dropout_prob = opt.hidden_dropout_prob
        self.attn_dropout_prob = opt.attn_dropout_prob
        self.hidden_act = opt.hidden_act
        self.layer_norm_eps = opt.layer_norm_eps
        self.initializer_range = opt.initializer_range

        self.position_embedding = nn.Embedding(60, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.fn = nn.Linear(self.hidden_size, 1)

        self.apply(self._init_weights)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        alpha = self.fn(output).to(torch.double)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        return alpha

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class DC2R(Module):
    def __init__(self, opt, n_node):
        super(DC2R, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.item_embedding = nn.Embedding(self.n_node, self.hidden_size)  # num_embeddings, embedding_dim

        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.mse = nn.MSELoss()

        self.n_attr1 = opt.n_attr1
        self.n_taxo = opt.n_taxo
        self.sentinel = opt.sentinel
        self.dropout2 = nn.Dropout(0.2)
        self.attr_emb1 = nn.Embedding(self.n_attr1, self.hidden_size)
        self.taxo_emb = nn.Embedding(self.n_taxo, self.hidden_size)
        self.linear_attr = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.linear_combine_taxo = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=False)
        self.mse = nn.MSELoss()
        self.temperature = opt.temperature
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, seq_output):
        all_item_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, all_item_emb.transpose(0, 1))

        return scores

    def forward(self, alias_inputs, A, mask, item_seq, attr1, attr2, taxo1, taxo2, taxo3):
        h = self.item_embedding(item_seq)
        taxo_emb1 = self.taxo_emb(taxo1)
        taxo_emb2 = self.taxo_emb(taxo2)
        taxo_emb3 = self.taxo_emb(taxo3)
        h = self.linear_combine_taxo(torch.cat((h, taxo_emb1, taxo_emb2, taxo_emb3), dim=-1))

        alpha = self.ave_net(item_seq)
        E_p = torch.sum(alpha * h, dim=1)

        attr_emb1 = self.attr_emb1(attr1)  # [bs, seq_len, dim]
        attr_emb2 = self.attr_emb1(attr2)
        attr_emb = self.linear_attr(torch.cat((attr_emb1, attr_emb2), dim=-1))
        attr_emb = self.linear_zero(attr_emb)

        alpha = self.ave_net(attr1)
        I_p = torch.sum(alpha * attr_emb, dim=1)

        return h, E_p, attr_emb, I_p

    def ave_net(self, item_seq):
        mask = item_seq.gt(0)
        alpha = mask.to(torch.float) / mask.sum(dim=-1, keepdim=True)
        alpha = torch.nan_to_num(alpha, nan=0.000001)
        return alpha.unsqueeze(-1)

    def zeroshot(self, seq_hidden, attr_emb, mask, alias_inputs):
        oriitem = torch.sum(seq_hidden * mask.view(mask.shape[0], -1, 1).float(), 1)

        get = lambda i: attr_emb[i][alias_inputs[i]]
        seq_attr = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        intent = torch.sum(seq_attr * mask.view(mask.shape[0], -1, 1).float(), 1)

        loss = self.mse(self.linear_zero(intent), oriitem)
        return loss

    def compute_cand_scores(self, attr_pre, cand_attr1, cand_attr2):
        attr_emb1 = self.attr_emb1(cand_attr1)
        attr_emb2 = self.attr_emb1(cand_attr2)
        cand_attr_emb = self.linear_attr(torch.cat((attr_emb1, attr_emb2), dim=-1))  # [n_item, dim]
        cand_attr_emb = self.linear_zero(cand_attr_emb)
        scores = torch.matmul(attr_pre, cand_attr_emb.transpose(0, 1)) / self.temperature
        return scores

    def get_session_embedding(self, h, mask):
        ht = h[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(h)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        try:
            session_emb = torch.sum(alpha * h * mask.view(mask.shape[0], -1, 1).float(), 1)  # sg
        except:
            mask = mask[:, h.shape[1]]
            session_emb = torch.sum(alpha * h * mask.view(mask.shape[0], -1, 1).float(), 1)
        session_emb = self.linear_transform(torch.cat([session_emb, ht], 1))
        return session_emb



class DC2Rtrm(DC2R):
    def __init__(self, opt, n_node):
        super(DC2Rtrm, self).__init__(opt, n_node)
        self.net = TransNet(opt)
        self.reset_parameters()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


    def forward(self, alias_inputs, A, mask, item_seq, attr1, attr2, taxo1, taxo2, taxo3):
        h = self.item_embedding(item_seq)
        taxo_emb1 = self.taxo_emb(taxo1)
        taxo_emb2 = self.taxo_emb(taxo2)
        taxo_emb3 = self.taxo_emb(taxo3)
        h = self.linear_combine_taxo(torch.cat((h, taxo_emb1, taxo_emb2, taxo_emb3), dim=-1))

        alpha = self.net(item_seq, h)
        E_p = torch.sum(alpha * h, dim=1)

        attr_emb1 = self.attr_emb1(attr1)  # [bs, seq_len, dim]
        attr_emb2 = self.attr_emb1(attr2)
        attr_emb = self.linear_attr(torch.cat((attr_emb1, attr_emb2), dim=-1))
        attr_emb = self.linear_zero(attr_emb)

        alpha = self.net(attr1, attr_emb)
        I_p = torch.sum(alpha * attr_emb, dim=1)

        return h, E_p, attr_emb, I_p


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data, attr_data, taxo_data, sentinel):
    alias_inputs, items, mask, targets, \
    attr1, attr2, taxo1, taxo2, taxo3, ca_a1, ca_a2, A = data.get_slice(i, attr_data,taxo_data)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    A = trans_to_cuda(torch.Tensor(A).float())
    attr1 = trans_to_cuda(torch.LongTensor(attr1))
    attr2 = trans_to_cuda(torch.LongTensor(attr2))
    taxo1 = trans_to_cuda(torch.Tensor(taxo1).long())
    taxo2 = trans_to_cuda(torch.Tensor(taxo2).long())
    taxo3 = trans_to_cuda(torch.Tensor(taxo3).long())
    ca_a1 = trans_to_cuda(torch.LongTensor(ca_a1))
    ca_a2 = trans_to_cuda(torch.LongTensor(ca_a2))

    hidden, E_p, attr_emb, I_p = model(alias_inputs, A, mask, items, attr1, attr2, taxo1, taxo2, taxo3)

    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    zeroloss = model.zeroshot(seq_hidden, attr_emb, mask, alias_inputs)

    score1 = model.compute_scores(E_p)
    if sentinel:
        score2 = model.compute_cand_scores(I_p, ca_a1[1:], ca_a2[1:])
    else:
        score2 = model.compute_cand_scores(I_p, ca_a1, ca_a2)

    score = score1 + score2
    return targets, score, zeroloss


def train_test(model, train_data, test_data, attr_data, taxo_data, opt):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size) 
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad() 
        targets, scores, zeroloss = forward(model, i, train_data, attr_data, taxo_data, opt.sentinel)
        targets = trans_to_cuda(torch.Tensor(targets).long()) 
        loss = model.loss_function(scores, targets - 1) + opt.gama * zeroloss
        loss.backward()

        model.optimizer.step()

        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr, hit10, mrr10 = [], [], [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores, _ = forward(model, i, test_data, attr_data, taxo_data, opt.sentinel)#model, i, train_data
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
        sub_scores10 = scores.topk(10)[1]
        sub_scores10 = trans_to_cpu(sub_scores10).detach().numpy()
        for score, target, mask in zip(sub_scores10, targets, test_data.mask):
            hit10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr10.append(0)
            else:
                mrr10.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    hit10 = np.mean(hit10) * 100
    mrr10 = np.mean(mrr10) * 100

    return hit, mrr,hit10,mrr10


