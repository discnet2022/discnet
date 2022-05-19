import torch
from torch import nn
import numpy as np
from models_discnet_re.GRU_encoder import *
from torch.nn.utils.rnn import pad_sequence


class DiscNet_RE(nn.Module):
    def __init__(self, config, code_fq):
        super(DiscNet_RE, self).__init__()
        self.config = config
        self.code_fq = code_fq

        
        self.hidden_dim = self.config['hidden_dim']

        
        
        self.word_dense = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.sent_dense = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.node_dense = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.sent_span_dense = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.node_span_dense = nn.Linear(self.hidden_dim * 2, self.hidden_dim)


        self.code_weights = nn.Parameter(torch.empty(self.config['num_labels'], self.hidden_dim))
        torch.nn.init.xavier_uniform_(self.code_weights.data)

        self.code_dense = nn.Linear(self.hidden_dim, self.hidden_dim)
        torch.nn.init.xavier_uniform_(self.code_dense.weight.data)
        self.code_dense.bias.data.zero_()

        self.code_dense_final = nn.Linear(self.hidden_dim, self.hidden_dim)
        torch.nn.init.xavier_uniform_(self.code_dense_final.weight.data)
        self.code_dense_final.bias.data.zero_()

        self.code_gate_dense = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        torch.nn.init.xavier_uniform_(self.code_gate_dense.weight.data)
        self.code_gate_dense.bias.data.zero_()
        self.code_gate_output = nn.Linear(self.hidden_dim, self.hidden_dim)
        torch.nn.init.xavier_uniform_(self.code_gate_output.weight.data)
        self.code_gate_output.bias.data.zero_()

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.clues_dense = nn.Linear(self.hidden_dim, self.hidden_dim)
        torch.nn.init.xavier_uniform_(self.clues_dense.weight.data)
        self.clues_dense.bias.data.zero_()

        self.logits_bias = nn.Parameter(torch.empty(self.config['num_labels']))
        self.logits_bias.data.zero_()
        self.loss = nn.BCELoss(reduction='mean')
        
        
        self.word_LN = nn.LayerNorm(self.hidden_dim)

        self.sent_LN = nn.LayerNorm(self.hidden_dim)
        self.node_LN = nn.LayerNorm(self.hidden_dim)
        self.code_LN = nn.LayerNorm(self.hidden_dim)
        self.seg_dropout = nn.Dropout(0.6)
        
        self.apply(self.init_weights)
        self.initialize_embeddings()
        
        self.code_encoder = GRU_Encoder(self.config['rnn_config'])
        self.word_encoder = GRU_Encoder(self.config['rnn_config'])
        self.sent_encoder = GRU_Encoder(self.config['rnn_config'])
        self.node_encoder = GRU_Encoder(self.config['rnn_config'])
        
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear)):
            torch.nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def initialize_embeddings(self):
        word_embeddings_path = self.config["word_embeddings_path"]
        self.word_embeddings = nn.Embedding.from_pretrained(
            torch.FloatTensor(np.load(word_embeddings_path)), freeze=False)
        self.code_embeddings = nn.Embedding.from_pretrained(
            torch.FloatTensor(np.load(word_embeddings_path)), freeze=False)
        print(word_embeddings_path + "has been loaded!")
        self.heading_embeddings = nn.Embedding(101, 100)
        torch.nn.init.xavier_uniform_(self.heading_embeddings.weight.data)

    def hierarchy_forward(self, x, x_masks, code_text, encoder,
                          linear,
                          span_linear=None,
                          start_pos=None,
                          end_pos=None,
                          spans=None,
                          LN=None):
        """
        :param x: [batch, len, hidden]
        :param x_masks: [batch, len]
        :param code_text: [num_labels, hidden]
        :param encoder:
        :return:
        """
        x_len = x_masks.sum(-1)
        _x_2_direction, _, _x_h = encoder(x, x_len, return_sequence=True)
        # [batch, len, hidden]'
        _x = linear(_x_2_direction)
        _x_forward = _x_2_direction[:,:,:self.hidden_dim]
        _x_backward = _x_2_direction[:,:,self.hidden_dim:]
        alpha = torch.matmul(_x, code_text.T)
        # [batch, len, num_labels]
        alpha = alpha * x_masks.unsqueeze(-1)
        attention_mask = (1.0 - x_masks.unsqueeze(-1)) * -10000.0
        alpha = alpha + attention_mask
        alpha_clues = nn.Softmax(dim=1)(alpha)
        x_clues = torch.matmul(alpha_clues.transpose(-1, -2), _x)
        # [batch, num_labels, hidden]

        if spans is not None:
            x_max = []
            for i in range(len(spans)):
                sens = torch.split(x[i][:sum(spans[i])], spans[i], dim=-2)
                x_max.append(torch.stack([i.max(dim=0)[0] for i in sens]))

            x_max = pad_sequence(x_max, batch_first=True)
            _x_next_forward = pad_sequence([_x_forward[i, _ids, :] for i, _ids in enumerate(end_pos)], batch_first=True)
            _x_next_backward = pad_sequence([_x_backward[i, _ids, :] for i, _ids in enumerate(start_pos)], batch_first=True)

            _x_next = span_linear(torch.cat([_x_next_forward, _x_next_backward], dim=-1))


            next_masks = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor([1.] * len(i)) for i in spans],
                batch_first=True).to(_x_next.device).requires_grad_(False)

            beta = torch.matmul(_x_next, code_text.T)
            # [batch, len, num_labels]
            beta = beta * next_masks.unsqueeze(-1)
            attention_mask = (1.0 - next_masks.unsqueeze(-1)) * -10000.0
            beta = beta + attention_mask
            beta = nn.Softmax(dim=-1)(beta)
            _x_next = torch.matmul(beta, code_text)
            _x_next = self.relu(x_max + _x_next)


            return _x_next, x_clues, next_masks
        return x_clues

    def forward(self, text,
                segments,
                sen_start_pos, sen_end_pos,
                node_start_pos, node_end_pos,
                sen_spans, node_spans,
                code_desc_text, labels):
        """
        :param text: [batch, len]
        :param n_nodes: [num of nodes of each HADM_ID]
        :param n_sen: [num of sentences of each node]
        :param code_desc_text: [num_labels, len]
        :param labels: [batch, 8921]
        :return:
        """
        word_masks = (text > 0).float()
        code_len = (code_desc_text > 0).sum(-1)

        word_embed = self.word_embeddings(text)
        
        seg_embed = self.heading_embeddings(segments)
        seg_embed = seg_embed / torch.norm(seg_embed, dim=-1, keepdim=True)
        seg_embed = self.seg_dropout(seg_embed)
        word_embed = (seg_embed + word_embed) * word_masks.unsqueeze(-1)
        

        code_text = self.code_embeddings(code_desc_text)
        # [num_labels, len, hidden]
        code_text = self.code_encoder(code_text, code_len, return_sequence=False)
        beta = self.relu(self.code_gate_dense(torch.cat([code_text, self.code_weights], dim=-1)))
        beta = self.sigmoid(self.code_gate_output(beta))

        w = self.code_weights / torch.norm(self.code_weights, dim=-1, keepdim=True)
        _code_text = code_text * beta + w
        
        
        reg = torch.pow(beta, 2)
        reg = reg * self.code_fq.unsqueeze(-1)
        reg = - reg.mean()

        _code_text = self.code_LN(_code_text)
        code_text = self.tanh(self.code_dense(_code_text))

        sent_embed, word_clues, sent_masks = self.hierarchy_forward(x=word_embed,
                                                       x_masks=word_masks,
                                                        linear=self.word_dense,
                                                        span_linear=self.sent_span_dense,
                                                        start_pos=sen_start_pos,
                                                        end_pos=sen_end_pos,
                                                        spans=sen_spans,
                                                        code_text=code_text,
                                                       encoder=self.word_encoder,
                                                       LN=self.sent_LN,
                                                       )
        
        sent_clues = self.hierarchy_forward(x=sent_embed,
                                                x_masks=sent_masks,
                                                linear=self.sent_dense,
                                                span_linear=self.node_span_dense,
                                                start_pos=node_start_pos,
                                                end_pos=node_end_pos,
                                                spans=None,
                                                code_text=code_text,
                                                encoder=self.sent_encoder,
                                                LN=self.node_LN)
        
        
        clues = torch.cat([word_clues.unsqueeze(-2),
                           sent_clues.unsqueeze(-2),
                           ],
                          dim=-2)
        clues = torch.max(clues, dim=-2)[0]

        clues = self.relu(self.clues_dense(clues))
        code_text = self.tanh(self.code_dense_final(_code_text))
        logits = code_text.mul(clues).sum(dim=2).add(self.logits_bias)
        probs = self.sigmoid(logits)
        labels = labels.float()
        loss = self.loss(probs, labels)
        return probs, loss, reg





