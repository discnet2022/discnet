import torch
from torch import nn

class GRU_Encoder(nn.Module):
    def __init__(self, config,):
        super(GRU_Encoder, self).__init__()
        self.config = config
        self.hidden_dim = self.config["hidden_dim"]
        self.num_layers = self.config["num_layers"]
        self.lstm = nn.GRU(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.dense = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        # self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.config["dropout_rate"])
        for key, param in self.named_parameters():
            if 'weight_ih' in key:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in key:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in key:
                param.data.fill_(0.)


    def get_final_hidden_state(self, h_n):
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # 如果是stackinglstm则num_layers > 1
        # 2 是方向的数量 bidirectional = 2 directions
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_dim)
        h_n = h_n[-1, :, :, :]
        h_n = h_n.transpose(dim0=0, dim1=1)
        # h_n [batch_size, 2, hidden_dim]
        h_n = h_n.reshape(-1, 2 * self.hidden_dim)
        return h_n

    def forward(self, embed, seq_len, return_sequence=False):
        # embed: 查好嵌入的序列[batch_size, seq_len, hidden_dim]
        embed = self.dropout(embed)
        embed = \
            torch.nn.utils.rnn.pack_padded_sequence(embed, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        outputs, h_n = self.lstm(embed)
        h_n = self.get_final_hidden_state(h_n)
        h_n = self.dense(h_n)
        if return_sequence:
            outputs, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            # outputs = self.dense(outputs)
            # (outputs[0][-1][:384] == h_n[0][:384]).sum() forward pass
            # (outputs[0][0][384:] == h_n[0][384:]).sum() backward pass
            return outputs, seq_lens, h_n
        return h_n

if __name__ == '__main__':
    a = torch.randn(16, 10, 384)
    config = {
        "dropout_rate": 0.,
        "output_dim": 384,
        "hidden_dim": 384,
        "num_layers": 1,
    }
    model = GRU_Encoder(config)
    seq_len = torch.ones(16).long() * 10
    a, _, a_h = model(a, seq_len, return_sequence=True)
    print(a.shape)
    print(a_h.shape)