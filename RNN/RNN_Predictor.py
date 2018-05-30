import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from name_dataset import NameDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNTimeRegressor(nn.Module):

    def __init__(self, input_size=2, hidden_size=100, output_size=1, n_layers=1, bidirectional=True):
        super(RNNTimeRegressor, self).__init__()
        
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.n_layers     = n_layers
        self.n_directions = int(bidirectional) + 1

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU (hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        x = x.t()
        batch_size = x.size(1)

        hidden = self._init_hidden(batch_size)
        embedded = self.embedding(x)
        gru_input = pack_padded_sequence(embedded, seq_lengths.data.cpu().numpy())

        self.gru.flatten_parameters()
        gru_out, hidden = self.gru(gru_input, hidden)

        out = self.fc(hidden[-1])
        return out

    def init_hidden_layers(self, batch_size):
        # pylint: disable=E1101
        hidden = torch.zeros(self.n_layers * self.n_directions,
                batch_size, self.hidden_size)
        # pylint: enable=E1101

        if self.gpu_en:
            return Variable(hidden.cuda())
        else:
            return Variable(hidden)