import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1, att_len=196, att_size=1024):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.att_size = att_size
        self.dropout_p = dropout_p
        self.att_len = att_len

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru1 = nn.GRUCell(self.input_size, self.hidden_size)

        self.attn = nn.Linear(self.hidden_size, self.att_len)
        self.attn_combine = nn.Linear(self.hidden_size + self.att_size, self.hidden_size)

        self.gru2 = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, features):

        input = self.dropout(input)

        hidden = self.gru1(input, hidden)


        attn_weights = F.softmax(
            self.attn(hidden), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 features)

        att_input = torch.cat((hidden, attn_applied.squeeze(1)), 1)

        att_input = self.attn_combine(att_input)

        att_input = F.relu(att_input)


        hidden = self.gru2(att_input, hidden)
        output = self.out(hidden)

        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device=device)

    def initInput(self, batch_size):
        return torch.zeros(batch_size, self.input_size, dtype=torch.float32, device=device)