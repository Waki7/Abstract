import torch
import torch.nn as nn

import settings


class LSTM(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features

        lstm_in_features = in_features + hidden_features

        self.w_input = nn.Linear(in_features=lstm_in_features, out_features=hidden_features)
        self.w_forget = nn.Linear(in_features=lstm_in_features, out_features=hidden_features)
        self.w_output = nn.Linear(in_features=lstm_in_features, out_features=hidden_features)
        self.w_cstate = nn.Linear(in_features=lstm_in_features, out_features=hidden_features)

    def forward(self, x, hidden, context):
        prev_x = x
        prev_h = hidden
        prev_context = context

        cell_input = torch.cat([prev_x, prev_h], dim=-1)

        forget_t = torch.sigmoid(self.w_forget(cell_input))
        input_t = torch.sigmoid(self.w_input(cell_input))
        output_t = torch.sigmoid(self.w_output(cell_input))

        cell_state_t = torch.tanh(self.w_cstate(cell_input))
        context_t = (forget_t * prev_context) + (input_t * cell_state_t)
        new_hidden = output_t * torch.tanh(context_t)
        return new_hidden, context_t

    def get_zero_input(self, batch_size):
        hidden_state = torch.zeros((batch_size, self.hidden_features)).to(**settings.ARGS)
        context = torch.zeros((batch_size, self.hidden_features)).to(**settings.ARGS)
        return hidden_state, context
