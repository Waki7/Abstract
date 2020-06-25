import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(LSTM, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features

        lstm_in_features = in_features + hidden_features

        self.w_input = nn.Linear(in_features=lstm_in_features, out_features=hidden_features)
        self.w_forget = nn.Linear(in_features=lstm_in_features, out_features=hidden_features)
        self.w_output = nn.Linear(in_features=lstm_in_features, out_features=hidden_features)
        self.w_cstate = nn.Linear(in_features=lstm_in_features, out_features=hidden_features)

    def forward(self, x_t_1, h_t_1, context_t_1):
        x_cat_h_t = torch.cat((x_t_1, h_t_1), dim=-1)

        forget_t = torch.sigmoid(self.w_forget(x_cat_h_t))
        input_t = torch.sigmoid(self.w_input(x_cat_h_t))
        output_t = torch.sigmoid(self.w_output(x_cat_h_t))

        cell_state_t = torch.tanh(self.w_cstate(x_cat_h_t))
        print(forget_t.shape)
        print(context_t_1.shape)
        context_t = torch.sigmoid((forget_t * context_t_1) + (input_t * cell_state_t))
        print(context_t.shape)
        h_t = torch.tanh(context_t * output_t)
        print(h_t.shape)
        return h_t, context_t

    def update(self):
        return torch.sum(torch.dot(self.th_t1_c, self.th_t1), torch.dot(self.lr_, self.dL_t))
