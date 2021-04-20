import torch.nn as nn

######################################################
######################################################
##################      FNN      #####################
######################################################
######################################################


class FNNEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate, nonlinear=True):

        super(FNNEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.nonlinear = nonlinear

        print('DAN: input {}, hidden {}, output {}'.format(self.input_size, self.hidden_size, self.output_size))

        # first hidden layers
        if self.nonlinear:
            self.hidden = nn.ModuleList([nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
                                         nn.ReLU(),
                                         nn.Dropout(self.dropout_rate)])
        else:
            self.hidden = nn.ModuleList([nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
                                         nn.Dropout(self.dropout_rate)])

        # optional deep layers
        for k in range(1, self.num_layers):
            if self.nonlinear:
                self.hidden.extend([nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout_rate)])
            else:
                self.hidden.extend([nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                                    nn.Dropout(self.dropout_rate)])

        # output linear function (readout)
        self.final = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, x):

        y = x
        for i in range(len(self.hidden)):
            y = self.hidden[i](y)

        out = self.final(y)

        return out


######################################################
######################################################
##################      LSTM      ####################
######################################################
######################################################


class LSTMEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidir, dropout_rate):

        # input: (batch, seq_len, input_size)

        super(LSTMEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.num_layers = num_layers
        self.bidir = bidir

        # Define the LSTM layer
        self.lstm = nn.LSTM(batch_first=True, input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidir, dropout=dropout_rate)

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, x):

        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, hidden_size]
        # shape of self.hidden: (a, b), where a and b both have shape (num_layers, batch_size, hidden_dim).
        lstm_out, hidden = self.lstm(x)

        return lstm_out

    def linear_layer(self, x):

        out = self.dropout_layer(x)
        out = self.linear(out)

        return out
