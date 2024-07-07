import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, seq_len,num_classes=1):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.seq_len = seq_len

        self.bnn1 = nn.Linear(input_size, 32)
        self.bnn2 = nn.Linear(32,64)
        self.bnn3 = nn.Linear(64,64)
        self.bnn4 = nn.Linear(64,64)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.lstm1 = nn.LSTM(64, hidden_size1, num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.bn_lstm1 = nn.BatchNorm1d(2*hidden_size1,device=self.device)  
        # self.lstm2 = nn.LSTM(2*hidden_size1, hidden_size2, num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        # self.bn_lstm2 = nn.BatchNorm1d(2*hidden_size2,device=device)
        self.nn1 = nn.Linear(2*hidden_size1, 2*hidden_size1)
        self.nn2 = nn.Linear(2*hidden_size1, 512)
        self.nn3 = nn.Linear(512, 512)
        self.nn4 = nn.Linear(512, 256)
        self.nn5 = nn.Linear(256, 256)
        self.nn6 = nn.Linear(256, 128)
        self.nn7 = nn.Linear(128, 32)
        self.nn8 = nn.Linear(32, 1)

    
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.batch = nn.BatchNorm1d()
        self.drop = nn.Dropout(p=0.5)


        
    def forward(self, x, array_lengths):
        # Set initial hidden states (and cell states for LSTM)
        # print(x.size(0))
        inital_seq_len = x.size(1)
        x = Variable(x.float()).to(self.device)

        x = torch.reshape(x, (x.size(0)*x.size(1), x.size(2)))

        ## before nn
        out = self.bnn1(x)
        out = self.relu(out)
        out = self.bnn2(out)
        out = self.relu(out)
        out = self.bnn3(out)
        out = self.relu(out)
        out = self.bnn4(out)
        out = self.relu(out)

        ## reshaping again
        out = torch.reshape(out, (-1, inital_seq_len, out.size(1)))
        # print(out.size())
        # print(aaaaa)

        # out = torch.permute(out, (0,2,1))
        
        pack = nn.utils.rnn.pack_padded_sequence(out, array_lengths, batch_first=True, enforce_sorted=False)
        h0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size1).to(self.device))
        c0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size1).to(self.device))
        h1 = Variable(torch.zeros(2*self.num_layers, self.hidden_size1, self.hidden_size2).to(self.device))
        c1 = Variable(torch.zeros(2*self.num_layers, self.hidden_size1, self.hidden_size2).to(self.device))
        
        # Forward propagate RNN
        out, _ = self.lstm1(pack, (h0,c0))
        del(h0)
        del(c0)
        # out, _ = self.lstm2(out, (h1,c1))
        gc.collect()
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        this_batch_len = unpacked.size(1)
        out = unpacked
        # print('before', out.size())
        out = torch.reshape(out, (out.size(0)*out.size(1), out.size(2)))

        ##nn
        out = self.nn1(out)
        out = self.relu(out)
        out = self.nn2(out)
        out = self.relu(out)
        out = self.nn3(out)
        out = self.relu(out)
        out = self.nn4(out)
        out = self.relu(out)
        out = self.nn5(out)
        out = self.relu(out)
        out = self.nn6(out)
        out = self.relu(out)
        out = self.nn7(out)
        out = self.relu(out)
        out = self.nn8(out)
        
        ## reshaping
        out = torch.reshape(out, (-1, this_batch_len, 1))
        # print(out.size()) 
        # print(aaaaa)   
        

        return out