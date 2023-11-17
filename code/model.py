import torch.nn as nn


class SimpleFC(nn.Module):
    def __init__(self, featureSizeOfFC):
        super().__init__()
        self.FC1 = nn.Sequential(
            nn.Linear(in_features=featureSizeOfFC, out_features=1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=500),
            nn.BatchNorm1d(500),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.FC3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=300),
            nn.BatchNorm1d(300),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.FC4 = nn.Sequential(
            nn.Linear(in_features=300, out_features=100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.FC5 = nn.Sequential(
            nn.Linear(in_features=100, out_features=30),
            nn.BatchNorm1d(30),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.FC6 = nn.Linear(30,2)


    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        x = self.FC4(x)
        x=self.FC5(x)
        x=self.FC6(x)
        return x


class BiLstmDBP(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(BiLstmDBP,self).__init__()

        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
            dropout=0.5,
            num_layers=num_layers
        )

        self.fc1=nn.Sequential(
            nn.Linear(hidden_size * 2, 500),
            nn.Dropout(0.5),
            nn.LayerNorm(500),
            nn.ReLU(),

        )

        self.fc2=nn.Sequential(
            nn.Linear(500, 300),
            nn.Dropout(0.5),
            nn.LayerNorm(300),
            nn.ReLU(),
        )
        self.fc3=nn.Sequential(
            nn.Linear(300,100),
            nn.Dropout(0.5),
            nn.LayerNorm(100),
            nn.ReLU()
        )
        self.fc4=nn.Linear(100,2)

    # batch_size*seq_len*fea_dim
    def forward(self,input):
        out,(h_,c_)=self.LSTM(input)
        out=self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out