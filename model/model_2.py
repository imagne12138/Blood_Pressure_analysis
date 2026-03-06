import torch
import torch.nn as nn
# 参数多4w

class Model_2_Head(nn.Module):
    """
    三层1dcnn, 双层Bilstm, 线性注意力 
    最后对中间向量c分头输出，sbp和dbp各一个头(在output后加头和在output前加头)
    """
    def __init__(self, filters, num_layers, num_directions=2, hidden_dim=128, drop_prob=0.2, attn_out_dim=1, out_dim=2):
        super(Model_2_Head, self).__init__()
        
        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.filters = filters
        self.num_directions = num_directions
        self.attn_out = attn_out_dim
        self.output = out_dim

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.filters[i], 
                                    out_channels=self.filters[i+1], 
                                    kernel_size=3, 
                                    stride=1,
                                    padding=1),
                          nn.BatchNorm1d(num_features=self.filters[i+1]),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=2, stride=2)) # pool stride可选1，对比
                for i in range(len(self.filters) - 1)
            ]) # 输入1dCNN: [batch_size, feature_dim=1, window_size(seq_len)=1024]
               # 输出：[batch_size, feature_dim=128, window_size=128] 如果poolstride取1则还是1024
        if self.num_directions == 2:
            self.bilstm = nn.LSTM(input_size=128, 
                                hidden_size=self.hidden_dim, 
                                num_layers=self.num_layers, 
                                batch_first=True,
                                dropout=drop_prob, 
                                bidirectional=True) # 输入前先permute: (batch_size, window_size=128, feature_dim=128)
                                                    # 输出：(batch_size, seq_len, hidden_sizeXnum_directions)
                                                    # (batch_size, window_size=128, 128 X 2 = 256)
                                                    # hidden: (numdirectionsXn_layers=2X2=4, batch_size, hidden_size=128)
        else:
            self.bilstm = nn.LSTM(input_size=128, 
                                hidden_size=self.hidden_dim, 
                                num_layers=self.num_layers, 
                                batch_first=True,
                                dropout=drop_prob, 
                                bidirectional=False)
        # self.linear_attn = nn.Linear(self.hidden_dim * self.num_directions, self.attn_out) # 单层attention
        self.linear_attn = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_directions, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.attn_out),
        )

        # 方案1
        self.linear_sbp_out = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_directions, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.hidden_dim, 1)
        )
        self.linear_dbp_out = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_directions, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.hidden_dim, 1)
        )
        # 方案2
        # self.linear_out = nn.Sequential(
        #     nn.Linear(self.hidden_dim * self.num_directions, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(drop_prob),
        #     nn.Linear(self.hidden_dim, self.output),
        #     nn.Dropout(drop_prob)
        # ) # [batch_size, 2]

        # self.sbp_head = nn.Linear(self.hidden_dim, 1)
        # self.dbp_head = nn.Linear(self.hidden_dim, 1)


    def forward(self, x):
        """
        输入x形状为[batch_size, feature_dim=1, window_size=1024]
        输出为[batch_size, 2], 同时学习dbp和sbp特征
        """
        # x shape: [batch_size, feature_dim=1, window_size=1024]
        conv_out = x
        for conv in self.convs:
            conv_out = conv(conv_out) # conv_out shape: [batch_size, feature_dim=128, window_size=128]
        conv_out = conv_out.permute(0, 2, 1) # lstm输入：[batch_size, window_size, feature_dim]
        lstm_out, hidden_state = self.bilstm(conv_out) # 输出output: [batch_size, seq_len, hidden_sizeXnum_directions], hidden: (numdirectionsXn_layers=2X2=4, batch_size, hidden_size=128)
        lstm_out = self.dropout(lstm_out)
        # e = self.tanh(self.linear_attn(lstm_out)) # [batch_size, window_size, 1]
        e = self.linear_attn(lstm_out) # [batch_size, window_size, 1]
        alpha = self.softmax(e) # [batch_size, window_size, 1]
        c = torch.sum(alpha * lstm_out, dim=1) # [batch_size, hidden_sizeXnum_directions]

        # 方案一输出
        dbp_out = self.linear_dbp_out(c) # [batch_size, 1]
        sbp_out = self.linear_sbp_out(c) # [batch_size, 1]
        
        output = torch.concat((sbp_out, dbp_out), dim=1) # [batch_size, 2]

        # 方案二输出
        # output = self.linear_out(c) # [batch_size, 2]
        # sbp_out = self.sbp_head(output)
        # dbp_out = self.dbp_head(output)

        return output # [batch_size, 2]


