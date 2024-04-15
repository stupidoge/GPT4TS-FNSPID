import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.tools import convert_tsf_to_dataframe
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings('ignore')

class StockDataset(Dataset):
    def __init__(self, df_raw, root_path, flag='train', data_path='your_stock_data.csv', features='M', target='Close',
                 scale=True, size=None, timeenc=0, freq='h', percent=100, max_len=-1, train_all=False,
                 split_ratios={'train': 0.75, 'val': 0.10, 'test': 0.15}
                 ): # seq_len=50, label_len=20, pred_len=15

        # features = ['Volume', 'Open', 'High', 'Low', 'Close', 'Adj Close']
        # target = 'Close'

        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag
        self.split_ratios = split_ratios
        self.percent = percent
        self.df_raw = df_raw

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        if size == None:
            # self.seq_len = 50
            # self.label_len = 20
            # self.pred_len = 15
            self.seq_len = 50
            self.label_len = 50
            self.pred_len = 3
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]


        self.scaler = MinMaxScaler()
        self.__read_data__()


        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1



    def __read_data__(self):
        # df_selected = df[self.features]
        # print(df_raw.shape)
        # print(len(df_raw))
        # Split data into train, validation, and test sets
        df_raw = self.df_raw
        num_train = int(len(df_raw) * self.split_ratios['train'])
        num_val = int(len(df_raw) * self.split_ratios['val'])
        num_test = int(len(df_raw) * self.split_ratios['test'])
        print('num_train: {}, num_val: {}, num_test:{}'.format(num_train, num_val, num_test))



        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.fit_transform(df_data.values)
        else:
            data = df_data.values

        # use date_stamp as extral feature
        df_stamp = df_raw[['Date']][border1:border2]
        df_stamp['Date'] = pd.to_datetime(df_stamp.Date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['Date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)



        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


