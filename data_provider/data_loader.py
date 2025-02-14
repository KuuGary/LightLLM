import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import torch

warnings.filterwarnings('ignore')



class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='custom.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
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


# ====================  solar dataloader ===========================
class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='solar_data.csv',
                 target='solar1', scale=False, timeenc=0, freq='h', percent=100,seasonal_patterns=None):
        # size example: [24*4*4, 24*4, 24*4] for seq_len, label_len, pred_len respectively
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # the first column is 'date', followed by feature columns, then target columns
        feature_cols = [f'wavelength{i}' for i in range(1, 19)]  # wavelength 1-18
        target_col = self.target # solar 1

        # Split data into features (X) and target (Y)
        df_features = df_raw[feature_cols]
        df_target = df_raw[[target_col]]
        # print(df_features)
        # print(df_target)

        # Combining features and target for scaling
        df_combined = pd.concat([df_features, df_target], axis=1)
        
        if self.scale:
            combined = pd.concat([df_features, df_target], axis=1)
            combined_scaled = self.scaler.fit_transform(combined)
            # Assuming the target is the last column after scaling
            df_features = combined_scaled[:, :-1]
            df_target = combined_scaled[:, -1].reshape(-1, 1)
        else:
            df_features = df_features.values
            df_target = df_target.values

        # Concatenate features and target back together
        data = np.concatenate([df_features, df_target], axis=1)

        # split dataset
        total_samples = len(data)
        train_end = int(total_samples * 0.8)  # 80% for train
        val_end = train_end + int(total_samples * 0.1)  # next 10% for validation

        if self.set_type == 0:
            self.data = data[:train_end- self.seq_len]
        elif self.set_type == 1:
            self.data = data[train_end:val_end- self.seq_len]
        else:  
            self.data = data[val_end:]

        self.data_x = self.data[:, :-1]
        self.data_y = self.data[:, -1].reshape(-1, 1)

        
        # Mocking time stamp features
        self.data_stamp = np.zeros((len(df_features), 4)) 

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# ====================  solar dataloader v2===========================
class Dataset_Solar_V2(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='solar1_v2.csv',
                 target='value', scale=False, timeenc=0, freq='h', percent=100,seasonal_patterns=None):
        if size is None:
            self.seq_len = 18*21
            self.label_len = 3*21
            self.pred_len = 3*21
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))

        values = df['value'].values
        

        if self.scale:
            scaler = StandardScaler()
            values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
        data = values

        # split dataset
        total_samples = len(data)
        train_end = int(total_samples * 0.8)  # 80% for train
        val_end = train_end + int(total_samples * 0.1)  # next 10% for validation

        if self.set_type == 0:
            self.data = data[:train_end]
        elif self.set_type == 1:
            self.data = data[train_end:val_end]
        else:  
            self.data = data[val_end:]

        # Mocking time stamp features
        self.data_stamp = np.zeros((len(values), 4)) 

    def __getitem__(self, index):
        s_begin = (index // 21) * 21 # This line ensure the data always with structure [18 wavelengths + 3 solar energy]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float).unsqueeze(-1), torch.tensor(seq_y, dtype=torch.float).unsqueeze(-1), torch.tensor(seq_x_mark, dtype=torch.float), torch.tensor(seq_y_mark, dtype=torch.float)


    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar_V2_UNSEEN(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', train_data_path='solar1_v2.csv',
                 test_data_path='solar2_v2.csv',
                 target='value', scale=False, timeenc=0, freq='h', percent=100, seasonal_patterns=None):
        if size is None:
            self.seq_len = 18*21  # 18 wavelengths + 3 solar readings
            self.label_len = 3*21  # Only solar readings
            self.pred_len = 3*21   # Only solar readings
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.root_path = root_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.__read_data__()

    def __read_data__(self):
        if self.flag == 'train':
            df = pd.read_csv(os.path.join(self.root_path, self.train_data_path))
        else:
            df = pd.read_csv(os.path.join(self.root_path, self.test_data_path))
            # Calculate the split point for validation and test data
            split_point = int(len(df) * 0.5)
            if self.flag == 'val':
                df = df[:split_point]
            elif self.flag == 'test':
                df = df[split_point:]

        values = df['value'].values
        print(len(values))
        if self.scale:
            scaler = StandardScaler()
            values = scaler.fit_transform(values.reshape(-1, 1)).flatten()

        self.data = values
        # Mocking time stamp features
        self.data_stamp = np.zeros((len(values), 4))

    def __getitem__(self, index):
        s_begin = (index // 21) * 21
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (torch.tensor(seq_x, dtype=torch.float).unsqueeze(-1), 
                torch.tensor(seq_y, dtype=torch.float).unsqueeze(-1), 
                torch.tensor(seq_x_mark, dtype=torch.float), 
                torch.tensor(seq_y_mark, dtype=torch.float))

    def __len__(self):
        return (len(self.data) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# ====================  localization dataloader ===========================
class Dataset_Loc_UNSEEN(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', train_data_path='data-night2-i_v2.csv',
                 test_data_path='data-night5-i_v2.csv',
                 target='value', scale=False, timeenc=0, freq='h', percent=100,seasonal_patterns=None):
        # size example: [24*4*4, 24*4, 24*4] for seq_len, label_len, pred_len respectively
        if size is None:
            self.seq_len = 27*21
            self.label_len = 1 * 21
            self.pred_len = 1 * 21
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.root_path = root_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.__read_data__()

    def __read_data__(self):
        if self.flag == 'train':
            df = pd.read_csv(os.path.join(self.root_path, self.train_data_path))
        else:
            df = pd.read_csv(os.path.join(self.root_path, self.test_data_path))
            # Calculate the split point for validation and test data
            split_point = int(len(df) * 0.5)
            if self.flag == 'val':
                df = df[:split_point]
            elif self.flag == 'test':
                df = df[split_point:]

        values = df['value'].values
        print(len(values))
        if self.scale:
            scaler = StandardScaler()
            values = scaler.fit_transform(values.reshape(-1, 1)).flatten()

        self.data = values
        # Mocking time stamp features
        self.data_stamp = np.zeros((len(values), 4))

    def __getitem__(self, index):
        s_begin = (index // 28) * 28
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (torch.tensor(seq_x, dtype=torch.float).unsqueeze(-1), 
                torch.tensor(seq_y, dtype=torch.float).unsqueeze(-1), 
                torch.tensor(seq_x_mark, dtype=torch.float), 
                torch.tensor(seq_y_mark, dtype=torch.float))

    def __len__(self):
        return (len(self.data) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


