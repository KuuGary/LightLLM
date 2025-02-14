from data_provider.data_loader import Dataset_Solar, Dataset_Solar_V2, Dataset_Solar_V2_UNSEEN,Dataset_Loc_UNSEEN
from torch.utils.data import DataLoader

data_dict = {
    'solar':Dataset_Solar,
    'solar_v2':Dataset_Solar_V2,
    'solar_v2_unseen':Dataset_Solar_V2_UNSEEN,
    'loc_unseen':Dataset_Loc_UNSEEN
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    print(Data)
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'solar_v2_unseen' or args.data == 'loc_unseen':
        data_set = Data(
            root_path=args.root_path,
            train_data_path=args.data_path,
            test_data_path=args.val_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
