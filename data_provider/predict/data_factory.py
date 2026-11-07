from data_provider.predict.data_loader import (
    Dataset_Custom,
    Dataset_EnergyPower,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_RealE,
)
from torch.utils.data import DataLoader

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
    "energy": Dataset_EnergyPower,
    "power": Dataset_EnergyPower,
    "RealE": Dataset_RealE,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1
    dataset_flag = flag
    if flag == "test":
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        dataset_flag = "test"
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    calendar_cycle = args.cycle
    if args.model == "DriftAlignNet" and getattr(args, "anchor_cycle", None):
        calendar_cycle = args.anchor_cycle
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=dataset_flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        timeenc=timeenc,
        freq=freq,
        cycle=calendar_cycle,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return (data_set, data_loader)
