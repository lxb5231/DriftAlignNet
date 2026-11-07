import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        cycle=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        cycle=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        cycle=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]

        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_RealE(Dataset):
    """Real-E country data with the benchmark's chronological preprocessing."""

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="France (FR).csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        cycle=None,
    ):
        if size is None:
            self.seq_len = 12
            self.label_len = 12
            self.pred_len = 12
        else:
            self.seq_len, self.label_len, self.pred_len = size

        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {flag!r}.")
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = int(cycle) if cycle is not None else 24
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(
            os.path.join(self.root_path, self.data_path), low_memory=False
        )

        time_columns = [c for c in ("date", "MTU") if c in df_raw.columns]
        if not time_columns:
            raise ValueError("Real-E data must contain a 'date' or 'MTU' column.")
        df_raw = df_raw.rename(columns={time_columns[0]: "date"})
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        df_raw = df_raw.dropna(subset=["date"]).sort_values("date")
        df_raw = df_raw.reset_index(drop=True)

        value_columns = [c for c in df_raw.columns if c != "date"]
        for column in value_columns:
            values = (
                df_raw[column].astype(str).str.replace(",", "", regex=False).str.strip()
            )
            df_raw[column] = pd.to_numeric(values, errors="coerce")
        df_raw = df_raw.dropna(axis=1, how="all")

        value_columns = [c for c in df_raw.columns if c != "date"]
        if not value_columns:
            raise ValueError("Real-E data contains no numeric feature columns.")
        if self.features in {"S", "MS"}:
            if self.target not in value_columns:
                raise ValueError(
                    f"Target {self.target!r} is not a Real-E feature column."
                )
            if self.features == "S":
                value_columns = [self.target]
            else:
                value_columns = [c for c in value_columns if c != self.target] + [
                    self.target
                ]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_val = len(df_raw) - num_train - num_test
        self.split_sizes = {
            "train": num_train,
            "val": num_val,
            "test": num_test,
        }
        border1s = [0, num_train, num_train + num_val]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        values = df_raw[value_columns].to_numpy(dtype=np.float64)
        if self.scale:
            self.scaler.fit(values[:num_train])
            values = self.scaler.transform(values)
        # Match Real-E's temporal interpolation and edge filling. Applying it
        # after scaling is equivalent for this affine transform.
        values = (
            pd.DataFrame(values)
            .interpolate(method="linear", axis=0)
            .ffill()
            .bfill()
            .to_numpy(dtype=np.float32)
        )
        if not np.isfinite(values).all():
            raise ValueError("Real-E preprocessing left non-finite values.")

        stamps = df_raw[["date"]].iloc[border1:border2].copy()
        if self.timeenc == 0:
            stamps["month"] = stamps.date.dt.month
            stamps["day"] = stamps.date.dt.day
            stamps["weekday"] = stamps.date.dt.weekday
            stamps["hour"] = stamps.date.dt.hour
            stamps["minute"] = stamps.date.dt.minute
            data_stamp = stamps.drop(columns=["date"]).to_numpy()
        else:
            data_stamp = time_features(
                pd.to_datetime(stamps["date"].values), freq=self.freq
            ).transpose(1, 0)

        self.data_x = values[border1:border2]
        self.data_y = values[border1:border2]
        self.data_stamp = data_stamp
        self.cycle_index = (np.arange(len(df_raw), dtype=np.int64) % self.cycle)[
            border1:border2
        ]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
            torch.tensor(self.cycle_index[s_end]),
        )

    def __len__(self):
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_EnergyPower(Dataset_Custom):
    """CSV loader for energy.csv and power.csv with chronological 60/20/20 splits."""

    DEFAULT_TARGETS = {
        "energy.csv": "Appliances",
        "power.csv": "Zone 1 Power Consumption",
    }

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = (
            self.data_path
            if os.path.isabs(self.data_path)
            else os.path.join(self.root_path, self.data_path)
        )
        df_raw = pd.read_csv(file_path)
        if df_raw.empty:
            raise ValueError("{} contains no data rows.".format(file_path))

        # energy.csv uses "date" while power.csv uses "DateTime".
        df_raw.columns = [str(column).strip() for column in df_raw.columns]
        time_columns = {
            column.lower(): column
            for column in df_raw.columns
            if column.lower() in ("date", "datetime")
        }
        if not time_columns:
            raise ValueError(
                "{} must contain a date or DateTime column.".format(file_path)
            )
        time_column = time_columns.get("date", time_columns.get("datetime"))
        if time_column != "date":
            df_raw = df_raw.rename(columns={time_column: "date"})

        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        invalid_dates = int(df_raw["date"].isna().sum())
        if invalid_dates:
            raise ValueError(
                "{} contains {} invalid timestamps.".format(file_path, invalid_dates)
            )
        df_raw = df_raw.sort_values("date").reset_index(drop=True)

        feature_columns = [column for column in df_raw.columns if column != "date"]
        if not feature_columns:
            raise ValueError("{} contains no feature columns.".format(file_path))
        for column in feature_columns:
            df_raw[column] = pd.to_numeric(
                df_raw[column].astype(str).str.strip(), errors="coerce"
            )
        invalid_values = int(df_raw[feature_columns].isna().sum().sum())
        if invalid_values:
            raise ValueError(
                "{} contains {} missing or non-numeric feature values.".format(
                    file_path, invalid_values
                )
            )

        default_target = self.DEFAULT_TARGETS.get(os.path.basename(file_path).lower())
        if self.target in feature_columns:
            target = self.target
        elif self.target == "OT" and default_target in feature_columns:
            target = default_target
        elif self.features == "M":
            target = None
        else:
            raise ValueError(
                "Target '{}' was not found in {}. Available features: {}".format(
                    self.target, file_path, feature_columns
                )
            )

        if target is not None:
            other_columns = [column for column in feature_columns if column != target]
            df_raw = df_raw[["date"] + other_columns + [target]]
            self.target = target

        total_rows = len(df_raw)
        num_train = int(total_rows * 0.6)
        num_vali = int(total_rows * 0.2)
        num_test = total_rows - num_train - num_vali
        if (
            num_train < self.seq_len + self.pred_len
            or num_vali < self.pred_len
            or num_test < self.pred_len
        ):
            raise ValueError(
                "The 60/20/20 splits are too short for seq_len={} and pred_len={}.".format(
                    self.seq_len, self.pred_len
                )
            )

        split_starts = [
            0,
            num_train - self.seq_len,
            num_train + num_vali - self.seq_len,
        ]
        split_ends = [num_train, num_train + num_vali, total_rows]
        border1 = split_starts[self.set_type]
        border2 = split_ends[self.set_type]
        self.split_sizes = {
            "train": num_train,
            "val": num_vali,
            "test": num_test,
        }

        if self.features in ("M", "MS"):
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("features must be one of 'M', 'S', or 'MS'.")

        if self.scale:
            train_data = df_data.iloc[:num_train]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]].iloc[border1:border2].copy()
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour)
            data_stamp = df_stamp.drop(columns=["date"]).values
        else:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            ).transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        cycle = self.cycle if self.cycle is not None else 1
        self.cycle_index = (np.arange(total_rows) % cycle)[border1:border2]


## TODO add cycle
