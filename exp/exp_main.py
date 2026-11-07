from data_provider.predict.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DriftAlignNet
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from module.cuda_graph_runner import CUDAGraphInferenceRunner
from module.channel_selectors import (
    EfficientDirectionalLagMMRChannelSelector,
    FastExactDirectionalLagMMRChannelSelector,
    build_global_channel_graph,
    build_global_directional_candidate_graph,
)
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import os
import time


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = DriftAlignNet.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return (data_set, data_loader)

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _driftalignnet_inference_runner(self):
        return CUDAGraphInferenceRunner(self.model, use_amp=self.args.use_amp)

    def _driftalignnet_history_phase(self, batch_cycle):
        anchor_cycle = getattr(self.args, "anchor_cycle", None) or self.args.cycle
        return (batch_cycle - self.args.seq_len) % anchor_cycle

    def _prepare_driftalignnet_global_graph(self, train_data=None):
        """Build selector graphs exclusively from train inputs."""
        model = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )
        selector = getattr(model, "channel_selector", None)
        if not isinstance(
            selector,
            (
                EfficientDirectionalLagMMRChannelSelector,
                FastExactDirectionalLagMMRChannelSelector,
            ),
        ):
            return
        if (
            isinstance(selector, EfficientDirectionalLagMMRChannelSelector)
            and selector.has_global_neighbors
        ):
            return
        if isinstance(selector, FastExactDirectionalLagMMRChannelSelector):
            if (
                self.args.enc_in <= selector.strict_legacy_threshold
                or selector.has_global_candidates
                or (
                    selector.global_candidate_budget >= self.args.enc_in - 1
                    and selector.refine_budget >= self.args.enc_in - 1
                )
            ):
                return
        if train_data is None:
            train_data, _ = self._get_data(flag="train")
        if not hasattr(train_data, "data_x"):
            raise ValueError(
                "The train dataset must expose historical inputs as data_x to build the global channel graph."
            )
        if isinstance(selector, EfficientDirectionalLagMMRChannelSelector):
            neighbor_indices, neighbor_scores = build_global_channel_graph(
                train_data.data_x,
                candidate_budget=selector.candidate_budget,
                eps=selector.eps,
                device=self.device,
            )
            selector.set_global_neighbors(neighbor_indices, neighbor_scores)
            print(
                f"Built train-only global channel graph: {neighbor_indices.size(0)} targets x {neighbor_indices.size(1)} neighbors"
            )
            return
        cache_enabled = bool(getattr(self.args, "global_graph_cache", True))
        dataset_name = os.path.splitext(os.path.basename(self.args.data_path))[0]
        cache_dir = os.path.join("cache", "channel_graph")
        graph_budget = selector.global_candidate_budget
        cache_name = f"{dataset_name}_sl{self.args.seq_len}_lag{selector.max_lag}_train_directional_M{graph_budget}.pt"
        cache_path = os.path.join(cache_dir, cache_name)
        graph = None
        if cache_enabled and os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            metadata = cached.get("metadata", {})
            expected = {
                "num_channels": self.args.enc_in,
                "seq_len": self.args.seq_len,
                "max_lag": selector.max_lag,
                "train_size": int(len(train_data.data_x)),
                "sample_stride": int(
                    getattr(self.args, "global_graph_sample_stride", 24)
                ),
                "max_windows": int(
                    getattr(self.args, "global_graph_max_windows", 1024)
                ),
                "stability_gamma": float(selector.stability_gamma),
                "eps": float(selector.eps),
            }
            if all((metadata.get(key) == value for key, value in expected.items())):
                graph = (cached["indices"], cached["scores"], metadata)
                print(f"Loaded train-only directional candidate graph: {cache_path}")
        if graph is None:
            graph = build_global_directional_candidate_graph(
                train_data.data_x,
                seq_len=self.args.seq_len,
                max_lag=selector.max_lag,
                candidate_budget=graph_budget,
                stability_gamma=selector.stability_gamma,
                eps=selector.eps,
                sample_stride=getattr(self.args, "global_graph_sample_stride", 24),
                max_windows=getattr(self.args, "global_graph_max_windows", 1024),
                device=self.device,
            )
            if cache_enabled:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save(
                    {"indices": graph[0], "scores": graph[1], "metadata": graph[2]},
                    cache_path,
                )
                print(f"Saved train-only directional candidate graph: {cache_path}")
        selector.set_global_candidate_graph(graph[0], graph[1])
        print(
            f"Prepared train-only directional candidate graph: {graph[0].size(0)} targets x {graph[0].size(1)} neighbors"
        )

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_cycle,
            ) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_cycle = batch_cycle.int().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        history_phase = self._driftalignnet_history_phase(batch_cycle)
                        outputs = self.model(batch_x, history_phase)
                else:
                    history_phase = self._driftalignnet_history_phase(batch_cycle)
                    outputs = self.model(batch_x, history_phase)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
        self._prepare_driftalignnet_global_graph(train_data)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        if train_steps == 0:
            raise ValueError(
                "No training batches; reduce batch_size or provide more data."
            )
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_cycle,
            ) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        auxiliary_loss = 0.0
                        history_phase = self._driftalignnet_history_phase(batch_cycle)
                        outputs = self.model(batch_x, history_phase)
                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )
                        loss = criterion(outputs, batch_y) + auxiliary_loss
                        train_loss.append(loss.item())
                else:
                    auxiliary_loss = 0.0
                    history_phase = self._driftalignnet_history_phase(batch_cycle)
                    outputs = self.model(batch_x, history_phase)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y) + auxiliary_loss
                    train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                if self.args.lradj == "TST":
                    adjust_learning_rate(
                        model_optim, scheduler, epoch + 1, self.args, printout=False
                    )
                    scheduler.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != "TST":
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print("Updating learning rate to {}".format(scheduler.get_last_lr()[0]))
        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device, weights_only=True)
        )
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.args.checkpoints, setting, "checkpoint.pth"),
                    map_location=self.device,
                    weights_only=True,
                )
            )
        self._prepare_driftalignnet_global_graph()
        preds = []
        trues = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        driftalignnet_runner = self._driftalignnet_inference_runner()
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_cycle,
            ) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        history_phase = self._driftalignnet_history_phase(batch_cycle)
                        outputs = driftalignnet_runner(batch_x, history_phase)
                else:
                    history_phase = self._driftalignnet_history_phase(batch_cycle)
                    outputs = driftalignnet_runner(batch_x, history_phase)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy().copy()
                batch_y = batch_y.detach().cpu().numpy()
                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
        if driftalignnet_runner is not None:
            if driftalignnet_runner.graph_captured:
                print("DriftAlignNet CUDA Graph inference: enabled")
            driftalignnet_runner.clear()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        amse = []
        amae = []
        for j in range(self.args.pred_len):
            mae_j, mse_j, _, _, _, _, _ = metric(preds[:, j, :], trues[:, j, :])
            amse.append(mse_j)
            amae.append(mae_j)
        mse = np.mean(amse)
        mae = np.mean(amae)
        print("mse:{}, mae:{}".format(mse, mae))
        result_file = "result.txt"
        f = open(result_file, "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n")
        f.write("\n")
        f.close()
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device, weights_only=True)
            )
        self._prepare_driftalignnet_global_graph()
        preds = []
        self.model.eval()
        driftalignnet_runner = self._driftalignnet_inference_runner()
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_cycle,
            ) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_cycle = batch_cycle.int().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        history_phase = self._driftalignnet_history_phase(batch_cycle)
                        outputs = driftalignnet_runner(batch_x, history_phase)
                else:
                    history_phase = self._driftalignnet_history_phase(batch_cycle)
                    outputs = driftalignnet_runner(batch_x, history_phase)
                pred = outputs.detach().cpu().numpy().copy()
                preds.append(pred)
        if driftalignnet_runner is not None:
            if driftalignnet_runner.graph_captured:
                print("DriftAlignNet CUDA Graph inference: enabled")
            driftalignnet_runner.clear()
        preds = np.concatenate(preds, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + "real_prediction.npy", preds)
        return
