from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, SegRNN, CycleNet, \
    iTransformer, TimeXer, TQNet, TQDLinear, TQPatchTST, TQiTransformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import json
import os
import time
from datetime import datetime

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


def _load_state_dict(path, device):
    """Load a checkpoint written on any device.

    `map_location` lets a CUDA-trained checkpoint be evaluated on CPU or MPS, and
    `weights_only=True` is the torch >= 2.6 default made explicit so the call does
    not change meaning under a different torch version.
    """
    return torch.load(path, map_location=device, weights_only=True)


# J-12a Step 1 (STAGE2_WORKPLAN_2026-08-09.md sec 7i): the run writes its own
# resolved config beside the checkpoint, so nothing downstream ever has to
# reverse-engineer it from the checkpoint directory name again. `args` here is
# the object as it stands *after* run.py's --channel_criterion (Arm D) block
# has already overridden use_tq/channel_aggre -- these two functions only ever
# see the resolved values, never the raw command line, because they are called
# from inside Exp_Main, which run.py constructs after that block runs.
RESOLVED_CONFIG_SCHEMA = 1


def _resolved_model_fields(args, model):
    """The subset of fields that determine the model's *architecture* --

    i.e. the ones tools/validation_metrics.py must get right to rebuild a
    model whose state_dict matches the checkpoint. Factored out of
    `_write_resolved_config` so `test()`'s `summary` dict (which already
    carries its own broader set of fields, see line ~397) can merge these in
    once instead of repeating each getattr/int/float cast.
    """
    return {
        'enc_in': args.enc_in,
        'model_type': args.model_type,
        'use_revin': int(args.use_revin),
        'use_tq': int(getattr(args, 'use_tq', 1)),
        'channel_aggre': int(getattr(args, 'channel_aggre', 1)),
        # Whether Arm D's criterion (not this flag) is what decided use_tq /
        # channel_aggre above. Recorded so a resolved_config.json can be told
        # apart from a plain --use_tq/--channel_aggre run even though both
        # end up with the same two flag values.
        'channel_criterion': int(getattr(args, 'channel_criterion', 0)),
        'use_damped_trend': int(getattr(args, 'use_damped_trend', 0)),
        'damped_phi': float(getattr(args, 'damped_phi', 0.9)),
    }


def _arm_label(args):
    """Which arm this run's resolved config represents, by the same convention
    tools/backfill_checkpoint_config.py and tools/validation_metrics.py use:
    'armA' if the damped-trend flag is on (J-11), else 'reconstruction' if both
    ablation flags are at their published defaults, else 'armD' (J-09 -- whether
    --channel_criterion decided the flags or they were passed directly, as
    repro/run_etth1_ablation.sh does).
    """
    if int(getattr(args, 'use_damped_trend', 0)):
        return 'armA'
    if int(getattr(args, 'use_tq', 1)) == 1 and int(getattr(args, 'channel_aggre', 1)) == 1:
        return 'reconstruction'
    return 'armD'


def _write_resolved_config(args, model, setting, checkpoint_dir):
    """Write `resolved_config.json` beside `checkpoint.pth`.

    Every field `tools/validation_metrics.py`'s `build_args()` needs to
    rebuild this exact model, read from the resolved `args` object (not the
    raw command line -- see the module-level note above). `checkpoints/` is
    gitignored, so this file is never itself committed; Step 4 of J-12a
    embeds this same content into the committed `results/validation/*.json`
    sidecar instead, which is what actually carries T15' traceability.
    """
    config = {
        'setting': setting,
        'arm': _arm_label(args),
        'model': args.model,
        'data': args.data,
        'model_id': getattr(args, 'model_id', None),
        'features': args.features,
        'target': getattr(args, 'target', 'OT'),
        'freq': getattr(args, 'freq', 'h'),
        'data_path': getattr(args, 'data_path', None),
        'seq_len': args.seq_len,
        'label_len': getattr(args, 'label_len', 0),
        'pred_len': args.pred_len,
        'cycle': args.cycle,
        # Arm B (J-14, report/prereg-improvement.md sec 3): 'estimated' if
        # --cycle auto resolved this value from the training split
        # (TQNet/run.py), 'passed' if it was typed on the command line --
        # the default for every run that predates this flag and for every
        # plain --cycle <int> run after it, so this is purely additive.
        'cycle_source': getattr(args, 'cycle_source', 'passed'),
        'seed': getattr(args, 'random_seed', None),
        'accelerator': getattr(args, 'accelerator', 'unknown'),
        'd_model': getattr(args, 'd_model', None),
        'dropout': getattr(args, 'dropout', None),
        'batch_size': getattr(args, 'batch_size', None),
        'n_params': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        'resolved_config_schema': RESOLVED_CONFIG_SCHEMA,
        'written_by': 'TQNet/exp/exp_main.py Exp_Main.train (J-12a Step 1)',
        'written_at': datetime.now().isoformat(timespec='seconds'),
    }
    config.update(_resolved_model_fields(args, model))

    out_path = os.path.join(checkpoint_dir, 'resolved_config.json')
    with open(out_path, 'w') as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write('\n')
    return config


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SegRNN': SegRNN,
            'CycleNet': CycleNet,
            'iTransformer': iTransformer,
            'TimeXer': TimeXer,
            'TQNet': TQNet,
            'TQDLinear': TQDLinear,
            'TQPatchTST': TQPatchTST,
            'TQiTransformer': TQiTransformer
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(_load_state_dict(best_model_path, self.device))

        # J-12a Step 1: resolved_config.json, written from the model that was
        # just loaded from the best checkpoint, so n_params and every
        # architecture flag in it describe exactly the weights on disk next
        # to it -- not the args as passed on the command line before Arm D's
        # channel_criterion (if used) had a chance to override them.
        _write_resolved_config(self.args, self.model, setting, path)

        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(
                _load_state_dict(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), self.device))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()

                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    # np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                    # np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        ### denorm ###
        # denorm_preds = np.stack([test_data.inverse_transform(pred) for pred in preds])
        # denorm_trues = np.stack([test_data.inverse_transform(true) for true in trues])

        ### denorm ###

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        # mae, mse, rmse, mape, mspe, rse, corr = metric(denorm_preds, denorm_trues)

        print('mse:{}, mae:{}'.format(mse, mae))

        # Upstream appended to a hard-coded ./result.txt, which is the same file that
        # ships the authors published numbers. We write to args.result_path instead so
        # the reference evidence survives a run.
        result_path = getattr(self.args, 'result_path', './result_ours.txt')
        with open(result_path, 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')

        # Upstream left these saves commented out, so a finished run left nothing
        # behind but a printed line. The report needs the raw window-level arrays to
        # compute the course metrics, to draw charts, and to score the baseline on
        # exactly these windows.
        if getattr(self.args, 'save_outputs', 1):
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            summary = {
                'setting': setting,
                'model': self.args.model,
                'data': self.args.data,
                'features': self.args.features,
                'seq_len': self.args.seq_len,
                'pred_len': self.args.pred_len,
                'cycle': self.args.cycle,
                'seed': self.args.random_seed,
                'use_tq': getattr(self.args, 'use_tq', 1),
                'channel_aggre': getattr(self.args, 'channel_aggre', 1),
                'accelerator': getattr(self.args, 'accelerator', 'unknown'),
                'n_windows': int(preds.shape[0]),
                'n_params': int(sum(p.numel() for p in self.model.parameters() if p.requires_grad)),
                # As reported by the upstream metric() helper, on z-scored data.
                'upstream_mse': float(mse),
                'upstream_mae': float(mae),
                'upstream_rmse': float(rmse),
            }
            # J-12a Step 1: the same resolved-config fields that go into
            # resolved_config.json beside checkpoint.pth, so
            # tools/collect_results.py's ingestion of TQNet/results/<setting>/
            # gets them too, not just the checkpoint-adjacent file.
            summary.update(_resolved_model_fields(self.args, self.model))
            with open(folder_path + 'metrics.json', 'w') as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
                handle.write('\n')
            print('saved predictions and metrics under {}'.format(folder_path))

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(_load_state_dict(best_model_path, self.device))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
