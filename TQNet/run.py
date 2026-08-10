import argparse
import os
import sys
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

import channel_criterion

# Repo root (parent of TQNet/) on sys.path, so `common` -- shared, frozen,
# read-only from here -- is importable regardless of the caller's cwd.
# `channel_criterion` does this same insert as a side effect of import above;
# it is repeated here, guarded, so this file does not depend on that order.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common import split as split_mod
from common import results as results_mod
from common import data as data_mod
from tools import estimate_cycle

# report/horizon_sigma.md, section "split_hash by horizon". The frozen
# protocol (report/prereg-improvement.md sec 2) fixes seq_len at 96 for every
# arm and horizon, so pred_len alone keys this table. Deliberately not a
# single hard-coded H=96 constant: this file's run also serves pred_len
# 192/336/720 (J-09 dispatch, "Do not hard-code the H = 96 hash into a path
# that runs at other horizons").
_ETTH1_SPLIT_HASH_BY_PRED_LEN = {
    96: 'b66ee6b47e2b2eb8',
    192: '5b9f41f467356285',
    336: 'a5bcaa4090739908',
    720: '17f9f51a6d81e0a2',
}

parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='TQNet',
                    help='model name, options: [TQNet, Informer, Autoformer, ...]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')  #fixed
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# TQNet & CycleNet
def _cycle_arg(value):
    """--cycle accepts an integer, or the literal string 'auto' (Arm B, J-14:
    report/prereg-improvement.md sec 3). 'auto' is resolved to an int later,
    in the Arm B block below, once --data/--seq_len are known; an integer
    passed here is left exactly as it always was -- this function changes
    nothing about the --cycle 24 path except making 'auto' also legal."""
    if value == 'auto':
        return 'auto'
    return int(value)


parser.add_argument('--cycle', type=_cycle_arg, default=24,
                     help="cycle length, or 'auto' to estimate it from the training "
                          "split (Arm B, J-14, report/prereg-improvement.md sec 3: "
                          "ACF local-maximum and periodogram argmax on the training "
                          "channel-mean, agreeing or the run fails loudly). ETTh1 only.")
parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')
parser.add_argument('--use_tq', type=int, default=1,
                    help='TQNet ablation: 1 keep the Temporal Query, 0 fall back to self-attention')
parser.add_argument('--channel_aggre', type=int, default=1,
                    help='TQNet ablation: 1 keep the channel attention layer, 0 remove it')
parser.add_argument('--channel_criterion', type=int, default=0,
                    help='Arm D (J-09, report/prereg-improvement.md sec 3): 1 computes the '
                         'training-split channel-correlation criterion and overrides '
                         '--use_tq/--channel_aggre with its decision. 0 (default) leaves '
                         '--use_tq/--channel_aggre exactly as passed, so the default '
                         '(both 1) reproduces the published model bit-for-bit. ETTh1 only.')
parser.add_argument('--use_damped_trend', type=int, default=0,
                    help='Arm A (J-11, report/prereg-improvement.md sec 3): 1 enables '
                         'damped-trend instance normalisation -- fit a least-squares '
                         'line per window per channel, subtract it before the instance '
                         'norm, and add slope * sum_{k=1..h} phi^k back after the '
                         'de-normalisation. 0 (default) leaves the instance norm exactly '
                         'as published, so the default reproduces the published model '
                         'bit-for-bit.')
parser.add_argument('--damped_phi', type=float, default=0.9,
                    help='Arm A damping factor phi, 0 < phi <= 1. Ignored unless '
                         '--use_damped_trend 1. phi = 1 is plain linear extrapolation; '
                         'phi -> 0 projects no trend forward. The pre-registration fixes '
                         'the candidate set phi in {0.8, 0.9, 0.95, 1.0} and fixes that '
                         'the choice is made once, on validation MSE at H=96 only, then '
                         'frozen for every horizon. This flag does not make that choice.')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# SegRNN
parser.add_argument('--rnn_type', default='gru', help='rnn_type')
parser.add_argument('--dec_way', default='pmf', help='decode way')
parser.add_argument('--seg_len', type=int, default=48, help='segment length')
parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=0,
                    help='data loader num workers. Upstream defaulted to 10, which on a dataset this '
                         'small costs more in process startup than it saves, and left worker RNG '
                         'unseeded. Sample order is drawn in the parent process, so 0 gives the same '
                         'batches as 10')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
parser.add_argument('--accelerator', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                    help='compute device; auto prefers cuda, then mps, then cpu')

# reproducibility and result plumbing
parser.add_argument('--result_path', type=str, default='./result_ours.txt',
                    help='append-mode log of test metrics. Deliberately NOT result.txt, which holds '
                         'the authors published numbers and is our reference evidence')
parser.add_argument('--save_outputs', type=int, default=1,
                    help='1: save test predictions/targets as .npy plus a metrics.json under ./results/<setting>/')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
# Upstream seeded only the three lines above, which leaves the CUDA generators and
# cuDNN kernel selection free. These calls are no-ops on CPU and do not touch the
# CPU RNG stream, so the published configuration is unaffected.
torch.cuda.manual_seed_all(fix_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.accelerator == 'auto':
    if torch.cuda.is_available():
        args.accelerator = 'cuda'
    elif torch.backends.mps.is_available():
        args.accelerator = 'mps'
    else:
        args.accelerator = 'cpu'
elif args.accelerator == 'cuda' and not torch.cuda.is_available():
    raise SystemExit('--accelerator cuda requested but torch.cuda.is_available() is False')
elif args.accelerator == 'mps' and not torch.backends.mps.is_available():
    raise SystemExit('--accelerator mps requested but torch.backends.mps.is_available() is False')

args.use_gpu = args.accelerator in ('cuda', 'mps')

if args.accelerator == 'cuda' and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# --- Arm D (J-09): channel-count-conditional TQ/attention criterion --------
# report/prereg-improvement.md sec 3, Arm D. Computed on the training split
# only -- common.split.borders(seq_len)['train'], rows [0, 8640) for ETTh1,
# not one row more -- never on validation or test (requirement B2). No
# training happens in this block; it only decides use_tq/channel_aggre
# before the model is built.
args.channel_criterion_record = None
if args.channel_criterion:
    if args.data != 'ETTh1':
        raise SystemExit(
            "--channel_criterion is only defined for ETTh1: it relies on "
            "common.split's ETT 12/4/4-month split scheme, which does not "
            "describe --data {!r}".format(args.data)
        )
    _cc_record = channel_criterion.evaluate_criterion(
        args.root_path, args.data_path, args.seq_len
    )
    args.use_tq = _cc_record['use_tq']
    args.channel_aggre = _cc_record['channel_aggre']
    args.channel_criterion_record = _cc_record
    print(
        '[channel_criterion] {}={:.6f} threshold={} (justification: {}) '
        'rows={} -> decision={} (use_tq={}, channel_aggre={})'.format(
            _cc_record['statistic_name'],
            _cc_record['statistic_value'],
            _cc_record['threshold_value'],
            _cc_record['threshold_justification'],
            _cc_record['row_range'],
            _cc_record['decision'],
            args.use_tq,
            args.channel_aggre,
        )
    )

# --- Arm A (J-11): damped-trend instance normalisation ---------------------
# report/prereg-improvement.md sec 3, Arm A. A config-driven switch only: the
# mechanism lives in TQNet/layers/DampedTrend.py and is wired into
# TQNet/models/TQNet.py's forward pass. Nothing is selected here -- phi is
# taken as given. The pre-registration fixes the candidate set and fixes that
# phi is chosen once, on validation MSE at H=96, and then frozen; that
# selection is J-12's job, not this flag's.
if args.use_damped_trend:
    if not (0.0 < args.damped_phi <= 1.0):
        raise SystemExit(
            "--damped_phi must satisfy 0 < phi <= 1 (report/prereg-improvement.md "
            "sec 3, Arm A), got {!r}".format(args.damped_phi)
        )
    if args.damped_phi not in (0.8, 0.9, 0.95, 1.0):
        print(
            '[damped_trend] WARNING: phi={!r} is outside the pre-registered '
            'candidate set {{0.8, 0.9, 0.95, 1.0}}'.format(args.damped_phi)
        )
    print(
        '[damped_trend] enabled: phi={:g}, seq_len={}, pred_len={} '
        '(trend added at step h = slope * phi * (1 - phi**h) / (1 - phi))'.format(
            args.damped_phi, args.seq_len, args.pred_len
        )
    )

# --- Arm B (J-14): estimate the period from the training split ------------
# report/prereg-improvement.md sec 3, Arm B. Computed on the training split
# only -- common.split.borders(seq_len)['train'], rows [0, 8640) for ETTh1,
# not one row more (requirement B2), never on validation or test. Mirrors
# the channel_criterion block above: a switch that resolves itself, from the
# training split alone, before the model is built, and prints its own
# decision. args.cycle_source records whether the value below reached
# args.cycle by estimation or by being typed on the command line, so
# resolved_config.json (TQNet/exp/exp_main.py) can tell the two apart even
# when they resolve to the same integer.
args.cycle_source = 'passed'
args.cycle_estimate_record = None
if args.cycle == 'auto':
    if args.data != 'ETTh1':
        raise SystemExit(
            "--cycle auto is only defined for ETTh1: it relies on "
            "common.split's ETT 12/4/4-month split scheme, which does not "
            "describe --data {!r}".format(args.data)
        )
    _cy_csv_path = os.path.join(args.root_path, args.data_path)
    _cy_train_start, _cy_train_stop = split_mod.borders(args.seq_len)['train']
    _cy_series = estimate_cycle.load_channel_mean(_cy_csv_path, _cy_train_start, _cy_train_stop)
    try:
        _cy_period, _cy_record = estimate_cycle.estimate_or_raise(
            _cy_series,
            label='{} channel-mean train[{}, {})'.format(args.data, _cy_train_start, _cy_train_stop),
        )
    except estimate_cycle.CycleDisagreementError as _cy_exc:
        raise SystemExit(
            "--cycle auto: {} (report/prereg-improvement.md sec 3 'Arm B': "
            "abandon)".format(_cy_exc)
        )
    args.cycle = _cy_period
    args.cycle_source = 'estimated'
    args.cycle_estimate_record = _cy_record
    print(
        '[cycle_auto] acf={} (peak={}) periodogram={} (power={}) agree={} '
        'rows=[{}, {}) -> cycle={} (source=estimated)'.format(
            _cy_record['acf_period'],
            _cy_record['acf_peak_value'],
            _cy_record['periodogram_period'],
            _cy_record['periodogram_power'],
            _cy_record['agree'],
            _cy_train_start,
            _cy_train_stop,
            args.cycle,
        )
    )

# Every ETTh1 run asserts the split fingerprint (standing order 12), with the
# two-argument form `assert_split_hash(expected, actual)` (common/results.py
# -- the pre-registration's one-argument form in sec 2 does not match the
# code; the code wins, standing order 5). Established here; later arms copy
# this call rather than re-deriving it.
if args.data == 'ETTh1':
    _split_csv_path = os.path.join(args.root_path, args.data_path)
    if os.path.exists(_split_csv_path):
        _expected_hash = _ETTH1_SPLIT_HASH_BY_PRED_LEN.get(args.pred_len)
        if _expected_hash is None:
            raise SystemExit(
                "no recorded split hash for --pred_len {}; "
                "report/horizon_sigma.md only covers 96/192/336/720 at "
                "seq_len=96".format(args.pred_len)
            )
        _actual_hash = split_mod.split_hash(
            args.seq_len, args.pred_len, data_mod.data_sha256(_split_csv_path)
        )
        results_mod.assert_split_hash(_expected_hash, _actual_hash)
        print('[split_hash] expected={} actual={} OK'.format(_expected_hash, _actual_hash))
    else:
        print('[split_hash] WARNING: {} not found, split hash not asserted'.format(_split_csv_path))


def release_cache():
    """Free accelerator memory where the concept exists."""
    if args.accelerator == 'cuda':
        torch.cuda.empty_cache()
    elif args.accelerator == 'mps':
        torch.mps.empty_cache()


print('Args in experiment:')
print(args)

Exp = Exp_Main

# Empty for the published configuration, so reproduction runs keep their original
# checkpoint paths and result labels untouched.
abl_tag = '' if (args.use_tq == 1 and args.channel_aggre == 1) \
    else '_tq{}ca{}'.format(args.use_tq, args.channel_aggre)

# Same rule for Arm A: empty unless the arm is on, so no existing run's
# checkpoint path or results directory changes name.
abl_tag += '' if not args.use_damped_trend else '_dphi{:g}'.format(args.damped_phi)


if args.is_training:
    for ii in range(args.itr):

        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.cycle,
            fix_seed,
            abl_tag)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        release_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.cycle,
        fix_seed,
        abl_tag)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    release_cache()
