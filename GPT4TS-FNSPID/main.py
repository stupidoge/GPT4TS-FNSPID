from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./datasets/traffic/')
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--num_pred', type=int, default=50)


args = parser.parse_args()

SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

# BRK-B non exist


names_50 = ['AAL.csv', 'AAPL.csv', 'ABBV.csv', 'AMD.csv', 'AMGN.csv', 'AMZN.csv', 'BABA.csv',
            'BHP.csv', 'BIDU.csv', 'BIIB.csv', 'C.csv', 'CAT.csv', 'CMCSA.csv', 'CMG.csv', 'BRK-B.csv',
            'COP.csv', 'COST.csv', 'CRM.csv', 'CVX.csv', 'DAL.csv', 'DIS.csv', 'EBAY.csv', 'GE.csv',
            'GILD.csv', 'GLD.csv', 'GOOG.csv', 'GSK.csv', 'INTC.csv', 'KO.csv', 'MRK.csv', 'MSFT.csv',
            'MU.csv', 'NKE.csv', 'nvda.csv', 'ORCL.csv', 'PEP.csv', 'pypl.csv', 'QCOM.csv', 'QQQ.csv',
            'SBUX.csv', 'T.csv', 'TGT.csv', 'TM.csv', 'TSLA.csv', 'TSM.csv', 'USO.csv', 'V.csv', 'WFC.csv',
            'WMT.csv', 'XLF.csv']

# Test csvs = 25
names_25 = ['AAPL.csv', 'ABBV.csv', 'AMZN.csv', 'BABA.csv', 'C.csv', 'COST.csv', 'CVX.csv', 'DIS.csv', 'BRK-B.csv',
            'GE.csv',
            'INTC.csv', 'MSFT.csv', 'nvda.csv', 'pypl.csv', 'QQQ.csv', 'SBUX.csv', 'T.csv', 'TSLA.csv', 'WFC.csv',
            'KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']

# Test csvs = 5
# names_5 = ['KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']
names_5 = ['AMD.csv']

if args.num_pred == 50:
    file_list = names_50
elif args.num_pred == 25:
    file_list = names_25
elif args.num_pred == 5:
    file_list = names_5
else:
    raise ValueError("Invalid --num_pred value. It must be 5, 25, or 50.")


mses_all = []
maes_all = []

for file_name in file_list:
    print(f"Processing {file_name}...")
    args.data_path = file_name
    file_path = os.path.join(args.root_path, file_name)
    df_raw = pd.read_csv(file_path).dropna()
    num_samples = len(df_raw) * 0.10

    if num_samples <= 400:
        args.seq_len = 50
        args.label_len = 20
        args.pred_len = 15
    else:
        args.seq_len = 336
        args.label_len = 168
        args.pred_len = 96

    mses = []
    maes = []

    for ii in range(args.itr):

        setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                        args.d_model, args.n_heads, args.e_layers, args.gpt_layers,
                                                                        args.d_ff, args.embed, ii)
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if args.freq == 0:
            args.freq = 'h'

        train_data, train_loader = data_provider(args, flag='train', df_raw=df_raw)
        vali_data, vali_loader = data_provider(args, flag='val', df_raw=df_raw)
        test_data, test_loader = data_provider(args, flag='test', df_raw=df_raw)

        if args.freq != 'h':
            args.freq = SEASONALITY_MAP[test_data.freq]
            print("freq = {}".format(args.freq))

        device = torch.device('cuda:0')

        time_now = time.time()
        train_steps = len(train_loader)

        if args.model == 'PatchTST':
            model = PatchTST(args, device)
            model.to(device)
        elif args.model == 'DLinear':
            model = DLinear(args, device)
            model.to(device)
        else:
            model = GPT4TS(args, device)
        # mse, mae = test(model, test_data, test_loader, args, device, ii)

        params = model.parameters()
        model_optim = torch.optim.Adam(params, lr=args.learning_rate)

        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        if args.loss_func == 'mse':
            criterion = nn.MSELoss()
        elif args.loss_func == 'smape':
            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()
                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
            criterion = SMAPE()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

        for epoch in range(args.train_epochs):

            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)

                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                outputs = model(batch_x, ii)

                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
            # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            if args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, args)
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
        print("------------------------------------")
        mse, mae = test(model, test_data, test_loader, args, device, ii)
        mses.append(mse)
        maes.append(mae)

    mses = np.array(mses)
    maes = np.array(maes)
    print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
    print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))

    mses_all.append(np.mean(mses))
    maes_all.append(np.mean(maes))

mses_all = np.array(mses_all)
maes_all = np.array(maes_all)

print('finish training...')
print("mse_mean_all = {:.4f}, mse_std_all = {:.4f}".format(np.mean(mses_all), np.std(mses_all)))
print("mae_mean_all = {:.4f}, mae_std_all = {:.4f}".format(np.mean(maes_all), np.std(maes_all)))



# --root_path
# ./datasets/full_history/
# --num_pred
# 5
# --model_id
# fin
# --data
# fin
# --seq_len
# 336
# --label_len
# 168
# --pred_len
# 96
# --batch_size
# 400
# --lradj
# type4
# --learning_rate
# 0.0001
# --train_epochs
# 10
# --decay_fac
# 0.5
# --d_model
# 768
# --n_heads
# 4
# --d_ff
# 768
# --dropout
# 0.3
# --enc_in
# 7
# --c_out
# 7
# --freq
# 0
# --patch_size
# 16
# --stride
# 8
# --percent
# 100
# --gpt_layer
# 6
# --itr
# 3
# --model
# GPT4TS
# --tmax
# 20
# --cos
# 1
# --is_gpt
# 1
# --target
# Close