import numpy as np
import pandas as pd
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse

from dataManager import DataMatrices
from model import make_model, NoamOpt
from train import SimpleLossCompute, train_net, Batch_Loss
from test import SimpleLossCompute_tst, test_net, Test_Loss


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--total_step', type=int, default=30000)
parser.add_argument('--window_size', type=int, default=31)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--stock_num', type=int, default=11)
parser.add_argument('--feature_number', type=int, default=4)
parser.add_argument('--output_step', type=int, default=500)
parser.add_argument('--model_index', type=int, default=71)
parser.add_argument('--multihead_num', type=int, default=2)
parser.add_argument('--local_context_length', type=int, default=5)
parser.add_argument('--model_dim', type=int, default=12)

parser.add_argument('--test_portion', type=float, default=0.08)
parser.add_argument('--trading_consumption', type=float, default=0.0025)
parser.add_argument('--variance_penalty', type=float, default=0.1)
parser.add_argument('--cost_penalty', type=float, default=0)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=5e-8)
parser.add_argument('--daily_interest_rate', type=float, default=0.001)


parser.add_argument('--start', type=str, default = "2015-06-01")
parser.add_argument('--end', type=str, default = "2019-12-31")
parser.add_argument('--market', type=str, default = "csi300")

# parser.add_argument('--start', type=str, default = "2013-02-08")
# parser.add_argument('--end', type=str, default = "2018-02-07")
# parser.add_argument('--market', type=str, default = "sp500")

parser.add_argument('--log_dir', type=str, default = "./log")
parser.add_argument('--model_dir', type=str, default = "./model")

args = parser.parse_args()

SEED=0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)

Data=DataMatrices(start_date=args.start,end_date=args.end,
             batch_size=args.batch_size,
             volume_average_days=30,
             buffer_bias_ratio=5e-5,
             market=args.market,
             stock_filter=args.stock_num,
             window_size=args.window_size,
             feature_number=args.feature_number,
             test_portion=args.test_portion,
             is_permed=False)


#################set learning rate###################
lr_model_sz=5120
factor=args.learning_rate
warmup=0

total_step=args.total_step
window_size=args.window_size

batch_size=args.batch_size
stock_num=args.stock_num
feature_number=args.feature_number
trading_consumption=args.trading_consumption
variance_penalty=args.variance_penalty
cost_penalty=args.cost_penalty
output_step=args.output_step
local_context_length=args.local_context_length
model_dim=args.model_dim
weight_decay=args.weight_decay
interest_rate=args.daily_interest_rate/24/2

model = make_model(batch_size, stock_num, window_size, feature_number,
                         N=1, d_model_Encoder=args.multihead_num*model_dim,
                         d_model_Decoder=args.multihead_num*model_dim, 
                         d_ff_Encoder=args.multihead_num*model_dim, 
                         d_ff_Decoder=args.multihead_num*model_dim, 
                         h=args.multihead_num, 
                         dropout=0.01,
                         local_context_length=local_context_length)


model = model.cuda()
model_opt = NoamOpt(lr_model_sz, factor, warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9,weight_decay=weight_decay))

loss_compute = SimpleLossCompute( Batch_Loss(trading_consumption,interest_rate,variance_penalty,cost_penalty), model_opt)
evaluate_loss_compute = SimpleLossCompute( Batch_Loss(trading_consumption,interest_rate,variance_penalty,cost_penalty),  None)
test_loss_compute = SimpleLossCompute_tst( Test_Loss(trading_consumption,interest_rate,variance_penalty,cost_penalty,False),  None)


##########################train net####################################################
tst_loss, tst_portfolio_value = train_net(Data, total_step, output_step, window_size, local_context_length ,model, args.model_dir, args.model_index, loss_compute, evaluate_loss_compute)

model=torch.load(args.model_dir+'/'+ str(args.model_index)+'.pkl')

##########################test net#####################################################
tst_portfolio_value, SR, CR, St_v,tst_pc_array,MDD,TO=test_net(Data, 1, 1, window_size, local_context_length ,model, loss_compute, test_loss_compute)


csv_dir=args.log_dir+"/"+"train_summary.csv"
d={"net_dir":[args.model_index],
    "fAPV":[tst_portfolio_value.item()],
    "SR":[SR.item()],
    "CR":[CR.item()],
    "MDD":[MDD.item()],
    "TO":[TO.item()],
    "St_v":[''.join(str(e)+', ' for e in St_v)],
    "backtest_test_history":[''.join(str(e)+', ' for e in tst_pc_array.cpu().numpy())],   
    }
new_data_frame = pd.DataFrame(data=d).set_index("net_dir")
if os.path.isfile(csv_dir):
    dataframe = pd.read_csv(csv_dir).set_index("net_dir")
    dataframe = dataframe.append(new_data_frame) 
else:
    dataframe = new_data_frame
dataframe.to_csv(csv_dir)