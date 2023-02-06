import torch
import torch.nn as nn
from model import make_std_mask

def max_drawdown(pc_array):
    """calculate the max drawdown with the portfolio changes
    @:param pc_array: all the portfolio changes during a trading process
    @:return: max drawdown
    """
    portfolio_values = []
    drawdown_list = []
    max_benefit = 0
    for i in range(pc_array.shape[0]):
        if i > 0:
            portfolio_values.append(portfolio_values[i - 1] * pc_array[i])
        else:
            portfolio_values.append(pc_array[i])
        if portfolio_values[i] > max_benefit:
            max_benefit = portfolio_values[i]
            drawdown_list.append(0.0)
        else:
            drawdown_list.append(1.0 - portfolio_values[i] / max_benefit)
    return max(drawdown_list)

class Batch_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, gamma=0.1,beta=0.1, size_average=True):
        super(Batch_Loss, self).__init__()
        self.gamma = gamma  #variance penalty
        self.beta = beta    #cost_penalty
        self.size_average = size_average
        self.commission_ratio=commission_ratio
        self.interest_rate=interest_rate

    def forward(self, w, y):            # w:[128,1,12]   y:[128,11,4] 
        close_price=y[:,:,0:1].cuda()   #   [128,11,1]
        #future close prise (including cash)
        close_price=torch.cat([torch.ones(close_price.size()[0],1,1).cuda(),close_price],1).cuda()         #[128,11,1]cat[128,1,1]->[128,12,1]
        reward=torch.matmul(w,close_price)                                                                 #[128,1,1]
        close_price=close_price.view(close_price.size()[0],close_price.size()[2],close_price.size()[1])    #[128,1,12] 
###############################################################################################################
        element_reward=w*close_price
        interest=torch.zeros(element_reward.size(),dtype=torch.float).cuda()
        interest[element_reward<0]=element_reward[element_reward<0]
        interest=torch.sum(interest,2).unsqueeze(2)*self.interest_rate  #[128,1,1]
###############################################################################################################
        future_omega=element_reward/reward  #[128,1,12]           
        wt=future_omega[:-1]               #[127,1,12]
        wt1=w[1:]                          #[127,1,12]
        
        cost_penalty = torch.sum(torch.abs(wt-wt1),-1)
        pure_pc=1-cost_penalty*self.commission_ratio   #[128,1]
        pure_pc=pure_pc.cuda()
        pure_pc=torch.cat([torch.ones([1,1]).cuda(),pure_pc],0)
        pure_pc=pure_pc.view(pure_pc.size()[0],1,pure_pc.size()[1])       #[128,1,1]
        reward=reward*pure_pc    #reward=pv_vector
        reward=reward+interest
        portfolio_value=torch.prod(reward)
        batch_loss=-torch.log(reward)

        variance_penalty = ((torch.log(reward)-torch.log(reward).mean())**2).mean()
        loss = batch_loss.mean() + self.gamma*variance_penalty + self.beta*cost_penalty.mean()

        return loss, portfolio_value

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self,  criterion, opt=None):
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y):
        loss, portfolio_value= self.criterion(x,y)         
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss, portfolio_value

def train_one_step(DM,x_window_size,model,loss_compute,local_context_length):
    batch=DM.next_batch()
    batch_input = batch["X"]        #(128, 4, 11, 31)
    batch_y = batch["y"]            #(128, 4, 11)
    batch_last_w = batch["last_w"]  #(128, 11)
    batch_w = batch["setw"]     
#############################################################################
    previous_w=torch.tensor(batch_last_w,dtype=torch.float).cuda()
    previous_w=torch.unsqueeze(previous_w,1)                         #[128, 11] -> [128,1,11]
    batch_input=batch_input.transpose((1,0,2,3))
    batch_input=batch_input.transpose((0,1,3,2))
    src=torch.tensor(batch_input,dtype=torch.float).cuda()               #[4,128,31,11]
    price_series_mask = (torch.ones(src.size()[1],1,x_window_size)==1)   #[128, 1, 31] 
    currt_price=src.permute((3,1,2,0))                                   #[4,128,31,11]->[11,128,31,4]

    if(local_context_length>1):
        padding_price=currt_price[:,:,-(local_context_length)*2+1:-1,:] 
    else:
        padding_price=None
    currt_price=currt_price[:,:,-1:,:]                                    #[11,128,31,4]->[11,128,1,4]
    trg_mask = make_std_mask(currt_price,src.size()[1])
    batch_y=batch_y.transpose((0,2,1))                                    #[128, 4, 11] ->#[128,11,4]
    trg_y=torch.tensor(batch_y,dtype=torch.float).cuda()
    out = model.forward(src, currt_price, previous_w,
                        price_series_mask, trg_mask, padding_price)
    new_w=out[:,:,1:]  #remove cash
    new_w=new_w[:,0,:]  # #[109,1,11]->#[109,11]
    new_w=new_w.detach().cpu().numpy()
    batch_w(new_w)  
    
    loss, portfolio_value = loss_compute(out,trg_y)           
    return loss, portfolio_value

def test_batch(DM,x_window_size,model,evaluate_loss_compute,local_context_length):
    tst_batch=DM.get_test_set()
    tst_batch_input = tst_batch["X"]       #(128, 4, 11, 31)
    tst_batch_y = tst_batch["y"]
    tst_batch_last_w = tst_batch["last_w"]

    tst_previous_w=torch.tensor(tst_batch_last_w,dtype=torch.float).cuda()
    tst_previous_w=torch.unsqueeze(tst_previous_w,1)                    #[2426, 1, 11]
    tst_batch_input=tst_batch_input.transpose((1,0,2,3))
    tst_batch_input=tst_batch_input.transpose((0,1,3,2))
    tst_src=torch.tensor(tst_batch_input,dtype=torch.float).cuda()         
    tst_src_mask = (torch.ones(tst_src.size()[1],1,x_window_size)==1)   #[128, 1, 31]   
    tst_currt_price=tst_src.permute((3,1,2,0))                          #(4,128,31,11)->(11,128,31,3)
#############################################################################
    if(local_context_length>1):
        padding_price=tst_currt_price[:,:,-(local_context_length)*2+1:-1,:]  #(11,128,8,4)
    else:
        padding_price=None
#########################################################################

    tst_currt_price=tst_currt_price[:,:,-1:,:]   #(11,128,31,4)->(11,128,1,4)
    tst_trg_mask = make_std_mask(tst_currt_price,tst_src.size()[1])
    tst_batch_y=tst_batch_y.transpose((0,2,1))   #(128, 4, 11) ->(128,11,4)
    tst_trg_y=torch.tensor(tst_batch_y,dtype=torch.float).cuda()
###########################################################################################################
    tst_out = model.forward(tst_src, tst_currt_price, tst_previous_w, #[128,1,11]   [128, 11, 31, 4]) 
                    tst_src_mask, tst_trg_mask,padding_price)

    tst_loss, tst_portfolio_value=evaluate_loss_compute(tst_out,tst_trg_y) 
    return tst_loss, tst_portfolio_value

def train_net(DM, total_step, output_step, x_window_size, local_context_length, model, model_dir, model_index, loss_compute,evaluate_loss_compute, is_trn=True, evaluate=True):
    "Standard Training and Logging Function"
    ####每个epoch开始时previous_w=0
    max_tst_portfolio_value=1.02
    max_train_value = 0
    for i in range(total_step):
        if(is_trn):
            model.train()
            loss, portfolio_value=train_one_step(DM,x_window_size,model,loss_compute,local_context_length)

            if(i % output_step == 0):
                print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f \r\n" %
                    (i,loss.item(), portfolio_value.item()))

            if(portfolio_value>max_train_value+0.001):
                max_train_value = portfolio_value
                print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f \r\n" %
                    (i,loss.item(), portfolio_value.item()))
#########################################################tst########################################################     
        with torch.no_grad():
            if(evaluate):
                model.eval()
                tst_loss, tst_portfolio_value=test_batch(DM,x_window_size,model,evaluate_loss_compute,local_context_length)
                
                if(i % output_step == 0):
                    print("Test: %d Loss: %f| Portfolio_Value: %f \r\n" %
                        (i,tst_loss.item(), tst_portfolio_value.item() ))

                if(tst_portfolio_value>max_tst_portfolio_value+0.001):
                    print("Test: %d Loss: %f| Portfolio_Value: %f \r\n" %
                        (i,tst_loss.item(), tst_portfolio_value.item() ))
                    max_tst_portfolio_value=tst_portfolio_value
                    torch.save(model, model_dir+'/'+str(model_index)+".pkl")
                    print("save model!")
    return tst_loss, tst_portfolio_value