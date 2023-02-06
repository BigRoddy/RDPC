import torch
import torch.nn as nn
import time
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

class Test_Loss(nn.Module):
    def __init__(self, commission_ratio,interest_rate,gamma=0.1,beta=0.1, size_average=True):
        super(Test_Loss, self).__init__()
        self.gamma = gamma  #variance penalty
        self.beta = beta
        self.size_average = size_average
        self.commission_ratio=commission_ratio
        self.interest_rate=interest_rate

    def forward(self, w, y):               # w:[128,10,1,12] y(128,10,11,4)
        close_price = y[:,:,:,0:1].cuda()    #   [128,10,11,1]
        close_price = torch.cat([torch.ones(close_price.size()[0],close_price.size()[1],1,1).cuda(),close_price],2).cuda()       #[128,10,11,1]cat[128,10,1,1]->[128,10,12,1]
        reward = torch.matmul(w,close_price)   #  [128,10,1,12] * [128,10,12,1] ->[128,10,1,1]
        close_price = close_price.view(close_price.size()[0],close_price.size()[1],close_price.size()[3],close_price.size()[2])  #[128,10,12,1] -> [128,10,1,12]
##############################################################################
        element_reward = w*close_price
        interest = torch.zeros(element_reward.size(),dtype = torch.float).cuda()
        interest[element_reward<0] = element_reward[element_reward<0]
#        print("interest:",interest.size(),interest,'\r\n')
        interest = torch.sum(interest,3).unsqueeze(3)*self.interest_rate  #[128,10,1,1]
##############################################################################
        future_omega = w*close_price/reward    #[128,10,1,12]*[128,10,1,12]/[128,10,1,1]
        wt=future_omega[:,:-1]                 #[128, 9,1,12]   
        wt1=w[:,1:]                            #[128, 9,1,12]
        pure_pc=1-torch.sum(torch.abs(wt-wt1),-1)*self.commission_ratio     #[128,9,1]
        pure_pc=pure_pc.cuda()
        pure_pc=torch.cat([torch.ones([pure_pc.size()[0],1,1]).cuda(),pure_pc],1)      #[128,1,1] cat  [128,9,1] ->[128,10,1]        
        pure_pc=pure_pc.view(pure_pc.size()[0],pure_pc.size()[1],1,pure_pc.size()[2])  #[128,10,1] ->[128,10,1,1]          
        cost_penalty = torch.sum(torch.abs(wt-wt1),-1)                                 #[128, 9, 1]      
################## Deduct transaction fee ##################
        reward = reward*pure_pc                                                        #[128,10,1,1]*[128,10,1,1]  test: [1,2808-31,1,1]
################## Deduct loan interest ####################
        reward= reward+interest
        if not self.size_average:
            tst_pc_array=reward.squeeze()
            sr_reward=tst_pc_array-1
            SR=sr_reward.mean()/sr_reward.std()
            SN=torch.prod(reward,1) #[1,1,1,1]
            SN=SN.squeeze() #
            St_v=[]
            St=1.            
            MDD=max_drawdown(tst_pc_array)
            for k in range(reward.size()[1]):  #2808-31
                St*=reward[0,k,0,0]
                St_v.append(St.item())
            CR=SN/MDD            
            TO=cost_penalty.mean()
##############################################
        portfolio_value=torch.prod(reward,1)     #[128,1,1]
        batch_loss=-torch.log(portfolio_value)   #[128,1,1]

        if self.size_average:
            loss = batch_loss.mean() 
            return loss, portfolio_value.mean()
        else:
            loss = batch_loss.mean() 
            return loss, portfolio_value[0][0][0],SR,CR,St_v,tst_pc_array,TO,MDD

class SimpleLossCompute_tst:
    "A simple loss compute and train function."
    def __init__(self,  criterion, opt=None):
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y):
        if self.opt is not None:
            loss, portfolio_value= self.criterion(x,y)
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
            return loss, portfolio_value
        else:
            loss, portfolio_value,SR,CR,St_v,tst_pc_array,TO,MDD = self.criterion(x,y)     
            return loss, portfolio_value,SR,CR,St_v,tst_pc_array,TO,MDD

def test_online(DM,x_window_size,model,evaluate_loss_compute,local_context_length):
    tst_batch=DM.get_test_set_online(DM._test_ind[0], DM._test_ind[-1])
    tst_batch_input = tst_batch["X"]         
    tst_batch_y = tst_batch["y"]              
    tst_batch_last_w = tst_batch["last_w"]

    tst_previous_w=torch.tensor(tst_batch_last_w,dtype=torch.float).cuda()
    tst_previous_w=torch.unsqueeze(tst_previous_w,1)  

    tst_batch_input=tst_batch_input.transpose((1,0,2,3))
    tst_batch_input=tst_batch_input.transpose((0,1,3,2))

    long_term_tst_src=torch.tensor(tst_batch_input,dtype=torch.float).cuda()      
#########################################################################################
    tst_src_mask = (torch.ones(long_term_tst_src.size()[1],1,x_window_size)==1)   

    long_term_tst_currt_price=long_term_tst_src.permute((3,1,2,0)) 
    long_term_tst_currt_price=long_term_tst_currt_price[:,:,x_window_size-1:,:]   
###############################################################################################    
    tst_trg_mask = make_std_mask(long_term_tst_currt_price[:,:,0:1,:],long_term_tst_src.size()[1])
   

    tst_batch_y=tst_batch_y.transpose((0,3,2,1))  
    tst_trg_y=torch.tensor(tst_batch_y,dtype=torch.float).cuda()
    tst_long_term_w=[]  
    tst_y_window_size=len(DM._test_ind)-x_window_size-1-1
    for j in range(tst_y_window_size+1): #0-9
        tst_src=long_term_tst_src[:,:,j:j+x_window_size,:]
        tst_currt_price=long_term_tst_currt_price[:,:,j:j+1,:]
        if(local_context_length>1):
            padding_price=long_term_tst_src[:,:,j+x_window_size-1-local_context_length*2+2:j+x_window_size-1,:]
            padding_price=padding_price.permute((3,1,2,0))  #[4, 1, 2, 11] ->[11,1,2,4]
        else:
            padding_price=None
        out = model.forward(tst_src, tst_currt_price, tst_previous_w,  #[109,1,11]   [109, 11, 31, 3]) torch.Size([109, 11, 3]
                        tst_src_mask, tst_trg_mask, padding_price)
        if(j==0):
            tst_long_term_w=out.unsqueeze(0)  #[1,109,1,12] 
        else:
            tst_long_term_w=torch.cat([tst_long_term_w,out.unsqueeze(0)],0)
        out=out[:,:,1:]  #remove cash #[109,1,11]
        tst_previous_w=out
    tst_long_term_w=tst_long_term_w.permute(1,0,2,3) ##[10,128,1,12]->#[128,10,1,12]
    tst_loss, tst_portfolio_value, SR, CR, St_v,tst_pc_array,TO,MDD=evaluate_loss_compute(tst_long_term_w,tst_trg_y)  
    return tst_loss, tst_portfolio_value, SR, CR, St_v,tst_pc_array,TO,MDD


def test_net(DM, total_step, output_step, x_window_size, local_context_length, model, loss_compute, evaluate_loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    max_tst_portfolio_value=0

    for i in range(total_step):
        with torch.no_grad():
            if(i % output_step == 0):
                model.eval()
                tst_loss, tst_portfolio_value, SR, CR, St_v,tst_pc_array,TO,MDD = test_online(DM,x_window_size, model, evaluate_loss_compute, local_context_length)                                      
                elapsed = time.time() - start
                print("Test: %d Loss: %f| Portfolio_Value: %f | SR: %f | CR: %f | MDD: %f | TO: %f |testset per Sec: %f" %
                        (i, tst_loss.item(), tst_portfolio_value.item() ,SR.item(), CR.item(), MDD.item(), TO.item(), 1/elapsed))
                start = time.time()

                if(tst_portfolio_value>max_tst_portfolio_value):
                    max_tst_portfolio_value=tst_portfolio_value
                    log_SR=SR
                    log_CR=CR
                    log_MDD=MDD
                    log_St_v=St_v
                    log_tst_pc_array=tst_pc_array
    return max_tst_portfolio_value, log_SR, log_CR, log_St_v, log_tst_pc_array,log_MDD,TO
