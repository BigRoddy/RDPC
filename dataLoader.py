import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, stock_number, dataset, start_date, end_date, volume_average_days=30, test_portion=0.15):
        self.__dataset = dataset
        self.__start_date = start_date
        self.__end_date = end_date
        self.__stock_number = stock_number
        self.__test_portion = test_portion
        self.__volume_average_days = volume_average_days
    
    def load_data(self, dataset, start_date, end_date):
        data_path = './data/'+str(dataset)+'/raw'
        fnames = [fname for fname in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, fname))]
        self.__stock_names = fnames
        print(len(fnames), ' stocks loading...')

        data_dfs = []
        for fname in fnames:
            if dataset == 'sp500':
                df = pd.read_csv(os.path.join(data_path, fname),parse_dates=['date'])
                df.rename(columns={'date':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'},inplace=True)
            else:
                df = pd.read_csv(os.path.join(data_path, fname),parse_dates=['Date'])
            df=df.query('Date>=@start_date and Date<=@end_date')
            df=df.sort_values(by=['Date'])
            df=df.reset_index(drop=True)
            data_dfs.append(df)
        return data_dfs
    
    def select_stocks(self, start_date, end_date, stock_number, test_portion, volume_average_days):

        print("select coins offline from %s to %s" % (start_date,end_date))
        
        sum_volumes=[]
        for ind in range(len(self.data_dfs)):
            sum_volume=self.data_dfs[ind].loc[len(self.data_dfs[ind].index)*(1-test_portion)-volume_average_days:len(self.data_dfs[ind].index)*(1-test_portion)].sum()['Volume']
            sum_volumes.append(sum_volume)

        selec_df = pd.DataFrame()
        selec_df['index']=range(len(self.data_dfs))
        selec_df['sum_volume']=sum_volumes

        selec_indexs = selec_df.sort_values(by=['sum_volume'],ascending=False).head(stock_number)['index'].tolist()
        if len(selec_indexs)!=stock_number:
            print("the sqlite error happend")

        stocks = []
        for index in selec_indexs:
            stocks.append(self.__stock_names[index][:-4])
        print("Selected stocks are: "+str(stocks))
        return selec_indexs

    def get_global_data(self, features=('close',)):
        """
        :param features: tuple or list of the feature names
        :return a panel, [feature, coin, time]
        """

        self.data_dfs = self.load_data(self.__dataset, self.__start_date, self.__end_date)
        self.__stock_indexs = self.select_stocks(self.__start_date, self.__end_date, self.__stock_number, self.__test_portion, self.__volume_average_days)
 
        if len(self.__stock_indexs)!=self.__stock_number:
            raise ValueError("the length of selected coins %d is not equal to expected %d"
                             % (len(self.__stock_indexs),self.__stock_number))
        
        selected_dfs=[]
        for index in self.__stock_indexs:
            selected_dfs.append(self.data_dfs[index])

        print("feature type list is %s" % str(features))
        for index in range(len(selected_dfs)):
            selected_dfs[index] = selected_dfs[index][features]
            selected_dfs[index] = selected_dfs[index].fillna(axis=1, method="bfill").fillna(axis=1, method="ffill")
            selected_dfs[index] = np.array(selected_dfs[index])
        selected_dfs=np.array(selected_dfs).transpose(2,0,1)

        return selected_dfs