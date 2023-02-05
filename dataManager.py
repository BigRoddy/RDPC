import numpy as np
import pandas as pd

from dataLoader import DataLoader

def get_type_list(feature_number):
    """
    :param feature_number: an int indicates the number of features
    :return: a list of features n
    """
    if feature_number == 1:
        type_list = ["Close"]
    elif feature_number == 2:
        type_list = ["Close", "Volume"]
        raise NotImplementedError("the feature volume is not supported currently")
    elif feature_number == 3:
        type_list = ["Close", "High", "Low"]
    elif feature_number == 4:
        type_list = ["Close", "High", "Low", "Open"]
    elif feature_number == 5:
        type_list = ["Close", "High", "Low", "Open", "Adj Close"]
    else:
        raise ValueError("feature number could not be %s" % feature_number)
    return type_list

class Experience:
    def __init__(self, state_index):
        self.state_index = int(state_index)

class ReplayBuffer:
    def __init__(self, start_index, end_index, batch_size, is_permed, sample_bias=1.0):
        """
        :param start_index: start index of the training set on the global data matrices
        :param end_index: end index of the training set on the global data matrices
        """
        self.__experiences = [Experience(i) for i in range(start_index, end_index)]
        self.__is_permed = is_permed
        self.__batch_size = batch_size
        self.__sample_bias = sample_bias
        print("buffer_bias is %f" % sample_bias)

    def __sample(self, start, end, bias):
        """
        @:param end: is excluded
        @:param bias: value in (0, 1)
        """
        ran = np.random.geometric(bias)
        while ran > end - start:
            ran = np.random.geometric(bias)
        result = end - ran
        return result

    def next_experience_batch(self):
        # First get a start point randomly
        batch = []
        if self.__is_permed:
            for i in range(self.__batch_size):
                batch.append(self.__experiences[self.__sample(self.__experiences[0].state_index,
                                                              self.__experiences[-1].state_index,
                                                              self.__sample_bias)])
        else:
            batch_start = self.__sample(0, len(self.__experiences) - self.__batch_size,
                                        self.__sample_bias)
            batch = self.__experiences[batch_start:batch_start+self.__batch_size]
        return batch


class DataMatrices:
    def __init__(self, start_date, end_date, batch_size=50, volume_average_days=30, buffer_bias_ratio=0,
                 market="acl18", stock_filter=10, window_size=50, feature_number=4, test_portion=0.15, is_permed=False):
        """
        :param start: start date
        :param end: end date
        :param stock_filter: number of stocks that would be selected
        :param window_size: periods of input data
        :param train_portion: portion of training set
        :param is_permed: if False, the sample inside a mini-batch is in order
        :param test_portion: portion of test set
        """
        self.__stock_num = stock_filter

        type_list = get_type_list(feature_number)
        data_loader = DataLoader(stock_filter, market, start_date, end_date, volume_average_days, test_portion)
        self.__global_data = data_loader.get_global_data(type_list)

        # portfolio vector memory, [time, assets]
        self.__PVM = pd.DataFrame(index=range(self.__global_data.shape[2]),
                                  columns=range(self.__global_data.shape[1]))
        self.__PVM = self.__PVM.fillna(1.0 / self.__stock_num)

        self._window_size = window_size
        self._num_periods = self.__global_data.shape[2]

        self.__divide_data(test_portion)

        self.__is_permed = is_permed
        self.__batch_size = batch_size
        self.__replay_buffer = ReplayBuffer(start_index=self._train_ind[0],
                                               end_index=self._train_ind[-1],
                                               sample_bias=buffer_bias_ratio,
                                               batch_size=self.__batch_size,
                                               is_permed=self.__is_permed)

        print("the number of training examples is %s"
                     ", of test examples is %s" % (self._num_train_samples, self._num_test_samples))
        print("the training set is from %s to %s" % (min(self._train_ind), max(self._train_ind)))
        print("the test set is from %s to %s" % (min(self._test_ind), max(self._test_ind)))

    
    def __divide_data(self, test_portion):
        train_portion = 1 - test_portion
        s = float(train_portion + test_portion)

        portions = np.array([train_portion]) / s
        portion_split = (portions * self._num_periods).astype(int)
        indices = np.arange(self._num_periods)
        self._train_ind, self._test_ind = np.split(indices, portion_split)
        
        self._train_ind = self._train_ind[:-(self._window_size + 1)]
        self._test_ind = self._test_ind[:-(self._window_size + 1)]

        # NOTE(zhengyao): change the logic here in order to fit both reversed and normal version
        self._train_ind = list(self._train_ind)
        self._num_train_samples = len(self._train_ind)
        self._num_test_samples = len(self._test_ind)

    # dim: [features,stocks,times]
    def get_submatrix(self, ind):
        return self.__global_data[:, :, ind:ind+self._window_size+1]
    
    def get_submatrix_test_online(self, ind_start,ind_end):
        return self.__global_data[:, :, ind_start:ind_end]

    def __pack_samples(self, indexs):
        indexs = np.array(indexs)
        last_w = self.__PVM.values[indexs-1, :]

        def setw(w):
            self.__PVM.iloc[indexs, :] = w

        M = [self.get_submatrix(index) for index in indexs]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}
    
    def __pack_samples_test_online(self, ind_start,ind_end):
        last_w = self.__PVM.values[ind_start-1:ind_start, :]
        M = [self.get_submatrix_test_online(ind_start,ind_end)]  #[1,4,11,2807]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, self._window_size:]/ M[:, 0, None, :, self._window_size-1:-1]
        return {"X": X, "y": y, "last_w": last_w}

##############################################################################
    def get_test_set(self):
        return self.__pack_samples(self._test_ind)
    
    def get_test_set_online(self,ind_start,ind_end):
        return self.__pack_samples_test_online(ind_start,ind_end)

    def next_batch(self):
        """
        @:return: the next batch of training sample. The sample is a dictionary
        with key "X"(input data); "y"(future relative price); "last_w" a numpy array
        with shape [batch_size, assets]; "w" a list of numpy arrays list length is
        batch_size
        """
        batch = self.__pack_samples([exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch
