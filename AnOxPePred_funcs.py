import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, tensorflow as tf, seaborn as sns
from Bio import pairwise2
import os, time
from sklearn.metrics import *
import matplotlib.pyplot as plt

from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import backend as K

"""
Data pre-process
"""

def homology_reduction(a_df, a_cutoff=.9):
    """
    Hardcoded function: Designed to do homology reduction on each individual
    data type in AO_df.
    """
    # Take out data with the same target, and do homology reduction with a defined threshold (90% default)
    _1df = hmlg_reduc(a_df[(a_df.FRS == 1) & (a_df.Chelator == 0)], cutoff=a_cutoff)
    _2df = hmlg_reduc(a_df[(a_df.FRS == 0) & (a_df.Chelator == 1)], cutoff=a_cutoff)
    _3df = hmlg_reduc(a_df[(a_df.FRS == 0) & (a_df.Chelator == 0)], cutoff=a_cutoff)
    _4df = hmlg_reduc(a_df[(a_df.FRS == 1) & (a_df.Chelator == 1)], cutoff=a_cutoff)

    # Concatenate the DataFrames back together
    return pd.concat([_1df, _2df, _3df, _4df]).reset_index(drop=True)


def hmlg_reduc(a_df, cutoff):
    """
    Homology reduction with a specified cutoff.
    """
    a_columns = a_df.columns
    a_list = np.array(a_df).tolist()
    b_list = []
    count=0
    for a_num, a_val in enumerate(a_list):
        count+=1

        if b_list == []:
            b_list.append(a_val)

        else:
            for b_num, b_val in enumerate(b_list):

                ident = calc_ident(a_val[0], b_val[0])

                if ident >= cutoff:
                    b_list[b_num] = b_val
                    break

            else:
                b_list.append(a_val)

    b_df = pd.DataFrame(data=b_list, columns=a_columns)

    return(b_df)


def calc_ident(a_seq, b_seq):
    """
    Calculates identity between two sequences by using a simple aligment
    algorithm. 1 for identical, 0 for not. -10 (this number ensures minimum gaps)
    for gap openings and extensions. Does not count end gaps.
    """
    match = pairwise2.align.globalms(a_seq, b_seq, 1, 0, -10, -10, penalize_end_gaps=False, score_only=False)[0]
    return match[2]/match[4]


def visualize_data(a_df, idx_name):
    """
    Hardcoded function. Visualizes the number of each type of data in AO_df.
    """
    tmp1 = a_df[(a_df.FRS == 1) & (a_df.Chelator == 0)].shape[0]
    tmp2 = a_df[(a_df.FRS == 0) & (a_df.Chelator == 1)].shape[0]
    tmp3 = a_df[(a_df.FRS == 0) & (a_df.Chelator == 0)].shape[0]
    tmp4 = a_df[(a_df.FRS == 1) & (a_df.Chelator == 1)].shape[0]

    return pd.DataFrame([[tmp1,tmp2,tmp4,tmp3,a_df.shape[0]]], index=[idx_name], columns=['FRS','CHEL','FRS/CHEL','NON-AO','TOTAL'])


"""
Peptide generator
"""

def pep_generator(a_df, fsa_file, a_nr):
    """
    Semi hardcoded function. Chops up a protein sequence or multiple from a
    fasta file into peptides with lengths corresponding to a database.
    """
    from itertools import cycle

    # Hardcoded part. Extracts an estimate of how many random peptides for each length to create.
    unique, counts = np.unique(np.array(a_df.query("FRS==1").Sequence.apply(len)), return_counts=True)
    pep_dict = dict(zip(unique, counts))

    # Reads in the aa sequence
    ran_pep_cycle = cycle(list(convert_fsa_list(fsa_file))[1][0])

    # Creates the peptides
    pep_list = []
    for a_dict in pep_dict:
        pep_list += [[''.join([next(ran_pep_cycle) for _ in range(a_dict)]),'-', 0,0] for _ in range(0, a_nr)]

    return pd.DataFrame(pep_list, columns=['Sequence','Source', 'FRS', 'Chelator'])


def convert_fsa_list(fasta_file):
    """
    Reads a fasta file with multiple entries
    """
    with open(fasta_file, 'r') as handle:
        yield [next(handle)[1:].strip()]
        seq = []
        for line in handle:
            if line.startswith('>'):
                yield [''.join(seq)]
                seq = []
                yield [line[1:].strip()]

            else:
                seq += [line.strip()]

        yield [''.join(seq)]


def reduce_df(a_df, b_df, max_len=1):
    """
    Hardcoded function: Removes duplicates and peptides present in another database
    """

    # Remove random negatives which are present in a_df and duplicates
    the_df = pd.concat([a_df, b_df, b_df]).drop_duplicates(subset=['Sequence'], keep=False)


    # Take only max_len - makes sure it is the same number, even after homology reduction and deletion of redundant.
    n_df = pd.DataFrame()
    for i in range(2, 31):
        tmp_df = the_df.loc[the_df.Sequence.apply(len) == i]
        n_df = pd.concat([n_df, tmp_df.iloc[:max_len]], sort=True)

    return n_df.reset_index(drop=True)


"""
Data Partitioning
"""

def homology_partition(a_s, ident, parts):
    """
    Identity based partitions. First, calculates an identity matrix; second,
    creates clusters of other sequences which are within the identity
    threshold (ident) of eachother; lastly, it groups these clusters into a
    specific number of partitions (parts).
    """
    sort_s = a_s.copy()

    # Calculate identity matrix
    a_list = []
    for seq in sort_s:
        tmp_s = sort_s.apply(lambda x: calc_ident(x, seq)).sort_values(ascending=False)
        a_list.append(tmp_s[tmp_s >= ident].index.values)

    c_list = a_list.copy()
    c_list = [list(i) for i in c_list]

    # Make clusters which share over a threshold identity
    while len(sum(c_list,[])) != len(set(sum(c_list,[]))):
        for num1, pdbs1 in enumerate(c_list):
            for num2, pdbs2 in enumerate(c_list):
                if not set(pdbs1).isdisjoint(pdbs2) and num1 != num2:
                    c_list[num1] += pdbs2
                    c_list[num1] = list(set(c_list[num1]))
                    c_list.pop(num2)
                    break

    # Put these clusters into a set number of partitions (parts)
    this_tmp = group_partition(c_list, part_num=parts)
    tmptmp = a_s.copy()

    return [[num for num, val in enumerate(tmptmp.index.values) if val in part] for part in this_tmp], len(c_list)


def group_partition(hom_list, part_num):
    """
    Hardcoded function: Groups a list of clusters into a number of
    partitions while keeping symetri between the groups. (Appr. same number of
    FRS and Chelators in each partition).
    """
    import itertools
    hom_list = sorted(hom_list, key=len, reverse=True)

    max_val = len(sum(hom_list,[]))/part_num

    idx_list = list(range(part_num))
    a_idx = itertools.cycle(idx_list)
    count = 0
    par_list = np.array(part_num*[[]]).tolist()
    for num, val in enumerate(hom_list):

        t_idx = next(a_idx)
        if len(par_list[t_idx]) < max_val:
            par_list[t_idx] += val

        else:
            hom_list.append(val)

    return par_list


def hc_part_visualizer(a_df, hom_list, hom_name):
    """
    Hardcoded function: Visualizes the type of data within each partition.
    """
    r_df = pd.DataFrame()
    for num, hom in enumerate(hom_list):
        tmp2_df = pd.DataFrame()
        for i in range(5):
            tmp_df = a_df.iloc[hom[i]]
            tmp3_df = pd.DataFrame([[len(tmp_df), tmp_df.FRS.sum(), tmp_df.Chelator.sum()]], columns=['Sum_P'+str(i+1), 'frs_P'+str(i+1), 'chel_P'+str(i+1)])
            tmp2_df = pd.concat([tmp2_df, tmp3_df], axis=1)

        tmp2_df.index = [hom_name[num]]
        r_df = pd.concat([r_df, tmp2_df])

    return r_df.reindex(sorted(r_df.columns), axis=1)


def gini(x):
    """
    Calculates the gini coefficient for an array.
    Credit goes here:
    https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
    """
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


"""
Data representation
"""

def split_into_parts(a_data, test_part, val_part, embed_file):
    """
    Hardcoded function: Splits data into test, val and train sets based on
    given partitions. Additionally, turns augments x_data into numerical values
    based on used embedding.
    """

    x_test = data_augmentation(a_data.query("partition==[@test_part]").Sequence.to_numpy(),embed_file)
    y_test = a_data.query("partition==[@test_part]").loc[:,['FRS','Chelator']].values
    x_val = data_augmentation(a_data.query("partition==[@val_part]").Sequence.to_numpy(),embed_file)
    y_val = a_data.query("partition==[@val_part]").loc[:,['FRS','Chelator']].values
    x_train = data_augmentation(a_data.query("partition!=[@test_part, @val_part]").Sequence.to_numpy(),embed_file)
    y_train = a_data.query("partition!=[@test_part, @val_part]").loc[:,['FRS','Chelator']].values


    return x_test, y_test, x_val, y_val, x_train, y_train


def data_augmentation(a_array, embed_file):
    """
    Applies embedding sequences. 30 is the hardcoded max length of sequences.
    Every sequence is padded up to this length.
    """
    e_dic = embed_read(embed_file, normalize=0)
    x_gen = seq_padding(a_array, 30)

    x_data = embed_appl(x_gen, e_dic)

    return x_data


def embed_read(Embedding_file, normalize=0):
    """
    Turns the embedding file into a dictionary for easy use. Additionally, the
    option to normalize the embeddings are also there.
    """
    from sklearn import preprocessing

    embedding_dic = {}
    with open(Embedding_file, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                line = line.split()
                embedding_dic[line[0]] = [float(i) for i in line[1:]]

    if normalize == 1:
        x = pd.DataFrame.from_dict(embedding_dic).T.values #returns a numpy array
        #x = x[:-1]
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.05,0.95))
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        df.index = pd.DataFrame.from_dict(embedding_dic).T.index #[:-1]
        embedding_dic = df.T.to_dict('list')
        #embedding_dic['X'] = [0 for i in embedding_dic['A']]

    return embedding_dic


def embed_appl(a_array, embed_dic):
    """
    Applies the score on a sequence based on embedding
    """
    return np.array([np.array([embed_dic[j] for j in i]) for i in a_array])


def seq_padding(a_array, length='max'):
    """
    Pads sequences to a specified length or the longest sequence. Additionally,
    centers the sequence in the middle (padding happens symetrically on
    both sides).
    """
    if length == 'max':
        max_l = len(max(a_array, key=len))
    else:
        max_l = int(length)

    for num, val in enumerate(a_array):
        if len(val) <= max_l:
            num_x = (max_l - len(val))
            # Put X's on both sides of the sequence
            yield ('X' * int(np.ceil(num_x / 2))) + val + ('X' * int(np.floor(num_x / 2)))


"""
Network
"""

def create_AnOxPePred_v1(hps):
    """
    Creates the model 'AnOxPePred_v1' and initializes it.
    """
    class AnOxPePred_v1(Model):
        def __init__(self, hps):
            super(AnOxPePred_v1, self).__init__()
            self.conv1 = Conv1D(filters=128, kernel_size=3, strides=1, activation='elu', padding='same')
            self.bn1a = BatchNormalization()
            self.maxpool1 = AveragePooling1D(pool_size=3, strides=3)
            self.dropout1 = Dropout(0.1)

            self.conv2 = Conv1D(filters=128, kernel_size=1, strides=1, activation='elu', padding='same')
            self.bn2a = BatchNormalization()
            self.maxpool2 = AveragePooling1D(pool_size=10, strides=1)
            self.dropout2 = Dropout(0.1)

            self.flatten = Flatten()
            self.d1 = Dense(256, activation='elu')
            self.dropout3 = Dropout(0.4)
            self.d2 = Dense(hps['y_out'], activation='sigmoid')#'sigmoid'

        def call(self, x):

            x1 = self.dropout1(self.maxpool1(self.bn1a(self.conv1(x))))
            x2 = self.dropout2(self.maxpool2(self.bn2a(self.conv2(x1))))
            x1 = self.flatten(x1)
            x2 = self.flatten(x2)
            x4 = tf.concat([x1, x2], 1)
            x5 = self.dropout3(self.d1(x4))
            return self.d2(x5)

    model = AnOxPePred_v1(hps)
    model.compile(loss=[focal_loss()], metrics=['accuracy'],
              optimizer=tf.keras.optimizers.Adam(lr = 0.00002, decay=0.00002, amsgrad=1, clipnorm=1, clipvalue=0.5))

    return model
#lr = 0.00002, decay=0.00001, amsgrad=1, clipnorm=1 200 epoch 20 early stopping  gamma=7 alpha=0.5


def focal_loss(gamma=7, alpha=.5, mask_val=-1):
    """
    Loss function that takes into account an imbalanced dataset.
    This function was written by Martin Closter Jespersen.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_mask = tf.cast(tf.where(tf.not_equal(y_true, mask_val), tf.ones_like(y_true), tf.zeros_like(y_true)), tf.bool)
        y_true = tf.boolean_mask(y_true, y_mask)
        y_pred = tf.boolean_mask(y_pred, y_mask)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
        return loss/tf.reduce_sum(y_true)
        # return loss
    return focal_loss_fixed


"""
Metrics
"""

def calc_metrics(y_label, y_pred, idx='metrics'):
    """
    Hardcoded function: Finds optimal threshold and calculates metrics with
    0.5 threshold and optimal threshold for easy comparison.
    """
    FRS_t = find_opt_thres(y_label[:,0], y_pred[:,0])
    CHEL_t = find_opt_thres(y_label[:,1], y_pred[:,1])

    t_05 = pd.concat([calc_metrics_single(y_label[:,0], y_pred[:,0], 0.5, idx,'0.5'), calc_metrics_single(y_label[:,1], y_pred[:,1], 0.5, idx,'0.5')], axis=1)
    t_custom = pd.concat([calc_metrics_single(y_label[:,0], y_pred[:,0], FRS_t, idx,'custom'), calc_metrics_single(y_label[:,1], y_pred[:,1], CHEL_t, idx, 'custom')], axis=1)
    tmp_df = pd.concat([t_05,t_custom], axis=1)

    tmp_df['FRS_Threshold'] = FRS_t
    tmp_df['CHEL_Threshold'] = CHEL_t

    return tmp_df


def calc_metrics_single(y_label, y_pred, thres, idx='metrics', nn=''):
    """
    Calculate metrics with threshold.
    """
    return ParaNet_Metrics(y_label, y_pred).calc_summary(threshold=thres, idx=idx, nn=nn)


class ParaNet_Metrics:
    """
    A class which can calculate a variety of metrics given
    a set of y_labels and y_preds.
    """
    def __init__(self, y_label, y_pred):
        self.y_pred = y_pred
        self.y_label = y_label
        self.record_size = self.y_label.shape[0]

    def calc_summary(self, threshold=0.5, idx='metrics', nn=''):
        return pd.DataFrame(data=[[ParaNet_Metrics.calc_auc(self),
        ParaNet_Metrics.calc_f1_score(self, threshold),
        ParaNet_Metrics.calc_mcc(self, threshold)]],
        columns=[nn+'_AUC', nn+'_F1', nn+'_MCC'], index=[idx])

    def calc_auc(self):
        return roc_auc_score(self.y_label, self.y_pred)

    def calc_f1_score(self, threshold=0.5):
        y_binary = np.array([np.where(np.array(x) >= threshold, 1, 0) for x in self.y_pred])

        return f1_score(self.y_label, y_binary)

    def calc_accuracy(self, threshold=0.5):
        y_binary = np.where(self.y_pred >= threshold, 1, 0)
        return accuracy_score(self.y_label.reshape(-1), y_binary.reshape(-1))

    def calc_mcc(self, threshold=0.5):
        y_binary = np.array([np.where(np.array(x) >= threshold, 1, 0) for x in self.y_pred])

        return matthews_corrcoef(self.y_label, y_binary)


def find_opt_thres(y_label, y_pred):
    """
    Calculates the optimal threshold based on MCC.
    """
    return pd.DataFrame([[ParaNet_Metrics(y_label, y_pred).calc_mcc(i), i] for i in np.arange(0,1,0.01)]).sort_values(by=0, ascending=False).iloc[:1,1].mean()


"""
kNN Predictor
"""

def kNN_pred(a_series, a_df, k=5):
    """
    kNN predictor: Predicts a sequences activity based on the average activity
    of the k most similar sequences (in terms of sequence identity).
    """

    b_df = a_df.copy()
    b_df['Ident'] = b_df.query("partition!=@a_series.partition").Sequence.apply(lambda x: calc_ident(x, a_series.Sequence))
    d_sorted = b_df.sort_values(by='Ident',ascending=False)[:k]

    return pd.Series([d_sorted.FRS.mean(), d_sorted.Chelator.mean()])


"""
AnOxPePred predictor
"""

def get_fsa_file(fsa_file):
    """
    Turns fasta file into list.
    """
    with open(fsa_file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line.startswith('>'):
                AC = [line[1:]]
            elif line != '':
                yield AC + [line]


"""
Analysis functions
"""

def re_bin(x, y, new_bins):
    """
    Re-bins data according to new_bins
    """
    new_y = np.array(list([0]*len(new_bins)))
    for x_, y_ in zip(x, y):
        for num, ele in enumerate(new_bins):
            if x_ in ele:
                new_y[num] += int(y_)

    return new_y


def change_width(ax, new_value) :
    """
    No credit goes to me for this function. Changes the width of bars in the
    bar plot.
    """
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def if_pval(a_array, target_array, a_pval=0.05):
    """
    Returns a boolean array for whether the a_array elements are significant
    different from the target_array elements.
    """
    from statsmodels.stats.proportion import proportions_ztest
    bool_result = []
    for num, val in enumerate(target_array):
        count = int(a_array[num])
        nobs = a_array.sum()
        val = float(target_array[num])
        stat, pval = proportions_ztest(count, nobs, value=val, alternative='two-sided')

        #print('{0:.2E}'.format(Decimal(pval)))
        if pval < a_pval:

            bool_result.append(False)
        else:
            bool_result.append(True)

    return bool_result
