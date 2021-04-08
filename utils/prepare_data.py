# encoding: utf-8
# @author: xinhchen
# reference: zxding
# email: xinhchen2-c@my.cityu.edu.hk

import codecs
import random
import numpy as np
import scipy as sp
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pdb, time, logging, datetime


def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))


def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile1 = open(train_file_path, 'r', encoding='utf-8')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend([emotion] + clause.split())
    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words)) # 每个词及词的位置
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words)) # 每个词及词的位置

    w2v = {}
    inputFile2 = open(embedding_path, 'r', encoding='utf-8')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1) # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)])

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)

    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos


def load_data(input_file, word_idx, max_doc_len=75, max_sen_len=45, max_cau_num=3):
    print('load data_file: {}'.format(input_file))
    x, sen_len, doc_len, y, doc_id = [], [], [], [], []
    con_doc_len = []
    emo, cau, con = [], [], []
    wocy = []

    n_cut = 0
    inputFile = open(input_file, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '':
            break
        line = line.strip().split()
        doc_id.append(line[0])
        d_len = int(line[1])
        doc_len.append(d_len)
        y_tmp = np.zeros(2, np.int32)
        y_tmp[int(line[2])] = 1
        wocy_tmp = np.zeros(2, np.int32)
        wocy_tmp[1-int(line[3])] = 1

        pairs = eval('[' + inputFile.readline().strip() + ']')
        emo_all, cau_all = zip(*pairs)
        emo_tmp = np.zeros((1, max_doc_len), dtype=np.int32)
        cau_tmp = np.zeros((max_cau_num, max_doc_len), dtype=np.int32)
        con_tmp = np.zeros((max_doc_len, max_doc_len), dtype=np.int32)

        sen_len_tmp = np.zeros(max_doc_len, dtype=np.int32)
        x_tmp = np.zeros((max_doc_len, max_sen_len), dtype=np.int32)
        cau_cnt = 0
        con_cnt = 0
        for i in range(d_len):
            [_, _, _, words] = inputFile.readline().strip().split(',')
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            if i+1 in emo_all:
                emo_tmp[0][i] = 1
                if i+1 in cau_all:
                    cau_tmp[cau_cnt][i] = 1
                    cau_cnt += 1
            elif i+1 in cau_all:
                cau_tmp[cau_cnt][i] = 1
                cau_cnt += 1
            else:
                con_tmp[con_cnt][i] = 1
                con_cnt += 1
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[i][j] = int(word_idx[word])

        x.append(x_tmp)
        y.append(y_tmp)
        sen_len.append(sen_len_tmp)
        emo.append(emo_tmp)
        cau.append(cau_tmp)
        con.append(con_tmp)
        con_doc_len.append(con_cnt)
        wocy.append(wocy_tmp)

    y, x, sen_len, doc_len, emo, cau, con, con_doc_len, wocy = map(np.array, [y, x, sen_len, doc_len, emo, cau, con, con_doc_len, wocy])
    for var in ['y', 'x', 'sen_len', 'doc_len', 'emo', 'cau', 'con']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('n_cut {}'.format(n_cut))
    print('load data done!\n')
    return doc_id, x, sen_len, doc_len, y, emo, cau, con, wocy


def acc_prf(pred_y, true_y):
    true_y = np.argmax(true_y, 1)
    auc = roc_auc_score(true_y, pred_y[:, 1])
    pred_y = np.argmax(pred_y, 1)
    acc = precision_score(true_y, pred_y, average='micro')
    p = precision_score(true_y, pred_y, average='binary')
    r = recall_score(true_y, pred_y, average='binary')
    f1 = f1_score(true_y, pred_y, average='binary')

    return acc, p, r, f1, auc


def get_logger(log_dir, scope):
    log_file = log_dir + "/" + (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-')) + scope + ".log"
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    # log into file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # log into terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d: %H %M %S'))

    return logger


if __name__ == "__main__":
    emo_dict = {}
    emo_dict["happiness"] = 0
    emo_dict["sadness"] = 1
    emo_dict["fear"] = 2
    emo_dict["anger"] = 3
    emo_dict["disgust"] = 4
    emo_dict["surprise"] = 5

    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(200, 50, '../data/clause_keywords.csv', '../data/w2v_200.txt')
    # print(word_embedding)
    for fold in range(1, 11):
        filename = '../nega_data/fold{}_train.txt'.format(fold)
        _, tr_x, tr_sen_len, tr_doc_len, tr_y, tr_cau, tr_con = load_data_dist(filename, word_id_mapping, emo_dict, 75, 45, 3)
        break
