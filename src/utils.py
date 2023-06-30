import torch
import torch.nn as nn
import numpy as np
import random
import logging
from model import KPRE
from tqdm import tqdm


def set_random_seed(seed, np_seed, torch_seed):
    random.seed(seed)
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)


def init_model(args, n_params):
    model = KPRE(args, n_params)
    model.to(args.device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.l2_weight,
    )
    loss_func = nn.BCELoss()
    return model, optimizer, loss_func


def get_user_triple_tensor(args, objs, triple_set):
    u_h, i_t = [], []
    u_h.append(torch.LongTensor([triple_set[obj][0][0] for obj in objs]))
    i_t.append(torch.LongTensor([triple_set[obj][0][1] for obj in objs]))

    u_h = list(map(lambda x: x.to(args.device), u_h))
    i_t = list(map(lambda x: x.to(args.device), i_t))
    return [u_h, i_t]


def get_item_triple_tensor(args, objs, triple_set):
    h, r, t = [], [], []
    for i in range(args.n_layer):
        h.append(torch.LongTensor([triple_set[obj][i][0] for obj in objs]))
        r.append(torch.LongTensor([triple_set[obj][i][1] for obj in objs]))
        t.append(torch.LongTensor([triple_set[obj][i][2] for obj in objs]))

        h = list(map(lambda x: x.to(args.device), h))
        r = list(map(lambda x: x.to(args.device), r))
        t = list(map(lambda x: x.to(args.device), t))
    return [h, r, t]


def topk_eval(args, model, train_data, test_data, user_item_sets, item_entity_sets):
    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}
    precision_list = {k: [] for k in k_list}

    item_set = set(train_data[:, 1].tolist() + test_data[:, 1].tolist())
    train_record = _get_user_record(train_data, True)
    test_record = _get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 100
    if len(user_list) > user_num:
        np.random.seed()
        user_list = np.random.choice(user_list, size=user_num, replace=False)

    model.eval()
    for user in tqdm(user_list):
        test_item_list = list(item_set-set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size]
            usersID = torch.LongTensor([user] * len(items)).to(args.device)
            itemsID = torch.LongTensor(items).to(args.device)
            user_triple = get_user_triple_tensor(args, [user] * len(items), user_item_sets)
            item_triple = get_item_triple_tensor(args, items, item_entity_sets)
            scores = model(usersID, itemsID, user_triple, item_triple)
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size

        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            usersID = torch.LongTensor([user] * len(res_items)).to(args.device)
            itemsID = torch.LongTensor(res_items).to(args.device)
            user_triple = get_user_triple_tensor(args, [user] * len(res_items), user_item_sets)
            item_triple = get_item_triple_tensor(args, res_items, item_entity_sets)
            scores = model(usersID, itemsID, user_triple, item_triple)
            for item, score in zip(res_items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))
            precision_list[k].append(hit_num / k)
    model.train()
    recall = [np.mean(recall_list[k]) for k in k_list]
    precision = [np.mean(precision_list[k]) for k in k_list]
    _show_top_k_info(zip(k_list, recall), zip(k_list, precision))


def _get_user_record(data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict


def _show_top_k_info(recall_zip, precision_zip):
    res_recall = ""
    for i, j in recall_zip:
        res_recall += "Recall@K%d:%.4f  " % (i, j)
    logging.info(res_recall)
    res_precision = ""
    for i, j in precision_zip:
        res_precision += "Precision@K%d:%.4f  " % (i, j)
    logging.info(res_precision)

