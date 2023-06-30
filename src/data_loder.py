import collections
import numpy as np
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def load_data(args):
    logging.info("================== load cf data ===================")
    cf_data = load_rating(args)
    userID = list(set(cf_data[:, 0].tolist()))
    itemID = list(set(cf_data[:, 1].tolist()))
    n_user, n_item = len(userID), len(itemID)
    logging.info("========= contructing user-item sets =========")
    user_item_sets = cf_propagation(args, cf_data)
    logging.info("================ split dataset(cf) ================")
    train_data, eval_data, test_data = dataset_split(cf_data)
    logging.info("=================== load kg data ===================")
    kg_data, n_entity, n_relation = load_kg(args)
    logging.info("========= contructing items' kg triple sets =========")
    item_entity_sets = kg_propagation(args, cf_data, kg_data)

    return train_data, eval_data, test_data, user_item_sets, item_entity_sets, [n_user, n_item, n_entity, n_relation]


def load_rating(args):
    rating_file = '../data/' + args.dataset + '/ratings_final.txt'
    logging.info("load rating file: %s", rating_file)
    rating_np = np.loadtxt(rating_file, dtype=np.int32)
    return rating_np


def cf_propagation(args, cf_data):
    cf_dict_data = dict()
    for record in cf_data:
        if record[0] not in cf_dict_data:
            cf_dict_data[record[0]] = []
        if record[2] == 1:
            cf_dict_data[record[0]].append(record[1])

    triple_sets = collections.defaultdict(list)
    adj_size = args.adj_size
    for user_id in cf_dict_data.keys():
        u_h, i_t = [], []
        u_h = [user_id] * adj_size
        cf_list = cf_dict_data[user_id]
        indices = np.random.choice(len(cf_list), size=adj_size, replace=(len(cf_list) < adj_size))
        i_t = [cf_list[i] for i in indices]
        triple_sets[user_id].append((u_h, i_t))

    return triple_sets


def load_kg(args):
    rating_file = '../data/' + args.dataset + '/kg_final.txt'
    logging.info("load kg file: %s", rating_file)

    kg_np = np.loadtxt(rating_file, dtype=np.int32)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))

    inv_kg_np = kg_np.copy()
    inv_kg_np[:, 0] = kg_np[:, 2]
    inv_kg_np[:, 2] = kg_np[:, 0]
    inv_kg_np[:, 1] = kg_np[:, 1] + max(kg_np[:, 1]) + 1
    kg_data = np.concatenate((kg_np, inv_kg_np), axis=0)
    n_relation = len(set(kg_data[:, 1]))

    kg_dict_data = dict()
    for record in kg_data:
        if record[0] not in kg_dict_data:
            kg_dict_data[record[0]] = []
        kg_dict_data[record[0]].append((record[1], record[2]))

    return kg_dict_data, n_entity, n_relation


def dataset_split(cf_data):
    logging.info("splitting dataset to 6:2:2 ...")
    eval_ratio = 0.2
    test_ratio = 0.2
    n_cf_data = cf_data.shape[0]

    # splitting datasets
    eval_indices = np.random.choice(n_cf_data, size=int(n_cf_data * eval_ratio), replace=False)
    left = set(range(n_cf_data)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_cf_data * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    train_data = cf_data[train_indices]
    eval_data = cf_data[eval_indices]
    test_data = cf_data[test_indices]

    return train_data, eval_data, test_data


def kg_propagation(args, cf_data, kg_data):
    triple_sets = collections.defaultdict(list)
    layer, aimNum = args.n_layer, args.aim_num
    next_head = []
    for record in cf_data:
        item_id = record[1]
        if item_id in triple_sets:
            continue
        for l in range(layer):
            h, r, t = [], [], []
            if l == 0:
                entities_set = kg_data[item_id]
                for tail_and_relation in entities_set:
                    h.append(item_id)
                    r.append(tail_and_relation[0])
                    t.append(tail_and_relation[1])
                indices = np.random.choice(len(h), size=aimNum, replace=(len(h) < aimNum))
                h = repeat([h[i] for i in indices], aimNum, layer, l)
                r = repeat([r[i] for i in indices], aimNum, layer, l)
                next_head = [t[i] for i in indices]
                t = repeat([t[i] for i in indices], aimNum, layer, l)
                triple_sets[item_id].append((h, r, t))
            else:
                entities = next_head
                next_head = []
                for entity in entities:
                    entities_set = kg_data[entity]
                    e_h, e_r, e_t = [], [], []
                    did = triple_sets[item_id][l-1][0][triple_sets[item_id][l-1][2].index(entity)]
                    for tail_and_relation in entities_set:
                        if tail_and_relation[1] == did:
                            continue
                        e_h.append(entity)
                        e_r.append(tail_and_relation[0])
                        e_t.append(tail_and_relation[1])
                    if len(e_h) == 0:
                        for tail_and_relation in entities_set:
                            e_h.append(entity)
                            e_r.append(tail_and_relation[0])
                            e_t.append(tail_and_relation[1])
                    indices = np.random.choice(len(e_h), size=aimNum, replace=(len(e_h) < aimNum))
                    h = h + repeat([e_h[i] for i in indices], aimNum, layer, l)
                    r = r + repeat([e_r[i] for i in indices], aimNum, layer, l)
                    t = t + repeat([e_t[i] for i in indices], aimNum, layer, l)
                    next_head = next_head + [e_t[i] for i in indices]
                triple_sets[item_id].append((h, r, t))
    return triple_sets


def repeat(e_list, aimNum, layer, l):
    e = []
    for i in e_list:
        repeat_list = [i] * (aimNum ** (layer-l-1))
        e = e + repeat_list
    return e