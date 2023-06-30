from utils import *
from data_loder import *
from src.parser import parse_args
from sklearn.metrics import roc_auc_score, f1_score
import torch

if __name__ == '__main__':
    # read args
    args = parse_args()
    device = torch.device(args.device)

    # set the random seed
    if args.random_flag:
        set_random_seed(1, 304, 2023)

    # load data and process data
    train_data, eval_data, test_data, user_item_sets, item_entity_sets, n_params = load_data(args)

    logging.info("================== training model ====================")
    model, optimizer, loss_func = init_model(args, n_params)
    for epoch in range(args.n_epoch):

        np.random.shuffle(train_data)
        # train
        start = 0
        auc_train_list = []
        f1_train_list = []
        model.train()
        while start < train_data.shape[0]:
            users = torch.LongTensor(train_data[start:start + args.batch_size, 0]).to(device)  # get userID
            items = torch.LongTensor(train_data[start:start + args.batch_size, 1]).to(device)  # get itemID
            labels = torch.FloatTensor(train_data[start:start + args.batch_size, 2]).to(device)

            user_triple = get_user_triple_tensor(args, train_data[start:start + args.batch_size, 0], user_item_sets)
            item_triple = get_item_triple_tensor(args, train_data[start:start + args.batch_size, 1], item_entity_sets)

            scores = model(users, items, user_triple, item_triple)
            loss = loss_func(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scores = scores.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            auc = roc_auc_score(y_true=labels, y_score=scores)
            predictions = [1 if i >= 0.5 else 0 for i in scores]
            f1 = f1_score(y_true=labels, y_pred=predictions)
            auc_train_list.append(auc)
            f1_train_list.append(f1)

            start += args.batch_size

        auc_train = float(np.mean(auc_train_list))
        f1_train = float(np.mean(f1_train_list))
        print("epoch:", epoch, " auc_train: ", auc_train, " f1_train: ", f1_train, end="")

        # eval
        start = 0
        auc_eval_list = []
        f1_eval_list = []
        model.eval()
        while start < eval_data.shape[0]:
            users = torch.LongTensor(eval_data[start:start + args.batch_size, 0]).to(device)
            items = torch.LongTensor(eval_data[start:start + args.batch_size, 1]).to(device)
            labels = torch.FloatTensor(eval_data[start:start + args.batch_size, 2]).to(device)

            user_triple = get_user_triple_tensor(args, eval_data[start:start + args.batch_size, 0], user_item_sets)
            item_triple = get_item_triple_tensor(args, eval_data[start:start + args.batch_size, 1], item_entity_sets)

            scores = model(users, items, user_triple, item_triple)
            loss = loss_func(scores, labels)

            scores = scores.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            auc = roc_auc_score(y_true=labels, y_score=scores)
            predictions = [1 if i >= 0.5 else 0 for i in scores]
            f1 = f1_score(y_true=labels, y_pred=predictions)
            auc_eval_list.append(auc)
            f1_eval_list.append(f1)

            start += args.batch_size

        auc_eval = float(np.mean(auc_eval_list))
        f1_eval = float(np.mean(f1_eval_list))
        print(" auc_eval: ", auc_eval, " f1_eval: ", f1_eval, end="")

        # test
        start = 0
        auc_test_list = []
        f1_test_list = []
        model.eval()
        while start < test_data.shape[0]:
            users = torch.LongTensor(test_data[start:start + args.batch_size, 0]).to(device)
            items = torch.LongTensor(test_data[start:start + args.batch_size, 1]).to(device)
            labels = torch.FloatTensor(test_data[start:start + args.batch_size, 2]).to(device)

            user_triple = get_user_triple_tensor(args, test_data[start:start + args.batch_size, 0], user_item_sets)
            item_triple = get_item_triple_tensor(args, test_data[start:start + args.batch_size, 1], item_entity_sets)

            scores = model(users, items, user_triple, item_triple)
            loss = loss_func(scores, labels)

            scores = scores.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            auc = roc_auc_score(y_true=labels, y_score=scores)
            predictions = [1 if i >= 0.5 else 0 for i in scores]
            f1 = f1_score(y_true=labels, y_pred=predictions)
            auc_test_list.append(auc)
            f1_test_list.append(f1)

            start += args.batch_size

        auc_test = float(np.mean(auc_test_list))
        f1_test = float(np.mean(f1_test_list))
        print(" auc_test: ", auc_test, " f1_test: ", f1_test)

        if args.show_topk:
            topk_eval(args, model, train_data, test_data, user_item_sets, item_entity_sets)
