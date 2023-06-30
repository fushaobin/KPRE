import torch
import torch.nn as nn
import math


class KPRE(nn.Module):
    def __init__(self, args, n_params):
        super(KPRE, self).__init__()
        self.n_layer = args.n_layer
        self.aim_num = args.aim_num
        self.n_relation = n_params[3]
        self.n_entity = n_params[2]
        self.LeakyRelu = nn.LeakyReLU()
        self.Sigmoid = nn.Sigmoid()
        self.n_user = n_params[0]
        self.n_item = n_params[1]
        self.dim = args.dim
        self.sqrt_dim = math.sqrt(args.dim)
        self.agg = args.agg

        self.user_emb = nn.Embedding(self.n_user, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)

        if self.agg == 'concat':
            self.path_agg_weight = nn.Parameter(torch.randn((self.aim_num**self.n_layer)*self.dim, self.dim))

        self.att_weight = nn.Parameter(torch.randn(self.dim*2, 1))

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.user_emb.weight)

        if self.agg == 'concat':
            nn.init.xavier_uniform_(self.path_agg_weight)
        nn.init.xavier_uniform_(self.att_weight)

    def entity_path_agg_neighbor(self, item_triple: list):
        global with_relation_emb_i
        for i in range(self.n_layer - 1, -1, -1):
            if i == self.n_layer-1:
                entity_emb_i = self.entity_emb(item_triple[2][i])
                with_relation_emb_i = torch.mul(self.relation_emb(item_triple[1][i]), entity_emb_i)
            else:
                entity_emb_i = with_relation_emb_i + self.entity_emb(item_triple[2][i])
                with_relation_emb_i = torch.mul(entity_emb_i, self.relation_emb(item_triple[1][i]))
        item_entity_path_emb = (with_relation_emb_i + self.entity_emb(item_triple[0][0])) / (self.n_layer + 1)

        if self.agg == 'average':
            item_entity_path_emb = item_entity_path_emb.sum(axis=1) / (self.aim_num**self.n_layer)
        elif self.agg == 'max':
            item_entity_path_emb = item_entity_path_emb.max(axis=1).values
        elif self.agg == 'min':
            item_entity_path_emb = item_entity_path_emb.min(axis=1).values
        elif self.agg == 'concat':
            item_entity_path_emb = item_entity_path_emb.reshape(-1, (self.aim_num ** self.n_layer) * self.dim)
            item_entity_path_emb = torch.matmul(item_entity_path_emb, self.path_agg_weight)
        else:
            raise Exception("Wrong aggregator")

        return item_entity_path_emb

    def agg_item_neighbor(self, user_triple):
        user_embedding_0 = self.user_emb(user_triple[0][0])
        item_embedding_0 = self.entity_emb(user_triple[1][0])

        user_item = torch.cat((user_embedding_0, item_embedding_0), dim=2)
        att = self.LeakyRelu(torch.matmul(user_item, self.att_weight))
        att_score = torch.softmax(att, dim=1)

        item = item_embedding_0 * att_score
        item_agg = torch.sum(item, dim=1)

        return item_agg

    def predict(self, user_embedding: torch.Tensor, item_embedding: torch.Tensor):
        scores = (item_embedding * user_embedding).sum(dim=1)
        scores = self.Sigmoid(scores)
        return scores

    def forward(self, users: torch.LongTensor, items: torch.LongTensor, user_triple: list, item_triple: list):
        users_embedding_0 = self.user_emb(users)
        items_embedding_0 = self.entity_emb(items)

        item_entity_path_emb = self.entity_path_agg_neighbor(item_triple)
        item_embedding = torch.cat((items_embedding_0, item_entity_path_emb), dim=1)

        item_agg = self.agg_item_neighbor(user_triple)
        user_embedding = torch.cat((users_embedding_0, item_agg), dim=1)

        scores = self.predict(user_embedding, item_embedding)

        return scores
