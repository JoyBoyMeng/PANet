import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict
import cupy as cp
import cupyx.scipy.sparse as css
import scipy.sparse as ss
import time

from utils.utils import NeighborSampler
from models.modules import TimeEncoder, MergeLayer, TemporalMultiHeadAttention, SpatialMultiHeadAttention


class DyConNet(torch.nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, model_name: str = 'DyConNet', num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1,
                 src_node_mean_time_shift: float = 0.0, src_node_std_time_shift: float = 1.0,
                 dst_node_mean_time_shift_dst: float = 0.0,
                 dst_node_std_time_shift: float = 1.0, device: str = 'cpu', device_id: int = 0,
                 window_size: int = 100000, neighbor_order_range: int = 3, order1_neighbor_nums: int = 100,
                 order2_neighbor_nums: int = 100, order3_neighbor_nums: int = 100, relation_order: int = 3,
                 bipartite: bool = False, matrix_on_gpu: bool = True):
        """
        General framework for memory-based models, support TGN, DyRep and JODIE.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param model_name: str, name of memory-based models, could be TGN, DyRep or JODIE
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param src_node_mean_time_shift: float, mean of source node time shifts
        :param src_node_std_time_shift: float, standard deviation of source node time shifts
        :param dst_node_mean_time_shift_dst: float, mean of destination node time shifts
        :param dst_node_std_time_shift: float, standard deviation of destination node time shifts
        :param device: str, device
        """
        super(DyConNet, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.device_id = device_id
        self.src_node_mean_time_shift = src_node_mean_time_shift
        self.src_node_std_time_shift = src_node_std_time_shift
        self.dst_node_mean_time_shift_dst = dst_node_mean_time_shift_dst
        self.dst_node_std_time_shift = dst_node_std_time_shift

        self.model_name = model_name
        # number of nodes, including the padded node
        self.num_nodes = self.node_raw_features.shape[0]
        self.memory_dim = self.node_feat_dim
        # since models use the identity function for message encoding, message dimension is 2 * memory_dim + time_feat_dim + edge_feat_dim
        self.message_dim = self.memory_dim + self.memory_dim + self.time_feat_dim + self.edge_feat_dim

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        # message module (models use the identity function for message encoding, hence, we only create MessageAggregator)
        self.message_aggregator = MessageAggregator()

        # memory modules
        self.memory_bank = MemoryBank(num_nodes=self.num_nodes, memory_dim=self.memory_dim)
        self.memory_updater = GRUMemoryUpdater(memory_bank=self.memory_bank, message_dim=self.message_dim,
                                               memory_dim=self.memory_dim)

        self.temporal_embedding_module = TemporalGraphAttentionEmbedding(node_raw_features=self.node_raw_features,
                                                                         edge_raw_features=self.edge_raw_features,
                                                                         neighbor_sampler=neighbor_sampler,
                                                                         time_encoder=self.time_encoder,
                                                                         node_feat_dim=self.node_feat_dim,
                                                                         edge_feat_dim=self.edge_feat_dim,
                                                                         time_feat_dim=self.time_feat_dim,
                                                                         frequence_feat_dim=self.time_feat_dim,
                                                                         num_layers=1,
                                                                         num_heads=self.num_heads,
                                                                         dropout=self.dropout)
        # embedding module
        # Adjacency Matrix within the Window
        self.adjacency_matrix = None
        self.source = None
        self.destination = None
        self.del_index_start = 0
        self.window_size = window_size
        self.neighbor_order_range = neighbor_order_range
        self.relation_order = relation_order
        self.order1_neighbor_nums = order1_neighbor_nums
        self.order2_neighbor_nums = order2_neighbor_nums
        self.order3_neighbor_nums = order3_neighbor_nums
        self.bipartite = bipartite
        self.matrix_on_gpu = matrix_on_gpu
        self.space_feat_dim = self.time_feat_dim
        self.spatial_embedding_module = SpatialAttentionEmbedding(node_raw_features=self.node_raw_features,
                                                                  time_encoder=self.time_encoder,
                                                                  node_feat_dim=self.node_feat_dim,
                                                                  space_feat_dim=self.space_feat_dim,
                                                                  time_feat_dim=self.time_feat_dim,
                                                                  num_layers=1,
                                                                  num_heads=self.num_heads,
                                                                  dropout=self.dropout)
        self.neighbor_co_occurrence_encode_layer_0 = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.space_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.space_feat_dim, out_features=self.space_feat_dim))
        self.neighbor_co_occurrence_encode_layer_1 = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.space_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.space_feat_dim, out_features=self.space_feat_dim))
        self.neighbor_co_occurrence_encode_layer_2 = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.space_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.space_feat_dim, out_features=self.space_feat_dim))
        self.neighbor_co_occurrence_encode_layer_3 = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.space_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.space_feat_dim, out_features=self.space_feat_dim))
        self.neighbor_encode_layer = nn.Sequential(
            nn.Linear(in_features=3 * self.space_feat_dim, out_features=self.space_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.space_feat_dim, out_features=self.space_feat_dim))

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray,
                                                 edge_ids: np.ndarray, edges_are_positive: bool = True,
                                                 num_neighbors: int = 20, slide_window: bool = False,
                                                 del_src_node_ids: np.ndarray = None,
                                                 del_dst_node_ids: np.ndarray = None):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :param edges_are_positive: boolean, whether the edges are positive,
        determine whether to update the memories and raw messages for nodes in src_node_ids and dst_node_ids or not
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        src_node_nums = len(src_node_ids)
        # Tensor, shape (2 * batch_size, )
        node_ids = np.concatenate([src_node_ids, dst_node_ids])

        if self.adjacency_matrix is None:
            src1_dst0_co_values = torch.zeros((src_node_nums, self.order1_neighbor_nums, 2)).to(self.device)
            src1_dst1_co_values = torch.zeros((src_node_nums, self.order1_neighbor_nums, 2)).to(self.device)
            src1_dst2_co_values = torch.zeros((src_node_nums, self.order1_neighbor_nums, 2)).to(self.device)
            dst1_src0_co_values = torch.zeros((src_node_nums, self.order1_neighbor_nums, 2)).to(self.device)
            dst1_src1_co_values = torch.zeros((src_node_nums, self.order1_neighbor_nums, 2)).to(self.device)
            dst1_src2_co_values = torch.zeros((src_node_nums, self.order1_neighbor_nums, 2)).to(self.device)
            neighbors_1 = torch.zeros((2 * src_node_nums, self.order1_neighbor_nums), dtype=torch.int32).to(self.device)
            first_order_dense = torch.zeros((2 * src_node_nums, self.num_nodes+1), dtype=torch.float32).to(self.device)
        else:
            first_order_neighbors, second_order_neighbors, third_order_neighbors = self.compute_neighbors(src_node_ids,
                                                                                                          dst_node_ids)
            neighbors_0 = torch.from_numpy(node_ids).to(self.device).unsqueeze(1)
            values_0 = torch.full(neighbors_0.shape, self.window_size, device=self.device)

            mask_1 = None
            neighbors_1, values_1, counts_1, non_zero_indices_1, first_order_dense = self.select_edges_topk(
                first_order_neighbors,
                self.order1_neighbor_nums,
                mask_1,
                return_dense=True)
            mask_2 = non_zero_indices_1
            neighbors_2, values_2, counts_2, non_zero_indices_2, _ = self.select_edges_topk(second_order_neighbors,
                                                                                            self.order2_neighbor_nums,
                                                                                            mask_2)

            src0_dst1_co_values, dst1_src0_co_values = self.co_occurrence(neighbors_0[:src_node_nums],
                                                                          neighbors_1[src_node_nums:],
                                                                          values_0[:src_node_nums],
                                                                          values_1[src_node_nums:])
            src1_dst0_co_values, dst0_src1_co_values = self.co_occurrence(neighbors_1[:src_node_nums],
                                                                          neighbors_0[src_node_nums:],
                                                                          values_1[:src_node_nums],
                                                                          values_0[src_node_nums:])
            src1_dst1_co_values, dst1_src1_co_values = self.co_occurrence(neighbors_1[:src_node_nums],
                                                                          neighbors_1[src_node_nums:],
                                                                          values_1[:src_node_nums],
                                                                          values_1[src_node_nums:])
            src1_dst2_co_values, dst2_src1_co_values = self.co_occurrence(neighbors_1[:src_node_nums],
                                                                          neighbors_2[src_node_nums:],
                                                                          values_1[:src_node_nums],
                                                                          values_2[src_node_nums:])
            src2_dst1_co_values, dst1_src2_co_values = self.co_occurrence(neighbors_2[:src_node_nums],
                                                                          neighbors_1[src_node_nums:],
                                                                          values_2[:src_node_nums],
                                                                          values_1[src_node_nums:])

            if self.neighbor_order_range > 3:
                print(f'Higher-order neighbor calculations beyond {self.neighbor_order_range}-order are not supported.')
                exit()

        # adj_size = len(self.source) if len(self.source) < self.window_size else self.window_size
        # adj_size = adj_size / 1000
        adj_size = 1
        # src embedding
        emb_src1_dst0 = self.embed_layer(src1_dst0_co_values, 1, 0, adj_size)
        emb_src1_dst1 = self.embed_layer(src1_dst1_co_values, 1, 1, adj_size)
        emb_src1_dst2 = self.embed_layer(src1_dst2_co_values, 1, 2, adj_size)
        emb_src_1 = torch.cat((emb_src1_dst0, emb_src1_dst1, emb_src1_dst2), dim=2)
        emb_src = self.neighbor_encode_layer(emb_src_1)
        # dst embedding
        emb_dst1_src0 = self.embed_layer(dst1_src0_co_values, 1, 0, adj_size)
        emb_dst1_src1 = self.embed_layer(dst1_src1_co_values, 1, 1, adj_size)
        emb_dst1_src2 = self.embed_layer(dst1_src2_co_values, 1, 2, adj_size)
        emb_dst_1 = torch.cat((emb_dst1_src0, emb_dst1_src1, emb_dst1_src2), dim=2)
        emb_dst = self.neighbor_encode_layer(emb_dst_1)
        neighbor_space_features = torch.cat((emb_src, emb_dst), dim=0)
        assert neighbor_space_features.shape == (
            len(node_ids), self.order1_neighbor_nums, self.time_feat_dim), 'co-neighbor compute wrong'

        updated_node_memories, updated_node_last_updated_times = self.get_updated_memories(
            node_ids=np.array(range(self.num_nodes)),
            node_raw_messages=self.memory_bank.node_raw_messages)
        node_spatial_embeddings = self.spatial_embedding_module.compute_node_temporal_embeddings(
            node_memories=updated_node_memories,
            node_ids=node_ids,
            neighbor_node_ids=neighbors_1,
            neighbor_space_features=neighbor_space_features,
            current_layer_num=1)
        src_node_spatial_embeddings, dst_node_spatial_embeddings = node_spatial_embeddings[
                                                                   :len(src_node_ids)], node_spatial_embeddings[
                                                                                        len(src_node_ids): len(
                                                                                            src_node_ids) + len(
                                                                                            dst_node_ids)]

        node_temporal_embeddings = self.temporal_embedding_module.compute_node_temporal_embeddings(
            node_memories=updated_node_memories,
            node_ids=node_ids,
            node_interact_times=np.concatenate([node_interact_times,node_interact_times]),
            dense_interaction=first_order_dense,
            current_layer_num=self.num_layers,
            num_neighbors=num_neighbors)
        src_node_temporal_embeddings, dst_node_temporal_embeddings = node_temporal_embeddings[
                                                                     :len(src_node_ids)], node_temporal_embeddings[
                                                                                          len(src_node_ids): len(
                                                                                              src_node_ids) + len(
                                                                                              dst_node_ids)]

        src_node_embeddings = torch.cat((src_node_temporal_embeddings, src_node_spatial_embeddings), dim=1)
        dst_node_embeddings = torch.cat((dst_node_temporal_embeddings, dst_node_spatial_embeddings), dim=1)

        # src_node_embeddings, dst_node_embeddings = src_node_temporal_embeddings, dst_node_temporal_embeddings

        if edges_are_positive:
            assert edge_ids is not None
            # if the edges are positive, update the memories for source and destination nodes (since now we have new messages for them)
            self.update_memories(node_ids=node_ids, node_raw_messages=self.memory_bank.node_raw_messages)

            # clear raw messages for source and destination nodes since we have already updated the memory using them
            self.memory_bank.clear_node_raw_messages(node_ids=node_ids)

            # compute new raw messages for source and destination nodes
            unique_src_node_ids, new_src_node_raw_messages = self.compute_new_node_raw_messages(
                src_node_ids=src_node_ids,
                dst_node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
                edge_ids=edge_ids)
            unique_dst_node_ids, new_dst_node_raw_messages = self.compute_new_node_raw_messages(
                src_node_ids=dst_node_ids,
                dst_node_ids=src_node_ids,
                node_interact_times=node_interact_times,
                edge_ids=edge_ids)

            # store new raw messages for source and destination nodes
            self.memory_bank.store_node_raw_messages(node_ids=unique_src_node_ids,
                                                     new_node_raw_messages=new_src_node_raw_messages)
            self.memory_bank.store_node_raw_messages(node_ids=unique_dst_node_ids,
                                                     new_node_raw_messages=new_dst_node_raw_messages)

        if edges_are_positive:
            # Adjacency Matrix Initialization
            if self.adjacency_matrix is None:
                self.build_sparse_matrix(src_node_ids, dst_node_ids)
                self.source = src_node_ids
                self.destination = dst_node_ids
            # Update After Adjacency Matrix Calculation
            else:
                self.add_sparse_matrix(src_node_ids, dst_node_ids)
                self.source = np.concatenate((self.source, src_node_ids))
                self.destination = np.concatenate((self.destination, dst_node_ids))
                # Slide After the Window is Filled
                # assert (len(self.source) > self.window_size) == slide_window, '是否delete判断错误'
                if len(self.source) > self.window_size:
                    # Determine the starting and ending indices of the deleted IDs
                    shift = len(src_node_ids)
                    del_index_start = self.del_index_start
                    del_index_end = del_index_start + shift
                    del_src_node_ids_cal = self.source[del_index_start: del_index_end]
                    del_dst_node_ids_cal = self.destination[del_index_start: del_index_end]
                    # assert np.array_equal(del_src_node_ids_cal, del_src_node_ids), 'src_node_ids计算错误'
                    # assert np.array_equal(del_dst_node_ids_cal, del_dst_node_ids), 'dst_node_ids计算错误'
                    self.del_sparse_matrix(del_src_node_ids_cal, del_dst_node_ids_cal)
                    # Sliding the starting deletion point
                    self.del_index_start += shift

        return src_node_embeddings, dst_node_embeddings

    def get_updated_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        get the updated memories based on node_ids and node_raw_messages (just for computation), but not update the memories
        :param node_ids: ndarray, shape (num_nodes, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        # aggregate messages for the same nodes
        # unique_node_ids, ndarray, shape (num_unique_node_ids, ), array of unique node ids
        # unique_node_messages, Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        # unique_node_timestamps, ndarray, shape (num_unique_node_ids, ), array of timestamps for unique nodes
        unique_node_ids, unique_node_messages, unique_node_timestamps = self.message_aggregator.aggregate_messages(
            node_ids=node_ids,
            node_raw_messages=node_raw_messages)
        # get updated memory for all nodes with messages stored in previous batches (just for computation)
        # updated_node_memories, Tensor, shape (num_nodes, memory_dim)
        # updated_node_last_updated_times, Tensor, shape (num_nodes, )
        updated_node_memories, updated_node_last_updated_times = self.memory_updater.get_updated_memories(
            unique_node_ids=unique_node_ids,
            unique_node_messages=unique_node_messages,
            unique_node_timestamps=unique_node_timestamps)

        return updated_node_memories, updated_node_last_updated_times

    def update_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        update memories for nodes in node_ids
        :param node_ids: ndarray, shape (num_nodes, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        # aggregate messages for the same nodes
        # unique_node_ids, ndarray, shape (num_unique_node_ids, ), array of unique node ids
        # unique_node_messages, Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        # unique_node_timestamps, ndarray, shape (num_unique_node_ids, ), array of timestamps for unique nodes
        unique_node_ids, unique_node_messages, unique_node_timestamps = self.message_aggregator.aggregate_messages(
            node_ids=node_ids,
            node_raw_messages=node_raw_messages)

        # update the memories with the aggregated messages
        self.memory_updater.update_memories(unique_node_ids=unique_node_ids, unique_node_messages=unique_node_messages,
                                            unique_node_timestamps=unique_node_timestamps)

    def compute_new_node_raw_messages(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                      node_interact_times: np.ndarray, edge_ids: np.ndarray):
        """
        compute new raw messages for nodes in src_node_ids
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param dst_node_embeddings: Tensor, shape (batch_size, node_feat_dim)
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :return:
        """
        # Tensor, shape (batch_size, memory_dim)
        src_node_memories = self.memory_bank.get_memories(node_ids=src_node_ids)
        dst_node_memories = self.memory_bank.get_memories(node_ids=dst_node_ids)

        # Tensor, shape (batch_size, )
        src_node_delta_times = torch.from_numpy(node_interact_times).float().to(self.device) - \
                               self.memory_bank.node_last_updated_times[torch.from_numpy(src_node_ids)]
        # Tensor, shape (batch_size, time_feat_dim)
        src_node_delta_time_features = self.time_encoder(src_node_delta_times.unsqueeze(dim=1)).reshape(
            len(src_node_ids), -1)

        # Tensor, shape (batch_size, edge_feat_dim)
        edge_features = self.edge_raw_features[torch.from_numpy(edge_ids)]

        # Tensor, shape (batch_size, message_dim = memory_dim + memory_dim + time_feat_dim + edge_feat_dim)
        new_src_node_raw_messages = torch.cat(
            [src_node_memories, dst_node_memories, src_node_delta_time_features, edge_features], dim=1)

        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        new_node_raw_messages = defaultdict(list)
        # ndarray, shape (num_unique_node_ids, )
        unique_node_ids = np.unique(src_node_ids)

        for i in range(len(src_node_ids)):
            new_node_raw_messages[src_node_ids[i]].append((new_src_node_raw_messages[i], node_interact_times[i]))

        return unique_node_ids, new_node_raw_messages

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        assert self.model_name in ['TGN', 'DyRep',
                                   'DyConNet'], f'Neighbor sampler is not defined in model {self.model_name}!'
        self.temporal_embedding_module.neighbor_sampler = neighbor_sampler
        if self.temporal_embedding_module.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.temporal_embedding_module.neighbor_sampler.seed is not None
            self.temporal_embedding_module.neighbor_sampler.reset_random_state()

    def build_sparse_matrix(self, source, destination):
        time1 = time.time()
        if self.matrix_on_gpu:
            with cp.cuda.Device(self.device_id):
                source_gpu = cp.asarray(source, dtype=cp.int32)
                destination_gpu = cp.asarray(destination, dtype=cp.int32)
                data = cp.ones(len(source) * 2, dtype=cp.float32)
                rows = cp.concatenate([source_gpu, destination_gpu])
                cols = cp.concatenate([destination_gpu, source_gpu])
                self.adjacency_matrix = css.coo_matrix((data, (rows, cols)),
                                                       shape=(self.num_nodes, self.num_nodes)).tocsr()
                # return adjacency_matrix.tocsr()
        else:
            data = np.ones(len(source) * 2)
            rows = np.concatenate([source, destination])
            cols = np.concatenate([destination, source])
            self.adjacency_matrix = ss.coo_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes)).tocsr()
            # return adjacency_matrix.tocsr()

    def add_sparse_matrix(self, source, destination):
        if self.matrix_on_gpu:
            with cp.cuda.Device(self.device_id):
                source_gpu = cp.asarray(source, dtype=cp.int32)
                destination_gpu = cp.asarray(destination, dtype=cp.int32)
                data = cp.ones(len(source) * 2, dtype=cp.float32)
                # print(f'data, source, destination{data.dtype, source_gpu.dtype, destination_gpu.dtype}')
                rows = cp.concatenate([source_gpu, destination_gpu])
                cols = cp.concatenate([destination_gpu, source_gpu])
                new_edges = css.coo_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes)).tocsr()
                self.adjacency_matrix = self.adjacency_matrix + new_edges
                # return adjacency_matrix
        else:
            data = np.ones(len(source) * 2)
            rows = np.concatenate([source, destination])
            cols = np.concatenate([destination, source])
            new_edges = ss.coo_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes)).tocsr()
            self.adjacency_matrix = self.adjacency_matrix + new_edges
            # return adjacency_matrix

    def del_sparse_matrix(self, source, destination):
        if self.matrix_on_gpu:
            with cp.cuda.Device(self.device_id):
                source_gpu = cp.asarray(source, dtype=cp.int32)
                destination_gpu = cp.asarray(destination, dtype=cp.int32)
                data = cp.ones(len(source) * 2, dtype=cp.float32)
                rows = cp.concatenate([source_gpu, destination_gpu])
                cols = cp.concatenate([destination_gpu, source_gpu])
                del_edges = css.coo_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes)).tocsr()
                self.adjacency_matrix = self.adjacency_matrix - del_edges
        else:
            data = np.ones(len(source) * 2)
            rows = np.concatenate([source, destination])
            cols = np.concatenate([destination, source])
            del_edges = ss.coo_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes)).tocsr()
            self.adjacency_matrix = self.adjacency_matrix - del_edges
        print('delete complete')

    # Calculate Higher-order Neighbors through Sparse Matrix Multiplication
    def compute_neighbors(self, source, destination):
        first_order_neighbors, second_order_neighbors, third_order_neighbors = None, None, None
        if self.adjacency_matrix is None:
            return first_order_neighbors, second_order_neighbors, third_order_neighbors
        nodes = np.concatenate([source, destination])
        if self.matrix_on_gpu:
            with (cp.cuda.Device(self.device_id)):
                if self.neighbor_order_range >= 1:
                    first_order_neighbors = self.adjacency_matrix[nodes, :]
                if self.neighbor_order_range >= 2:
                    second_order_neighbors = first_order_neighbors @ self.adjacency_matrix
                if self.neighbor_order_range >= 3:
                    third_order_neighbors = second_order_neighbors @ self.adjacency_matrix
                    third_order_neighbors = third_order_neighbors - first_order_neighbors
                if self.neighbor_order_range >= 4 or self.neighbor_order_range <= 0:
                    print(f'{self.neighbor_order_range}-order neighbor calculations are not supported.')
                    exit()
        else:
            if self.neighbor_order_range >= 1:
                first_order_neighbors = self.adjacency_matrix[nodes, :]
            if self.neighbor_order_range >= 2:
                second_order_neighbors = first_order_neighbors @ self.adjacency_matrix
            if self.neighbor_order_range >= 3:
                third_order_neighbors = second_order_neighbors @ self.adjacency_matrix
                third_order_neighbors = third_order_neighbors - first_order_neighbors
            if self.neighbor_order_range >= 4 or self.neighbor_order_range <= 0:
                print(f'{self.neighbor_order_range}-order neighbor calculations are not supported.')
                exit()

        # print(f'first_order_neighbors.data.dtype, second_order_neighbors.data.dtype, third_order_neighbors.data.dtype:'
        #       f'{first_order_neighbors.data.dtype, second_order_neighbors.data.dtype, third_order_neighbors.data.dtype}')
        return first_order_neighbors, second_order_neighbors, third_order_neighbors

    def select_edges_topk(self, matrix, neighbor_nums, mask, return_dense=False):
        time1 = time.time()
        data = torch.tensor(matrix.data, dtype=torch.float32, device=self.device)  # Convert to torch tensor
        indices = torch.tensor(matrix.indices, dtype=torch.int32, device=self.device)  # Column indices
        indptr = torch.tensor(matrix.indptr, dtype=torch.int32, device=self.device)  # Row pointers
        num_rows, num_cols = matrix.shape

        row_counts = torch.diff(indptr)
        range_row = torch.arange(num_rows, dtype=torch.int32).to(self.device)
        rows = torch.repeat_interleave(range_row, row_counts)
        cols = indices

        dense_matrix = torch.zeros((num_rows, num_cols), device=self.device)
        dense_matrix[rows, cols] = data
        # dense_matrix = torch.tensor(matrix.toarray()).to(self.device)
        if mask != None:
            dense_matrix[mask] = 0
        non_zero_indices = (dense_matrix != 0)
        time1 = time.time()
        values, indices = dense_matrix.topk(neighbor_nums, dim=1, largest=True, sorted=False)
        mask_values = values == 0
        indices[mask_values] = 0
        # print('debug')
        # print(indices[0])
        # print(values[0])
        if return_dense:
            return indices, values, None, non_zero_indices, dense_matrix
        return indices, values, None, non_zero_indices, None

    def select_edges_above_threshold(self, adjacency_matrix, K):
        """
        Get the column indices, corresponding values of edges where weight >= K for a CSR sparse matrix,
        and the count of valid edges per row, using PyTorch tensors.

        Parameters:
        adjacency_matrix (csr_matrix): The sparse adjacency matrix.
        K (float): The weight threshold. Count edges where weight >= K.

        Returns:
        tuple: A tuple of three tensors:
            - (torch.Tensor) The column indices of edges with weight >= K, grouped by rows.
            - (torch.Tensor) The corresponding edge weights.
            - (torch.Tensor) The count of valid edges per row.
        """
        time1 = time.time()
        # Extract sparse matrix data
        data = torch.tensor(adjacency_matrix.data, device=self.device)  # Convert to torch tensor
        indices = torch.tensor(adjacency_matrix.indices, device=self.device)  # Column indices
        indptr = torch.tensor(adjacency_matrix.indptr, device=self.device)  # Row pointers

        # Create a boolean mask for edge weights >= K
        mask = data >= K

        # Filter indices and values based on the mask
        filtered_indices = indices[mask]
        filtered_values = data[mask]

        # Compute cumulative sum of the mask
        cumsum_mask = torch.cumsum(mask.int(), dim=0)

        # Ensure indptr[-1] does not exceed the size of cumsum_mask
        assert indptr[-1] <= cumsum_mask.size(0), "Invalid CSR matrix: indptr[-1] exceeds data size"

        # Use indptr to calculate valid row counts
        row_counts = torch.diff(
            torch.cat([torch.tensor([0], dtype=torch.int32, device=self.device), cumsum_mask])[indptr])

        # Determine the maximum number of valid edges per row
        num_rows = row_counts.size(0)
        max_count = int(row_counts.max())

        # Initialize output tensors with proper padding
        result_indices = torch.full((num_rows, max_count), -1, dtype=torch.int32,
                                    device=self.device)  # Fill unused with -1
        result_values = torch.zeros((num_rows, max_count), dtype=data.dtype, device=self.device)  # Fill unused with 0

        range_row = torch.arange(row_counts.shape[0], device=self.device)

        index_row = torch.repeat_interleave(range_row, row_counts)
        range_col = torch.arange(max_count, device=self.device).repeat(num_rows, 1)

        index_col = torch.cat([row[:num] for row, num in zip(range_col, row_counts)])

        # Fill the results with filtered indices and values
        result_indices[index_row, index_col] = filtered_indices
        result_values[index_row, index_col] = filtered_values

        print(f'阈值筛选用时：{time.time() - time1}')

        return result_indices, result_values, row_counts

    def co_occurrence(self, src_neighbors, dst_neighbors, src_values, dst_values):
        src_node_nums = src_neighbors.shape[0]
        # Obtain the co-occurrence position matrix
        nei_prefix = torch.full((src_node_nums, 1), -1, device=self.device)
        src_neighbors_prefixed = torch.cat((nei_prefix, src_neighbors), dim=1)  # (200, Ksrc+1)
        dst_neighbors_prefixed = torch.cat((nei_prefix, dst_neighbors), dim=1)  # (200, Kdst+1)
        src_dst_index, dst_src_index = self.compute_positions(src_neighbors_prefixed, dst_neighbors_prefixed)
        # Compute the co-occurrence connectivity matrix through the co-occurrence position matrix
        value_prefix = torch.zeros((src_node_nums, 1), device=self.device)
        src_values_prefixed = torch.cat((value_prefix, src_values), dim=1)  # (200, Ksrc+1)
        dst_values_prefixed = torch.cat((value_prefix, dst_values), dim=1)  # (200, Kdst+1)
        src_dst_co_values, dst_src_co_values = self.compute_values(src_dst_index, dst_src_index, src_values_prefixed,
                                                                   dst_values_prefixed)
        return src_dst_co_values.float()[:, 1:, :], dst_src_co_values.float()[:, 1:, :]

    def compute_positions(self, src_neighbors, dst_neighbors):
        batch_size1, Ksrc = src_neighbors.shape
        batch_size2, Kdst = dst_neighbors.shape
        assert batch_size1 == batch_size2, 'Error, wrong batch size in compute_positions'
        batch_size = batch_size1

        # Initialize the result tensor
        result_src = torch.full((batch_size, Ksrc, 2), -1, dtype=torch.int64, device=self.device)
        result_dst = torch.full((batch_size, Kdst, 2), -1, dtype=torch.int64, device=self.device)

        # Expand src_neighbors and dst_neighbors for broadcasting comparison
        src_neighbors_expanded = src_neighbors.unsqueeze(2)  # (batch_size, Ksrc+1, 1)
        dst_neighbors_expanded = dst_neighbors.unsqueeze(1)  # (batch_size, 1, Kdst+1)
        # Use broadcasting to compare all elements of the two tensors
        match_matrix = (src_neighbors_expanded == dst_neighbors_expanded)  # (batch_size, Ksrc, Kdst)
        # Convert the boolean matrix to an integer matrix, with matching elements as 1 and non-matching elements as 0
        match_matrix = match_matrix.to(torch.int32)  # bool->int32
        match_matrix = 2 * match_matrix - 1  # 1->1, 0->-1
        # Find the matching positions in the match_matrix
        indices_in_dst_neighbors = match_matrix.argmax(
            dim=2)  # Find the positions of src's neighbors in dst's neighbors
        indices_in_src_neighbors = match_matrix.argmax(
            dim=1)  # Find the positions of dst's neighbors in src's neighbors

        # The positions of elements in src_neighbors that are equal to elements in src_neighbors
        result_src[:, :, 0] = torch.arange(Ksrc, device=self.device).repeat(batch_size, 1)
        # The positions of elements in dst_neighbors that are equal to elements in src_neighbors
        result_src[:, :, 1] = indices_in_dst_neighbors
        # The positions of elements in dst_neighbors that are equal to elements in dst_neighbors
        result_dst[:, :, 0] = torch.arange(Kdst, device=self.device).repeat(batch_size, 1)
        # The positions of elements in src_neighbors that are equal to elements in dst_neighbors
        result_dst[:, :, 1] = indices_in_src_neighbors
        return result_src, result_dst

    def compute_values(self, tensor1_index, tensor2_index, tensor1_v, tensor2_v):
        '''
        :param tensor1_index: (200, k1+1, 2)
        :param tensor2_index: (200, k2+1, 2)
        :param tensor1_v: (200, k1+1)
        :param tensor2_v: (200, k2+1)
        :param device: 0
        :return:
        '''
        tensor1_values = torch.zeros(tensor1_index.shape, device=self.device)
        tensor2_values = torch.zeros(tensor2_index.shape, device=self.device)
        # 2-> (self, other)
        tensor1_values[:, :, 0] = torch.gather(tensor1_v, 1, tensor1_index[:, :, 0])
        tensor1_values[:, :, 1] = torch.gather(tensor2_v, 1, tensor1_index[:, :, 1])
        # 2-> (self, other)
        tensor2_values[:, :, 0] = torch.gather(tensor2_v, 1, tensor2_index[:, :, 0])
        tensor2_values[:, :, 1] = torch.gather(tensor1_v, 1, tensor2_index[:, :, 1])
        return tensor1_values, tensor2_values

    def embed_layer(self, tensor, k1, k2, P):

        tensor1, tensor2 = tensor[:, :, 0:1], tensor[:, :, 1:2]
        tensor1 = tensor1 / P
        tensor2 = tensor2 / P
        if k1 == 0:
            mlp1 = self.neighbor_co_occurrence_encode_layer_0
        elif k1 == 1:
            mlp1 = self.neighbor_co_occurrence_encode_layer_1
        elif k1 == 2:
            mlp1 = self.neighbor_co_occurrence_encode_layer_2
        elif k1 == 3:
            mlp1 = self.neighbor_co_occurrence_encode_layer_3
        else:
            exit()
        if k2 == 0:
            mlp2 = self.neighbor_co_occurrence_encode_layer_0
        elif k2 == 1:
            mlp2 = self.neighbor_co_occurrence_encode_layer_1
        elif k2 == 2:
            mlp2 = self.neighbor_co_occurrence_encode_layer_2
        elif k2 == 3:
            mlp2 = self.neighbor_co_occurrence_encode_layer_3
        else:
            exit()
        output1 = mlp1(tensor1)
        output2 = mlp2(tensor2)

        output = output1 + output2

        return output

    def reload_matrix(self, backup_matrix):
        self.adjacency_matrix = backup_matrix[0]
        self.source = backup_matrix[1]
        self.destination = backup_matrix[2]
        self.del_index_start = backup_matrix[3]

    def backup_matrix(self):
        return (self.adjacency_matrix, self.source, self.destination, self.del_index_start)

    def init_matrix(self):
        self.adjacency_matrix = None
        self.source = None
        self.destination = None
        self.del_index_start = 0

    # Message-related Modules


class MessageAggregator(nn.Module):

    def __init__(self):
        """
        Message aggregator. Given a batch of node ids and corresponding messages, aggregate messages with the same node id.
        """
        super(MessageAggregator, self).__init__()

    def aggregate_messages(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        given a list of node ids, and a list of messages of the same length,
        aggregate different messages with the same node id (only keep the last message for each node)
        :param node_ids: ndarray, shape (batch_size, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        unique_node_ids = np.unique(node_ids)
        unique_node_messages, unique_node_timestamps, to_update_node_ids = [], [], []

        for node_id in unique_node_ids:
            if len(node_raw_messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_node_messages.append(node_raw_messages[node_id][-1][0])
                unique_node_timestamps.append(node_raw_messages[node_id][-1][1])

        # ndarray, shape (num_unique_node_ids, ), array of unique node ids
        to_update_node_ids = np.array(to_update_node_ids)
        # Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        unique_node_messages = torch.stack(unique_node_messages, dim=0) if len(
            unique_node_messages) > 0 else torch.Tensor([])
        # ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        unique_node_timestamps = np.array(unique_node_timestamps)

        return to_update_node_ids, unique_node_messages, unique_node_timestamps


# Memory-related Modules
class MemoryBank(nn.Module):

    def __init__(self, num_nodes: int, memory_dim: int):
        """
        Memory bank, store node memories, node last updated times and node raw messages.
        :param num_nodes: int, number of nodes
        :param memory_dim: int, dimension of node memories
        """
        super(MemoryBank, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        # Parameter, treat memory as parameters so that it is saved and loaded together with the model, shape (num_nodes, memory_dim)
        self.node_memories = nn.Parameter(torch.zeros((self.num_nodes, self.memory_dim)), requires_grad=False)
        # Parameter, last updated time of nodes, shape (num_nodes, )
        self.node_last_updated_times = nn.Parameter(torch.zeros(self.num_nodes), requires_grad=False)
        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        self.node_raw_messages = defaultdict(list)

        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        """
        initialize all the memories and node_last_updated_times to zero vectors, reset the node_raw_messages, which should be called at the start of each epoch
        :return:
        """
        self.node_memories.data.zero_()
        self.node_last_updated_times.data.zero_()
        self.node_raw_messages = defaultdict(list)

    def get_memories(self, node_ids: np.ndarray):
        """
        get memories for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        return self.node_memories[torch.from_numpy(node_ids)]

    def set_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):
        """
        set memories for nodes in node_ids to updated_node_memories
        :param node_ids: ndarray, shape (batch_size, )
        :param updated_node_memories: Tensor, shape (num_unique_node_ids, memory_dim)
        :return:
        """
        self.node_memories[torch.from_numpy(node_ids)] = updated_node_memories

    def backup_memory_bank(self):
        """
        backup the memory bank, get the copy of current memories, node_last_updated_times and node_raw_messages
        :return:
        """
        cloned_node_raw_messages = {}
        for node_id, node_raw_messages in self.node_raw_messages.items():
            cloned_node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy()) for
                                                 node_raw_message in node_raw_messages]

        return self.node_memories.data.clone(), self.node_last_updated_times.data.clone(), cloned_node_raw_messages

    def reload_memory_bank(self, backup_memory_bank: tuple):
        """
        reload the memory bank based on backup_memory_bank
        :param backup_memory_bank: tuple (node_memories, node_last_updated_times, node_raw_messages)
        :return:
        """
        self.node_memories.data, self.node_last_updated_times.data = backup_memory_bank[0].clone(), backup_memory_bank[
            1].clone()

        self.node_raw_messages = defaultdict(list)
        for node_id, node_raw_messages in backup_memory_bank[2].items():
            self.node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy()) for
                                               node_raw_message in node_raw_messages]

    def detach_memory_bank(self):
        """
        detach the gradients of node memories and node raw messages
        :return:
        """
        self.node_memories.detach_()

        # Detach all stored messages
        for node_id, node_raw_messages in self.node_raw_messages.items():
            new_node_raw_messages = []
            for node_raw_message in node_raw_messages:
                new_node_raw_messages.append((node_raw_message[0].detach(), node_raw_message[1]))

            self.node_raw_messages[node_id] = new_node_raw_messages

    def store_node_raw_messages(self, node_ids: np.ndarray, new_node_raw_messages: dict):
        """
        store raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param new_node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id].extend(new_node_raw_messages[node_id])

    def clear_node_raw_messages(self, node_ids: np.ndarray):
        """
        clear raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id] = []

    def get_node_last_updated_times(self, unique_node_ids: np.ndarray):
        """
        get last updated times for nodes in unique_node_ids
        :param unique_node_ids: ndarray, (num_unique_node_ids, )
        :return:
        """
        return self.node_last_updated_times[torch.from_numpy(unique_node_ids)]

    def extra_repr(self):
        """
        set the extra representation of the module, print customized extra information
        :return:
        """
        return 'num_nodes={}, memory_dim={}'.format(self.node_memories.shape[0], self.node_memories.shape[1])


class MemoryUpdater(nn.Module):

    def __init__(self, memory_bank: MemoryBank):
        """
        Memory updater.
        :param memory_bank: MemoryBank
        """
        super(MemoryUpdater, self).__init__()
        self.memory_bank = memory_bank

    def update_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                        unique_node_timestamps: np.ndarray):
        """
        update memories for nodes in unique_node_ids
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, ), array of unique node ids
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        :return:
        """
        # if unique_node_ids is empty, return without updating operations
        if len(unique_node_ids) <= 0:
            return

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(
                    unique_node_messages.device)).all().item(), "Trying to update memory to time in the past!"

        # Tensor, shape (num_unique_node_ids, memory_dim)
        node_memories = self.memory_bank.get_memories(node_ids=unique_node_ids)
        # Tensor, shape (num_unique_node_ids, memory_dim)
        updated_node_memories = self.memory_updater(unique_node_messages, node_memories)
        # update memories for nodes in unique_node_ids
        self.memory_bank.set_memories(node_ids=unique_node_ids, updated_node_memories=updated_node_memories)

        # update last updated times for nodes in unique_node_ids
        self.memory_bank.node_last_updated_times[torch.from_numpy(unique_node_ids)] = torch.from_numpy(
            unique_node_timestamps).float().to(unique_node_messages.device)

    def get_updated_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                             unique_node_timestamps: np.ndarray):
        """
        get updated memories based on unique_node_ids, unique_node_messages and unique_node_timestamps
        (just for computation), but not update the memories
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, ), array of unique node ids
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        :return:
        """
        # if unique_node_ids is empty, directly return node_memories and node_last_updated_times without updating
        if len(unique_node_ids) <= 0:
            return self.memory_bank.node_memories.data.clone(), self.memory_bank.node_last_updated_times.data.clone()

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids=unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(
                    unique_node_messages.device)).all().item(), "Trying to update memory to time in the past!"

        # Tensor, shape (num_nodes, memory_dim)
        updated_node_memories = self.memory_bank.node_memories.data.clone()
        updated_node_memories[torch.from_numpy(unique_node_ids)] = self.memory_updater(unique_node_messages,
                                                                                       updated_node_memories[
                                                                                           torch.from_numpy(
                                                                                               unique_node_ids)])

        # Tensor, shape (num_nodes, )
        updated_node_last_updated_times = self.memory_bank.node_last_updated_times.data.clone()
        updated_node_last_updated_times[torch.from_numpy(unique_node_ids)] = torch.from_numpy(
            unique_node_timestamps).float().to(unique_node_messages.device)

        return updated_node_memories, updated_node_last_updated_times


class GRUMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(GRUMemoryUpdater, self).__init__(memory_bank)

        self.memory_updater = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)


class RNNMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        RNN-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(RNNMemoryUpdater, self).__init__(memory_bank)

        self.memory_updater = nn.RNNCell(input_size=message_dim, hidden_size=memory_dim)


# Embedding-related Modules
class TimeProjectionEmbedding(nn.Module):

    def __init__(self, memory_dim: int, dropout: float):
        """
        Time projection embedding module.
        :param memory_dim: int, dimension of node memories
        :param dropout: float, dropout rate
        """
        super(TimeProjectionEmbedding, self).__init__()

        self.memory_dim = memory_dim
        self.dropout = nn.Dropout(dropout)

        self.linear_layer = nn.Linear(1, self.memory_dim)

    def compute_node_temporal_embeddings(self, node_memories: torch.Tensor, node_ids: np.ndarray,
                                         node_time_intervals: torch.Tensor):
        """
        compute node temporal embeddings using the embedding projection operation in JODIE
        :param node_memories: Tensor, shape (num_nodes, memory_dim)
        :param node_ids: ndarray, shape (batch_size, )
        :param node_time_intervals: Tensor, shape (batch_size, )
        :return:
        """
        # Tensor, shape (batch_size, memory_dim)
        source_embeddings = self.dropout(
            node_memories[torch.from_numpy(node_ids)] * (1 + self.linear_layer(node_time_intervals.unsqueeze(dim=1))))

        return source_embeddings


class SpatialAttentionEmbedding(nn.Module):

    def __init__(self, node_raw_features: torch.Tensor,
                 time_encoder: TimeEncoder, node_feat_dim: int, space_feat_dim: int, time_feat_dim: int,
                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1):
        """
        Graph attention embedding module.
        :param node_raw_features: Tensor, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: Tensor, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_encoder: TimeEncoder
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim:  int, dimension of time features (encodings)
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(SpatialAttentionEmbedding, self).__init__()

        self.node_raw_features = node_raw_features
        self.time_encoder = time_encoder
        self.node_feat_dim = node_feat_dim
        self.time_feat_dim = time_feat_dim
        self.space_feat_dim = space_feat_dim
        self.num_layers = 1
        self.num_heads = num_heads
        self.dropout = dropout

        self.temporal_conv_layers = nn.ModuleList([SpatialMultiHeadAttention(node_feat_dim=self.node_feat_dim,
                                                                             space_feat_dim=self.space_feat_dim,
                                                                             time_feat_dim=self.time_feat_dim,
                                                                             num_heads=self.num_heads,
                                                                             dropout=self.dropout) for _ in
                                                   range(num_layers)])
        # follow the TGN paper, use merge layer to combine 1) the attention results, and 2) node raw feature + node memory
        self.merge_layers = nn.ModuleList(
            [MergeLayer(input_dim1=self.node_feat_dim, input_dim2=self.node_feat_dim,
                        hidden_dim=self.node_feat_dim, output_dim=self.node_feat_dim) for _ in range(num_layers)])

    def compute_node_temporal_embeddings(self, node_memories: torch.Tensor, node_ids: np.ndarray,
                                         neighbor_node_ids: torch.Tensor, neighbor_space_features: torch.Tensor,
                                         current_layer_num: int):
        """
        given memory, node ids node_ids, and the corresponding time node_interact_times,
        return the temporal embeddings after convolution at the current_layer_num
        :param node_memories: Tensor, shape (num_nodes, memory_dim)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param current_layer_num: int, current layer number
        :param num_neighbors: int, number of neighbors to sample for each node
        """

        assert (current_layer_num >= 0)
        device = self.node_raw_features.device

        # shape (batch_size, node_feat_dim)
        # add memory and node raw features to get node features
        # note that when using getting values of the ids from Tensor, convert the ndarray to tensor to avoid wrong retrieval
        node_features = node_memories[torch.from_numpy(node_ids)] + self.node_raw_features[torch.from_numpy(node_ids)]
        neighbor_node_features = node_memories[neighbor_node_ids] + self.node_raw_features[neighbor_node_ids]

        # neighbor_node_ids shape (batch_size, num_neighbors)
        # temporal graph convolution
        # Tensor, output shape (batch_size, node_feat_dim + time_feat_dim)
        output, _ = self.temporal_conv_layers[current_layer_num - 1](node_features=node_features,
                                                                     node_time_features=None,
                                                                     neighbor_node_features=neighbor_node_features,
                                                                     neighbor_node_time_features=None,
                                                                     neighbor_node_space_features=neighbor_space_features,
                                                                     neighbor_masks=neighbor_node_ids)

        # Tensor, output shape (batch_size, node_feat_dim)
        # follow the TGN paper, use merge layer to combine 1) the attention results, and 2) node raw feature + node memory
        output = self.merge_layers[current_layer_num - 1](input_1=output, input_2=node_features)
        return output


class TemporalGraphAttentionEmbedding(nn.Module):

    def __init__(self, node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor,
                 neighbor_sampler: NeighborSampler,
                 time_encoder: TimeEncoder, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int, frequence_feat_dim: int,
                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1):
        """
        Graph attention embedding module.
        :param node_raw_features: Tensor, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: Tensor, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_encoder: TimeEncoder
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim:  int, dimension of time features (encodings)
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TemporalGraphAttentionEmbedding, self).__init__()

        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.neighbor_sampler = neighbor_sampler
        self.time_encoder = time_encoder
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.frequence_feat_dim = frequence_feat_dim

        self.temporal_conv_layers = nn.ModuleList([TemporalMultiHeadAttention(node_feat_dim=self.node_feat_dim,
                                                                              edge_feat_dim=self.edge_feat_dim,
                                                                              time_feat_dim=self.time_feat_dim,
                                                                              frequence_feat_dim=self.frequence_feat_dim,
                                                                              num_heads=self.num_heads,
                                                                              dropout=self.dropout) for _ in
                                                   range(num_layers)])
        # follow the TGN paper, use merge layer to combine 1) the attention results, and 2) node raw feature + node memory
        self.merge_layers = nn.ModuleList(
            [MergeLayer(input_dim1=self.node_feat_dim + self.time_feat_dim, input_dim2=self.node_feat_dim,
                        hidden_dim=self.node_feat_dim, output_dim=self.node_feat_dim) for _ in range(num_layers)])

        self.neighbor_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.frequence_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.frequence_feat_dim, out_features=self.frequence_feat_dim))

    def compute_node_temporal_embeddings(self, node_memories: torch.Tensor, node_ids: np.ndarray,
                                         node_interact_times: np.ndarray, dense_interaction: torch.Tensor,
                                         current_layer_num: int, num_neighbors: int = 20):
        """
        given memory, node ids node_ids, and the corresponding time node_interact_times,
        return the temporal embeddings after convolution at the current_layer_num
        :param node_memories: Tensor, shape (num_nodes, memory_dim)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param current_layer_num: int, current layer number
        :param num_neighbors: int, number of neighbors to sample for each node
        """

        assert (current_layer_num >= 0)
        device = self.node_raw_features.device

        # query (source) node always has the start time with time interval == 0
        # shape (batch_size, 1, time_feat_dim)
        node_time_features = self.time_encoder(
            timestamps=torch.zeros(node_interact_times.shape).unsqueeze(dim=1).to(device))
        # shape (batch_size, node_feat_dim)
        # add memory and node raw features to get node features
        # note that when using getting values of the ids from Tensor, convert the ndarray to tensor to avoid wrong retrieval
        node_features = node_memories[torch.from_numpy(node_ids)] + self.node_raw_features[torch.from_numpy(node_ids)]

        if current_layer_num == 0:
            return node_features
        else:
            # get source node representations by aggregating embeddings from the previous (curr_layers - 1)-th layer
            # Tensor, shape (batch_size, node_feat_dim)
            node_conv_features = self.compute_node_temporal_embeddings(node_memories=node_memories,
                                                                       node_ids=node_ids,
                                                                       node_interact_times=node_interact_times,
                                                                       dense_interaction=None,
                                                                       current_layer_num=current_layer_num - 1,
                                                                       num_neighbors=num_neighbors)

            # get temporal neighbors, including neighbor ids, edge ids and time information
            # neighbor_node_ids ndarray, shape (batch_size, num_neighbors)
            # neighbor_edge_ids ndarray, shape (batch_size, num_neighbors)
            # neighbor_times ndarray, shape (batch_size, num_neighbors)
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                               node_interact_times=node_interact_times,
                                                               num_neighbors=num_neighbors)
            neighbors_tensor = torch.tensor(neighbor_node_ids)
            batch_size = neighbors_tensor.shape[0]
            assert dense_interaction.shape[0] == batch_size, 'Wrong batch size'
            row_sums = dense_interaction.sum(dim=1)
            row_sums[row_sums == 0] = 1
            occurrence = dense_interaction[torch.arange(batch_size).unsqueeze(1), neighbors_tensor]
            occurrence[occurrence == 0] = 1
            normalized_occurrence = occurrence / row_sums.unsqueeze(1)

            # print(neighbor_node_ids[0])
            # print(occurrence[0])
            # print(neighbor_node_ids[-1])
            # print(occurrence[-1])

            occurrence = normalized_occurrence.unsqueeze(-1)
            # occurrence = occurrence.unsqueeze(-1)

            # shape(batch_size, num_neighbors, frequence_feat_dim)
            occurrence_features = self.neighbor_occurrence_encode_layer(occurrence)
            # get neighbor features from previous layers
            # shape (batch_size * num_neighbors, node_feat_dim)
            neighbor_node_conv_features = self.compute_node_temporal_embeddings(node_memories=node_memories,
                                                                                node_ids=neighbor_node_ids.flatten(),
                                                                                node_interact_times=neighbor_times.flatten(),
                                                                                dense_interaction=None,
                                                                                current_layer_num=current_layer_num - 1,
                                                                                num_neighbors=num_neighbors)

            # shape (batch_size, num_neighbors, node_feat_dim)
            neighbor_node_conv_features = neighbor_node_conv_features.reshape(node_ids.shape[0], num_neighbors,
                                                                              self.node_feat_dim)

            # compute time interval between current time and historical interaction time
            # adarray, shape (batch_size, num_neighbors)
            neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times

            # shape (batch_size, num_neighbors, time_feat_dim)
            neighbor_time_features = self.time_encoder(
                timestamps=torch.from_numpy(neighbor_delta_times).float().to(device))

            # get edge features, shape (batch_size, num_neighbors, edge_feat_dim)
            neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
            # temporal graph convolution
            # Tensor, output shape (batch_size, node_feat_dim + time_feat_dim)
            output, _ = self.temporal_conv_layers[current_layer_num - 1](node_features=node_conv_features,
                                                                         node_time_features=node_time_features,
                                                                         neighbor_node_features=neighbor_node_conv_features,
                                                                         neighbor_node_time_features=neighbor_time_features,
                                                                         neighbor_node_edge_features=neighbor_edge_features,
                                                                         neighbor_node_frequence_features=occurrence_features,
                                                                         neighbor_masks=neighbor_node_ids)

            # Tensor, output shape (batch_size, node_feat_dim)
            # follow the TGN paper, use merge layer to combine 1) the attention results, and 2) node raw feature + node memory
            output = self.merge_layers[current_layer_num - 1](input_1=output, input_2=node_features)

            return output


def compute_src_dst_node_time_shifts(src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                     node_interact_times: np.ndarray):
    """
    compute the mean and standard deviation of time shifts
    :param src_node_ids: ndarray, shape (*, )
    :param dst_node_ids:: ndarray, shape (*, )
    :param node_interact_times: ndarray, shape (*, )
    :return:
    """
    src_node_last_timestamps = dict()
    dst_node_last_timestamps = dict()
    src_node_all_time_shifts = []
    dst_node_all_time_shifts = []
    for k in range(len(src_node_ids)):
        src_node_id = src_node_ids[k]
        dst_node_id = dst_node_ids[k]
        node_interact_time = node_interact_times[k]
        if src_node_id not in src_node_last_timestamps.keys():
            src_node_last_timestamps[src_node_id] = 0
        if dst_node_id not in dst_node_last_timestamps.keys():
            dst_node_last_timestamps[dst_node_id] = 0
        src_node_all_time_shifts.append(node_interact_time - src_node_last_timestamps[src_node_id])
        dst_node_all_time_shifts.append(node_interact_time - dst_node_last_timestamps[dst_node_id])
        src_node_last_timestamps[src_node_id] = node_interact_time
        dst_node_last_timestamps[dst_node_id] = node_interact_time
    assert len(src_node_all_time_shifts) == len(src_node_ids)
    assert len(dst_node_all_time_shifts) == len(dst_node_ids)
    src_node_mean_time_shift = np.mean(src_node_all_time_shifts)
    src_node_std_time_shift = np.std(src_node_all_time_shifts)
    dst_node_mean_time_shift_dst = np.mean(dst_node_all_time_shifts)
    dst_node_std_time_shift = np.std(dst_node_all_time_shifts)

    return src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift
