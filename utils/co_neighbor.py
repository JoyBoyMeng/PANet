import torch
import torch.nn as nn


def compute_positions(tensor1, tensor2, device='cuda'):
    batch_size1, K1 = tensor1.shape
    batch_size2, K2 = tensor2.shape

    assert batch_size1 == batch_size2, 'Error, wrong batch size in compute_positions'
    batch_size = batch_size1

    tensor1 = tensor1.to(device)
    tensor2 = tensor2.to(device)

    result1 = torch.full((batch_size, K1, 2), -1, dtype=torch.int64, device=device)
    result2 = torch.full((batch_size, K2, 2), -1, dtype=torch.int64, device=device)


    tensor1_expanded = tensor1.unsqueeze(2)  # (batch_size, K1, 1)
    # print(tensor1_expanded)
    tensor2_expanded = tensor2.unsqueeze(1)  # (batch_size, 1, K2)
    # print(tensor2_expanded)

    match_matrix = (tensor1_expanded == tensor2_expanded)  # (batch_size, K1, K2)
    match_matrix = match_matrix.to(torch.int32)
    match_matrix = 2 * match_matrix - 1

    indices_in_tensor2 = match_matrix.argmax(dim=2)
    indices_in_tensor1 = match_matrix.argmax(dim=1)
    # print(indices_in_tensor2)
    # print(indices_in_tensor1)

    result1[:, :, 0] = torch.arange(K1, device=device).repeat(batch_size, 1)
    # print(torch.arange(K1, device=device).repeat(batch_size, 1))
    # print(result1)
    result1[:, :, 1] = indices_in_tensor2
    # print(result1)

    result2[:, :, 0] = torch.arange(K2, device=device).repeat(batch_size, 1)
    result2[:, :, 1] = indices_in_tensor1



    return result1, result2


def split_data(result_indices, result_values, row_counts, l1, l2, hop):

    src_neighbors = result_indices[:l1]
    dst_neighbors = result_indices[l1:l1 + l2]
    neg_neighbors = result_indices[l1 + l2:]
    assert src_neighbors.shape[0] == dst_neighbors.shape[0] and dst_neighbors.shape[0] == \
           neg_neighbors.shape[0], f'Error, wrong {hop} hop neighbors'

    src_values = result_values[:l1]
    dst_values = result_values[l1:l1 + l2]
    neg_values = result_values[l1 + l2:]

    src_counts = row_counts[:l1]
    dst_counts = row_counts[l1:l1 + l2]
    neg_counts = row_counts[l1 + l2:]
    return src_neighbors, dst_neighbors, neg_neighbors, src_values, dst_values, neg_values, src_counts, dst_counts, neg_counts


def compute_values(tensor1_index, tensor2_index, tensor1_v, tensor2_v, device='cuda'):
    '''
    :param tensor1_index: (200, k1+1, 2)
    :param tensor2_index: (200, k2+1, 2)
    :param tensor1_v: (200, k1+1)
    :param tensor2_v: (200, k2+1)
    :param device: 0
    :return:
    '''
    tensor1_values = torch.zeros(tensor1_index.shape, device=device)
    tensor2_values = torch.zeros(tensor2_index.shape, device=device)
    tensor1_values[:, :, 0] = torch.gather(tensor1_v, 1, tensor1_index[:, :, 0])
    tensor1_values[:, :, 1] = torch.gather(tensor2_v, 1, tensor1_index[:, :, 1])

    tensor2_values[:, :, 0] = torch.gather(tensor2_v, 1, tensor2_index[:, :, 0])
    tensor2_values[:, :, 1] = torch.gather(tensor1_v, 1, tensor2_index[:, :, 1])
    return tensor1_values, tensor2_values

def condition_values(tensor1_co_values, tensor2_co_values):
    condition1 = (tensor1_co_values != 0).all(dim=2)
    co_count1 = condition1.sum(dim=1)
    condition2 = (tensor2_co_values != 0).all(dim=2)
    co_count2 = condition2.sum(dim=1)

    condition1 = ((tensor1_co_values == 0).sum(dim=2) == 1)
    unco_count1 = condition1.sum(dim=1)
    condition2 = ((tensor2_co_values == 0).sum(dim=2) == 1)
    unco_count2 = condition2.sum(dim=1)
    return co_count1, co_count2, unco_count1, unco_count2


def co_occurrence(src_1_neighbors, dst_2_neighbors, src_1_values, dst_2_values, batch_size, device):
    prefix = torch.full((batch_size, 1), -1, device=device)
    src_1_neighbors_prefix = torch.cat((prefix, src_1_neighbors), dim=1)
    dst_2_neighbors_prefix = torch.cat((prefix, dst_2_neighbors), dim=1)
    src1_dst2_index, dst2_src1_index = compute_positions(src_1_neighbors_prefix, dst_2_neighbors_prefix)

    prefix = torch.zeros((batch_size, 1), device=device)
    src_1_values_prefix = torch.cat((prefix, src_1_values), dim=1)  # (200,k1+1)
    dst_2_values_prefix = torch.cat((prefix, dst_2_values), dim=1)  # (200,k2+1)

    src_1_co_values, dst_2_co_values = compute_values(src1_dst2_index, dst2_src1_index,
                                                      src_1_values_prefix, dst_2_values_prefix,
                                                      device=device)
    return src_1_co_values.float(), dst_2_co_values.float()


def co_count(src1_dst2_co_values, dst2_src1_co_values, src_1_counts, dst_2_counts, batch_size):
    co_counts_src1_dst2, co_counts_dst2_src1, unco_counts_src1_dst2, unco_counts_dst2_src1 = condition_values(
        src1_dst2_co_values, dst2_src1_co_values)
    assert co_counts_src1_dst2.shape[0] == batch_size, 'num id error'
    assert co_counts_src1_dst2[0] + unco_counts_src1_dst2[0] == src_1_counts[
        0], 'src num neighbor error'
    assert co_counts_dst2_src1[0] + unco_counts_dst2_src1[0] == dst_2_counts[
        0], 'dst num neighbor error'
    co_counts_src1_dst2_sum = co_counts_src1_dst2.sum(dim=0)
    co_counts_dst2_src1_sum = co_counts_dst2_src1.sum(dim=0)
    src_1_counts_sum = src_1_counts.sum(dim=0)
    dst_2_counts_sum = dst_2_counts.sum(dim=0)

    return co_counts_src1_dst2_sum, co_counts_dst2_src1_sum, src_1_counts_sum, dst_2_counts_sum
