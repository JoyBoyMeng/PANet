import cupy as cp
import cupyx.scipy.sparse as css
import scipy.sparse as ss
import pandas as pd
import numpy as np
import time
import torch


def build_sparse_matrix(source, destination, num_nodes, device_id=0, use_gpu=True):
    if use_gpu:
        with cp.cuda.Device(device_id):
            source_gpu = cp.asarray(source)
            destination_gpu = cp.asarray(destination)
            data = cp.ones(len(source) * 2)
            rows = cp.concatenate([source_gpu, destination_gpu])
            cols = cp.concatenate([destination_gpu, source_gpu])
            adjacency_matrix = css.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
            return adjacency_matrix.tocsr()
    else:
        data = np.ones(len(source) * 2)
        rows = np.concatenate([source, destination])
        cols = np.concatenate([destination, source])
        adjacency_matrix = ss.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        return adjacency_matrix.tocsr()


def update_sparse_matrix(adjacency_matrix, source, destination, num_nodes, device_id=0):
    with cp.cuda.Device(device_id):
        source_gpu = cp.asarray(source)
        destination_gpu = cp.asarray(destination)

        data = cp.ones(len(source) * 2)
        rows = cp.concatenate([source_gpu, destination_gpu])
        cols = cp.concatenate([destination_gpu, source_gpu])
        new_edges = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
        adjacency_matrix = adjacency_matrix + new_edges
        return adjacency_matrix


def del_sparse_matrix(adjacency_matrix, del_adjacency_matrix, device_id):
    with cp.cuda.Device(device_id):
        adjacency_matrix = adjacency_matrix - del_adjacency_matrix
        return adjacency_matrix

def compute_neighbors(adjacency_matrix, batch_source, batch_destination, batch_negative, Train=True, device_id=0, hops=1):
    if adjacency_matrix is None:
        return None, None, None
    with cp.cuda.Device(device_id):
        if Train is True:
            nodes = np.concatenate([batch_source, batch_destination, batch_negative])
        else:
            nodes = np.concatenate([batch_source, batch_destination])

        first_order_neighbors, second_order_neighbors, third_order_neighbors = None, None, None


        if hops >= 1:
            adjacency_submatrix = adjacency_matrix[nodes, :]
            first_order_neighbors = adjacency_submatrix

        if hops >= 2:
            second_order_neighbors = adjacency_submatrix @ adjacency_matrix

        if hops >= 3:
            third_order_neighbors = second_order_neighbors @ adjacency_matrix

        if hops >= 4 or hops <= 0:
            print(f'{hops}-order neighbor calculations are not supported.')
            exit()

        return first_order_neighbors, second_order_neighbors, third_order_neighbors


def select_edges_above_threshold(adjacency_matrix, K, device):
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
    # Extract sparse matrix data
    data = torch.tensor(adjacency_matrix.data, device=device)  # Convert to torch tensor
    indices = torch.tensor(adjacency_matrix.indices, device=device)  # Column indices
    indptr = torch.tensor(adjacency_matrix.indptr, device=device)  # Row pointers

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
    row_counts = torch.diff(torch.cat([torch.tensor([0], dtype=torch.int32, device=device), cumsum_mask])[indptr])

    # Determine the maximum number of valid edges per row
    num_rows = row_counts.size(0)
    max_count = int(row_counts.max())

    # Initialize output tensors with proper padding
    result_indices = torch.full((num_rows, max_count), -1, dtype=torch.int32, device=device)  # Fill unused with -1
    result_values = torch.zeros((num_rows, max_count), dtype=data.dtype, device=device)  # Fill unused with 0

    range_row = torch.arange(row_counts.shape[0], device=device)

    index_row = torch.repeat_interleave(range_row, row_counts)
    range_col = torch.arange(max_count, device=device).repeat(num_rows, 1)

    index_col = torch.cat([row[:num] for row, num in zip(range_col, row_counts)])

    # Fill the results with filtered indices and values
    result_indices[index_row, index_col] = filtered_indices
    result_values[index_row, index_col] = filtered_values

    return result_indices, result_values, row_counts


def get_top_k_values_and_columns_gpu(A, K):

    values = A.data
    col_indices = A.indices
    row_ptr = A.indptr

    top_k_values = []
    top_k_columns = []

    for row_start, row_end in zip(row_ptr[:-1], row_ptr[1:]):
        row_values = values[row_start:row_end]
        row_columns = col_indices[row_start:row_end]

        if len(row_values) >= K:
            top_k_indices = cp.argsort(row_values)[-K:]
        else:
            top_k_indices = cp.argsort(row_values)

        top_k_row_values = row_values[top_k_indices]
        top_k_row_columns = row_columns[top_k_indices]

        if len(top_k_row_values) < K:
            pad_size = K - len(top_k_row_values)
            top_k_row_values = cp.pad(top_k_row_values, (0, pad_size), mode='constant', constant_values=0)
            top_k_row_columns = cp.pad(top_k_row_columns, (0, pad_size), mode='constant',
                                       constant_values=-1)

        top_k_values.append(top_k_row_values)
        top_k_columns.append(top_k_row_columns)

    top_k_values = cp.vstack(top_k_values)
    top_k_columns = cp.vstack(top_k_columns)

    return top_k_values, top_k_columns
