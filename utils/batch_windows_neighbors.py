import cupy as cp
from cupyx.scipy.sparse import coo_matrix, csr_matrix
import pandas as pd
import numpy as np
import cupyx.scipy.sparse as sp
import time


def set_gpu(device_id):
    cp.cuda.Device(device_id).use()


def load_data(csv_path):
    data = pd.read_csv(csv_path)
    source = data['u'].to_numpy()
    destination = data['i'].to_numpy()
    num_nodes = max(source.max(), destination.max()) + 1
    return source, destination, num_nodes



def build_sparse_matrix(source, destination, num_nodes):
    source_gpu = cp.asarray(source)
    destination_gpu = cp.asarray(destination)

    data = cp.ones(len(source) * 2)
    rows = cp.concatenate([source_gpu, destination_gpu])
    cols = cp.concatenate([destination_gpu, source_gpu])
    adjacency_matrix = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    return adjacency_matrix.tocsr()


def update_sparse_matrix(adjacency_matrix, source, destination, num_nodes):
    source_gpu = cp.asarray(source)
    destination_gpu = cp.asarray(destination)

    data = cp.ones(len(source) * 2)
    rows = cp.concatenate([source_gpu, destination_gpu])
    cols = cp.concatenate([destination_gpu, source_gpu])
    new_edges = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    adjacency_matrix = adjacency_matrix + new_edges
    return adjacency_matrix


def del_sparse_matrix(adjacency_matrix, del_adjacency_matrix):
    adjacency_matrix = adjacency_matrix - del_adjacency_matrix
    return adjacency_matrix

def count_edges_above_threshold(adjacency_matrix, K):
    """
    Count the number of edges where the edge weight is greater than or equal to K
    without modifying the original matrix.

    Parameters:
    adjacency_matrix (csr_matrix): The sparse adjacency matrix.
    K (float): The weight threshold. Count edges where weight >= K.

    Returns:
    int: The number of edges with weight >= K.
    """
    data = adjacency_matrix.data

    valid_edges = cp.asarray(data) >= K

    count = cp.sum(valid_edges).item()

    return count


def compute_neighbors(adjacency_matrix, batch_source, batch_destination):
    if adjacency_matrix is None:
        return 0, 0

    nodes = np.concatenate([batch_source, batch_destination])


    adjacency_submatrix = adjacency_matrix[nodes, :]
    # print(nodes)
    # print('----------------------------')
    # print(adjacency_submatrix)
    print(adjacency_submatrix.nnz / 400)

    # print(adjacency_submatrix.shape)
    # print(adjacency_matrix.shape)
    #
    # print(adjacency_submatrix)
    # print('--------------------------')
    # print(adjacency_matrix)
    # print('--------------------------')

    second_order_neighbors = adjacency_submatrix @ adjacency_matrix
    # print(second_order_neighbors.nnz / 400)
    second_order_count = count_edges_above_threshold(second_order_neighbors, 10) / 400

    third_order_neighbors = second_order_neighbors @ adjacency_matrix
    third_order_count = count_edges_above_threshold(third_order_neighbors, 2000) / 400

    return second_order_count, third_order_count


def process_graph(csv_path, batch_size=200, device_id=0, window_size=100000):

    set_gpu(device_id)


    source, destination, num_nodes = load_data(csv_path)

    adjacency_matrix = None
    batch_results = []

    for batch_start in range(0, len(source), batch_size):
        batch_end = min(batch_start + batch_size, len(source))

        batch_source = source[batch_start:batch_end]
        batch_destination = destination[batch_start:batch_end]

        second_avg, third_avg = compute_neighbors(adjacency_matrix, batch_source, batch_destination)

        if adjacency_matrix is None:
            adjacency_matrix = build_sparse_matrix(batch_source, batch_destination, num_nodes)
        else:
            adjacency_matrix = update_sparse_matrix(adjacency_matrix, batch_source, batch_destination, num_nodes)


        if batch_start >= window_size:
            del_batch_start = batch_start - window_size
            del_batch_end = del_batch_start + batch_size

            del_batch_source = source[del_batch_start:del_batch_end]
            del_batch_destination = destination[del_batch_start:del_batch_end]

            del_adjacency_matrix = build_sparse_matrix(del_batch_source,del_batch_destination,num_nodes)

            adjacency_matrix = del_sparse_matrix(adjacency_matrix, del_adjacency_matrix)
            print('delete complete')
        adjacency_matrix.eliminate_zeros()


        batch_results.append((second_avg, third_avg))
        print(f"Batch {batch_start // batch_size + 1}: "
              f"Average 2nd-order neighbors = {second_avg:.2f}, "
              f"Average 3rd-order neighbors = {third_avg:.2f}")

    return batch_results


csv_path = "ml_aminer.csv"
device_id = 0
start_time = time.time()
window_size = 100000
batch_size = 200
batch_results = process_graph(csv_path, batch_size=batch_size, device_id=device_id, window_size=window_size)
