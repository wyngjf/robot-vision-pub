from typing import List, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(array1: np.ndarray, array2: np.ndarray):
    """Compute similarity between two arrays using Euclidean distance"""
    return np.linalg.norm(array1 - array2, axis=1).mean()


def compute_cosine_similarity(array1: np.ndarray, array2: np.ndarray):
    """compute cosine similarity"""
    similarity = cosine_similarity(array1.reshape(1, -1), array2.reshape(1, -1))
    return similarity[0, 0]


def get_similarity_func(similarity_measure: str):
    return dict(
        cosine=compute_cosine_similarity,
        euc=compute_similarity
    )[similarity_measure]


def get_similarity_opt_max_flag(similarity_measure: str):
    return dict(
        cosine=True,
        euc=False
    )[similarity_measure]


def compute_similarity_matrix(
        list1: List[np.ndarray],
        list2: List[np.ndarray],
        similarity_measure: str = "cosine"
):
    n1, n2 = len(list1), len(list2)
    similarity_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            similarity = get_similarity_func(similarity_measure)(list1[i], list2[j])
            similarity_matrix[i, j] = similarity

    return similarity_matrix


def compute_assignment(
        prev_landmarks: List[np.ndarray],
        current_landmarks: List[np.ndarray],
        similarity_measure: Literal['cosine', 'euc'] = "cosine"
):
    """
    compute the assignment indices.

    Args:
        prev_landmarks:
        current_landmarks:
        similarity_measure:

    Returns: row_indices, the indices in the prev_landmarks corresponds to the indices in the col_indices of the
        current_landmarks

    """
    similarity_matrix = compute_similarity_matrix(prev_landmarks, current_landmarks, similarity_measure)
    row_indices, col_indices = linear_sum_assignment(
        similarity_matrix,
        maximize=get_similarity_opt_max_flag(similarity_measure)
    )
    return row_indices.tolist(), col_indices.tolist()
