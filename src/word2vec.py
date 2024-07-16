from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec


def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """

    # Inicializar una matriz para almacenar los vectores de documentos
    corpus_vectors = np.zeros((len(corpus), num_features), dtype="float32")

    # Para cada documento en el corpus
    for i, tokens in enumerate(corpus):
        # Inicializar un contador para el número de palabras en el documento
        num_words = 0
        # Inicializar un vector para el documento actual
        document_vector = np.zeros((num_features,), dtype="float32")

        # Para cada palabra en el documento
        for token in tokens:
            # Si la palabra está presente en el modelo Word2Vec
            if token in model.wv:
                # Sumar el vector de la palabra al vector del documento
                document_vector = np.add(document_vector, model.wv[token])
                # Incrementar el contador de palabras
                num_words += 1

        # Si hay al menos una palabra en el documento
        if num_words > 0:
            # Calcular el promedio dividiendo por el número de palabras
            document_vector = np.divide(document_vector, num_words)

        # Asignar el vector del documento a la matriz de vectores
        corpus_vectors[i] = document_vector

    return corpus_vectors
