import numpy as np
import random

def scoop_points(word, all_words, word_embedding, radius, alpha):
    scoop = []
    remaining_mass = []
    for w in all_words:
        if np.linalg.norm(word_embedding[w] - word_embedding[word]) < radius * alpha:
            scoop.append(w)
        else:
            remaining_mass.append(w)
    
    return scoop, remaining_mass
            

def random_cluster(word_embedding):
    # Delta = ?
    # random permutation of points
    # sample Gamma [1/2, 1/4]
    # scoop balls of radius Delta*Gamma
    words = list(word_embedding)
    random.shuffle(words)
    
    X = words

    max_dist = 0
    for i in range(len(words)):
        for j in range(i, len(words)):
            if np.linalg.norm(word_embedding[words[i]] - word_embedding[words[j]]) > max_dist:
                max_dist = np.linalg.norm(word_embedding[words[i]] - word_embedding[words[j]])
    
    clusters = []
    #mean_distance = sum_dist / c
    while len(X) > 0:
        alpha = np.random.uniform(0.25, 0.5)
        s, X = scoop_points(X[0], X, word_embedding, max_dist/16, alpha)    
        print("scoop of size", len(s))
        
        clusters.append(s)

    print(clusters)

if __name__ == "__main__":
    X = [np.array([i for x in range(100)]) for i in range(100)]
    w = [str(i) for i in range(100)]
    
    vocab = {}
    for i in range(len(X)):
        vocab[w[i]] = X[i]
    
    random_cluster(vocab)