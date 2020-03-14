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
    words = list(word_embedding)
    random.shuffle(words)
    
    X = words

    max_dist = 0
    for i in range(len(words)):
        for j in range(i, len(words)):
            if np.linalg.norm(word_embedding[words[i]] - word_embedding[words[j]]) > max_dist:
                max_dist = np.linalg.norm(word_embedding[words[i]] - word_embedding[words[j]])
    
    clusters = []
    
    while len(X) > 0:
        alpha = np.random.uniform(0.25, 0.5)
        s, X = scoop_points(X[0], X, word_embedding, max_dist/16, alpha)    
        
        clusters.append(s)

    return clusters

if __name__ == "__main__":
    s = 1000
    d = 100
    
    X = [np.array([i for x in range(d)]) for i in range(s)]
    w = [str(i) for i in range(s)]
    
    vocab = {}
    for i in range(len(X)):
        vocab[w[i]] = X[i]
    
    c = random_cluster(vocab)
    print(c)