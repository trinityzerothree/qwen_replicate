import torch, numpy
import random, faiss
from statistics import mean


def sample_eps(emb_model, n=5, k=1):
    families = random.sample(list(emb_model.keys()), n)

    sup_vecs, sup_labels = [], []
    qry_vecs, qry_labels = [], []

    for family in families:
        support, query = randimg(family, k)
        sup_vecs.append(support)
        qry_vecs.append(query)
        sup_labels.extend([family] * k)
        qry_labels.extend([family] * 5)

    support_set = torch.cat(sup_vecs, dim=0)
    query_set   = torch.cat(qry_vecs, dim=0)

    return support_set, sup_labels, query_set, qry_labels
    

def randimg(family, k=1):
    matrix = emb_model[family]
    n = len(matrix)

    idx = random.sample(range(n), k + 5)
    support = matrix[idx[:k]]
    query   = matrix[idx[k:]]

    return support, query




emb_model = torch.load('malimg_embeddings.pt')



accuracy = []
for i in range(1000):
    support_set, sup_labels, query_set, qry_labels = sample_eps(emb_model, k = 1)

    index = faiss.IndexFlatIP(2048)
    index.add(support_set.to(torch.float32).numpy())

    D, I = index.search(query_set.to(torch.float32).numpy(), k=10)    #A stack-of-one-row is still 2D; a row's contents is 1D. Hence [0:1] works but [0] doesnt. FAISS never sees labels

    predictions = []

    for row in range(len(I)):
        scores_by_family = {}
        for j in range(len(I[row])):
            pos = I[row][j]
            if pos == -1:
                continue
            family = sup_labels[pos]
            scores_by_family[family] = scores_by_family.get(family, 0.0) + D[row][j]

        pred = max(scores_by_family, key=scores_by_family.get)
        predictions.append(pred)

    c = 0
    for i in range(len(predictions)):
        if predictions[i] == qry_labels[i]: c += 1

    accuracy.append(c / len(predictions))




print(mean(accuracy))