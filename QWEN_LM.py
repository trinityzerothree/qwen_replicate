import torch, numpy
import random, faiss
from statistics import mean


def sample_eps(exemplar, n=5, k=1):
    families = random.sample(list(exemplar.keys()), n)

    sup_vecs, sup_labels = [], []
    qry_vecs, qry_labels = [], []

    for family in families:
        support, query = randimg(exemplar, family, k)
        sup_vecs.append(support)
        qry_vecs.append(query)
        sup_labels.extend([family] * k)
        qry_labels.extend([family] * 5)

    support_set = torch.cat(sup_vecs, dim=0)
    query_set   = torch.cat(qry_vecs, dim=0)

    return support_set, sup_labels, query_set, qry_labels
    

def randimg(exemplar, family, k=1):
    matrix = exemplar[family]
    n = len(matrix)

    idx = random.sample(range(n), k + 5)
    support = matrix[idx[:k]]
    query   = matrix[idx[k:]]

    return support, query




emb_model1 = torch.load('qwen_pooled_malimg_emb.pt')
emb_model2 = torch.load('qwen_lastpooled_malimg_emb.pt')

for model_num, exemplar in enumerate([emb_model1, emb_model2], 1):
    for fs in [1,3,5,10]:
        accuracy = []
        for i in range(1000):
            support_set, sup_labels, query_set, qry_labels = sample_eps(exemplar, k = fs)
            index = faiss.IndexFlatIP(support_set.shape[1])  #768
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

        print("at k = ", fs, "the mean acc is: ", mean(accuracy), " :",  model_num)