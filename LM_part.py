import torch, numpy
import random, faiss



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


labels, vecs = [],[]

for x,y in emb_model.items():
    for z in y:
        labels.append(x)
        vecs.append(z)


vecs = torch.stack(vecs).to(torch.float32).numpy()   #.numpy() because FAISS-cpu wants a numpy array, not a torch tensor

index = faiss.IndexFlatIP(2048)
index.add(vecs)

D, I = index.search(vecs[0:1], k=5)    #running a test to see if its working. (it is) A stack-of-one-row is still 2D; a row's contents is 1D. Hence [0:1] works but [0] doesnt


x = sample_eps(emb_model, k= 5)
print(x)