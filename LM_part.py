import torch
import numpy
import faiss

emb_model = torch.load('malimg_embeddings.pt')

#wanker = list(emb_model.values())
#print(wanker[0])

labels, vecs = [],[]

for x,y in emb_model.items():
    for z in y:
        labels.append(x)
        vecs.append(z)


vecs = torch.stack(vecs).to(torch.float32).numpy()   #.numpy() because FAISS-cpu wants a numpy array, not a torch tensor

index = faiss.IndexFlatIP(2048)
index.add(vecs)

D, I = index.search(vecs[0:1], k=5)    #running a test to see if its working. (it is) A stack-of-one-row is still 2D; a row's contents is 1D. Hence [0:1] works but [0] doesnt

print(D, I)