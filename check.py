import torch
d = torch.load("qwen_pooled_malimg_emb.pt")
names = list(d.keys())
X = torch.cat([d[k] for k in names], dim=0)
y = torch.cat([torch.full((d[k].shape[0],), i) for i, k in enumerate(names)])

Xf = X.float()
Xn = Xf / Xf.norm(dim=1, keepdim=True)

def vote(sims, i):
    v = torch.topk(sims, 11)
    idx, sc = v.indices[1:], v.values[1:]
    tally = {}
    for j, s in zip(idx.tolist(), sc.tolist()):
        tally[y[j].item()] = tally.get(y[j].item(), 0) + s
    return max(tally, key=tally.get)

changed = top1 = 0
for i in range(300):
    a = vote(Xf @ Xf[i], i)
    b = vote(Xn @ Xn[i], i)
    changed += (a != b)
    top1 += (torch.topk(Xf @ Xf[i], 2).indices[1] != torch.topk(Xn @ Xn[i], 2).indices[1]).item()
print(f"prediction changed: {changed}/300")
print(f"nearest neighbor changed: {top1}/300")