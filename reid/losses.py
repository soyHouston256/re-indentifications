"""Memory losses implemented directly from paper Eqs. 3-5 and 13-28."""
import torch
from torch.nn import functional as F


def centroids(memory, labels):
    valid=labels>=0; count=int(labels.max())+1
    centers=memory.new_zeros((count,memory.shape[1]))
    centers.index_add_(0,labels[valid],memory[valid])
    counts=torch.bincount(labels[valid],minlength=count).clamp_min(1)
    return F.normalize(centers/counts[:,None],dim=1)


def global_embedding(x, memory, labels, target, temperature=.05):
    valid=labels>=0
    logits=x @ memory[valid].detach().T / temperature
    logits=logits-logits.max(1,keepdim=True).values
    scores=x.new_zeros((len(x),int(labels.max())+1))
    scores.scatter_add_(1,labels[valid].expand(len(x),-1),logits.exp())
    return F.nll_loss((scores.clamp_min(1e-30)/scores.sum(1,keepdim=True).clamp_min(1e-30)).log(),target)


def cluster_triplet(x, target, centers):
    distance=torch.linalg.vector_norm(x[:,None,:]-x[None,:,:],dim=2)
    same=target[:,None]==target[None,:]
    same.fill_diagonal_(False)
    different=target[:,None]!=target[None,:]
    valid=same.any(1)&different.any(1)
    if not valid.any(): return x.sum()*0
    pos=distance.masked_fill(~same,-torch.inf).max(1).indices
    neg=distance.masked_fill(~different,torch.inf).min(1).indices
    ids=torch.arange(len(x),device=x.device)
    # Detach positive distance: Eq. 24 says to separate negatives only.
    pair=torch.stack((-distance[ids,pos].detach(),-distance[ids,neg]),1)
    log_q=F.log_softmax(pair,dim=1)[:,1]
    p=F.softmax(pair.detach(),dim=1)[:,1]
    reliability=torch.exp(-.5*torch.cdist(x.detach(),centers.detach()).min(1).values)
    # Eq.20 already defines log Q. Eq.24's extra log is inconsistent;
    # use -P * log-probability, avoiding log(negative values).
    return (-reliability[valid]*p[valid]*log_q[valid]).mean()


def objective(features, memories, centers, labels, ids, cem=True):
    target=labels[ids]
    ce=[F.cross_entropy(x @ c.detach().T/.05,target) for x,c in zip(features,centers)]
    loss=.8*ce[0]+.2*(ce[1]+ce[2])
    details={'base':float(loss.detach())}
    if cem:
        gel=global_embedding(features[0],memories[0],labels,target)
        tri=[cluster_triplet(x,target,c) for x,c in zip(features,centers)]
        triplet=tri[0]+.2*(tri[1]+tri[2])
        loss=loss+triplet+.6*gel
        details.update(gel=float(gel.detach()),triplet=float(triplet.detach()))
    return loss,details


@torch.no_grad()
def update_memories(features, memories, centers, labels, ids, momentum=.1):
    for x,mem,c in zip(features,memories,centers):
        x=x.detach()
        mem[ids]=F.normalize(momentum*mem[ids]+(1-momentum)*x,dim=1)
        # Deterministic sequential per-instance Eq.5 update.
        for feature,label in zip(x,labels[ids]):
            c[label]=F.normalize(momentum*c[label]+(1-momentum)*feature,dim=0)
