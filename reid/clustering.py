"""Sparse k-reciprocal Jaccard distances and the paper's Eqs. 6-12."""
import numpy as np
from scipy import sparse
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, sort_graph_by_row_values


def jaccard_graph(features, k=30, expansion_k=6):
    # Sparse memberships keep memory O(Nk), rather than three dense N x N arrays.
    n = len(features)
    k = min(k, n-1)
    distances, indices = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=1).fit(features).kneighbors(features)
    indices = np.asarray([[i] + [int(j) for j in row if j != i][:k] for i,row in enumerate(indices)])
    neighbors = [set(row[1:]) for row in indices]
    rows, cols, values = [], [], []
    for i in range(n):
        reciprocal = {j for j in neighbors[i] if i in neighbors[j]}
        reciprocal.add(i)
        expanded = set(reciprocal)
        for candidate in list(reciprocal):
            half = set(indices[candidate, 1:1+max(1,k//2)])
            half = {j for j in half if candidate in neighbors[j]}
            if half and len(half & reciprocal) > 2*len(half)/3: expanded.update(half)
        ids = np.asarray(sorted(expanded))
        weights = np.exp(-np.maximum(0, 1-features[i] @ features[ids].T))
        weights /= weights.sum()
        rows.extend([i]*len(ids)); cols.extend(ids); values.extend(weights)
    v = sparse.csr_matrix((np.asarray(values,np.float32),(rows,cols)),shape=(n,n))
    if expansion_k > 1:
        count = min(expansion_k,k+1)
        averaging = sparse.csr_matrix((np.full(n*count,1/count,np.float32),
                                      (np.repeat(np.arange(n),count),indices[:,:count].ravel())),shape=(n,n))
        v = (averaging @ v).tocsr()
    return v


def fused_distances(features, m1=.15, k=30, expansion_k=6):
    """Return disk-backed exact weighted Jaccard rows over sparse memberships."""
    import tempfile
    n = len(features[0])
    handle = tempfile.NamedTemporaryFile(prefix='reid-distance-', suffix='.dat', delete=False)
    path = handle.name; handle.close()
    out = np.memmap(path, mode='w+', dtype='float32', shape=(n,n)); out[:] = 0
    for weight, branch in zip((1-2*m1,m1,m1),features):
        if weight == 0: continue
        v = jaccard_graph(branch, k, expansion_k)
        inverted = v.tocsc()
        for i in range(n):
            overlap = np.zeros(n,np.float32)
            for col, val in zip(v.indices[v.indptr[i]:v.indptr[i+1]],v.data[v.indptr[i]:v.indptr[i+1]]):
                start,end = inverted.indptr[col:col+2]
                overlap[inverted.indices[start:end]] += np.minimum(val,inverted.data[start:end])
            out[i] += weight*(1-overlap/np.maximum(2-overlap,1e-8))
    np.fill_diagonal(out,0); out.flush()
    return out,path


def refine(distance, labels, s=5, sigma=.5, budget=40, consensus=.6):
    labels = labels.copy(); original = labels.copy(); n=len(labels)
    s = min(s,n-1)
    neighbors = [np.array([j for j in np.argsort(distance[i],kind='stable') if j != i][:s]) for i in range(n)]
    candidates=[]
    for i in np.where(original < 0)[0]:
        valid=[j for j in neighbors[i] if original[j]>=0 and i in neighbors[j]]
        if not valid: continue
        weights=np.exp(-np.square(distance[i,valid])/(2*sigma*sigma))
        confidence=float(weights.max())
        if confidence < .4: continue
        label=int(original[valid[int(weights.argmax())]])
        if confidence < .6:
            votes={int(c):float(weights[original[valid]==c].sum()) for c in np.unique(original[valid])}
            label=max(votes,key=votes.get)
            if votes[label]/sum(votes.values()) < consensus: continue
        candidates.append((confidence,int(i),label))
    candidates.sort(reverse=True)
    high=[c for c in candidates if c[0]>=.8]
    selected=high[:budget]
    if len(selected) < budget/2:
        selected += [c for c in candidates if c[0]<.8][:budget-len(selected)]
    for _,i,label in selected: labels[i]=label
    return labels, {'outliers_before':int((original<0).sum()),'recovered':len(selected),
                    'high_recovered':sum(c[0]>=.8 for c in selected)}


def cluster(features, eps=.6, min_samples=4, recover=True, k=30):
    import os
    distances,path = fused_distances(features,k=k)
    try:
        # Sparse radius graph avoids sklearn making another dense NxN copy.
        rows=[]; cols=[]; vals=[]
        for i,row in enumerate(distances):
            ids=np.flatnonzero(row<=eps)
            rows.extend([i]*len(ids)); cols.extend(ids); vals.extend(np.maximum(row[ids],1e-8))
        graph=sparse.csr_matrix((vals,(rows,cols)),shape=distances.shape)
        graph=sort_graph_by_row_values(graph,warn_when_not_sorted=False)
        labels=DBSCAN(eps=eps,min_samples=min_samples,metric='precomputed',n_jobs=1).fit_predict(graph)
        stats={'outliers_before':int((labels<0).sum()),'recovered':0}
        if recover: labels,stats=refine(distances,labels)
        stats['clusters']=int(labels.max()+1)
        return labels,stats
    finally:
        del distances
        os.unlink(path)
