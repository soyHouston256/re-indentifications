import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import Images, metadata


@torch.inference_mode()
def extract(model, dataset, device, batch_size=32):
    model.eval()
    result = [[], [], []]
    for images, _ in DataLoader(dataset, batch_size=batch_size, num_workers=0):
        for store, feature in zip(result, model(images.to(device))):
            store.append(feature.cpu())
    return tuple(torch.cat(parts) for parts in result)


def score_rows(query, gallery, chunk_size=64):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=4, user_api="blas"):
        for start in range(0, len(query), chunk_size):
            yield from query[start : start + chunk_size] @ gallery.T


def retrieval(query, gallery, qmeta, gmeta):
    """Market protocol: remove junk PID=-1 and same-person/same-camera images."""
    qids = np.asarray(qmeta["person_ids"]).reshape(-1)
    gids = np.asarray(gmeta["person_ids"]).reshape(-1)
    qcams = np.asarray(qmeta["camera_ids"]).reshape(-1)
    gcams = np.asarray(gmeta["camera_ids"]).reshape(-1)
    aps = []
    ranks = []
    examples = []
    for i, scores in enumerate(score_rows(query, gallery)):
        order = np.argsort(-scores, kind="stable")
        keep = (gids[order] != -1) & ~(
            (gids[order] == qids[i]) & (gcams[order] == qcams[i])
        )
        order = order[keep]
        matches = gids[order] == qids[i]
        if not matches.any():
            continue
        hits = np.flatnonzero(matches)
        ap = float(((np.arange(len(hits)) + 1) / (hits + 1)).mean())
        aps.append(ap)
        ranks.append([bool(matches[:k].any()) for k in (1, 5, 10)])
        examples.append(
            {
                "query_index": i,
                "top_gallery_indices": order[:10].tolist(),
                "correct_top1": bool(matches[0]),
                "ap": ap,
                "person_id": int(qids[i]),
            }
        )
    if not aps:
        raise ValueError(
            "No valid cross-camera query matches in this evaluation subset"
        )
    cmc = np.mean(ranks, axis=0)
    return {
        "mAP": float(np.mean(aps)),
        "Rank-1": float(cmc[0]),
        "Rank-5": float(cmc[1]),
        "Rank-10": float(cmc[2]),
        "valid_queries": len(aps),
        "total_queries": len(query),
    }, examples


def evaluate(model, path, device, batch_size=32):
    gallery = extract(model, Images(path, "gallery"), device, batch_size)[0].numpy()
    gmeta = metadata(path, "gallery")
    qmeta = metadata(path, "query")
    results = {}
    all_examples = {}
    for name, low in [("original", False), ("low_resolution", True)]:
        query = extract(model, Images(path, "query", low=low), device, batch_size)[
            0
        ].numpy()
        results[name], all_examples[name] = retrieval(query, gallery, qmeta, gmeta)
    results["mAP_drop_pp"] = 100 * (
        results["original"]["mAP"] - results["low_resolution"]["mAP"]
    )
    return results, all_examples
