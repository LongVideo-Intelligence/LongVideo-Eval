"""FlashVID vision-side compression: torch implementation.

Adapted from the official FlashVID implementation (https://github.com/Fanziyang-v/FlashVID,
MIT License, Copyright (c) 2026 Turbo; see THIRD_PARTY_NOTICES.md): ``flashvid_compression``,
``segment``, ``additional_segment``, ``segment_compression``, ``spatiotemporal_compression``,
``dpc_knn`` and the attention/diversity token selection. Changes: the config type is
``FlashVidVisionConfig``, and only the attention/diversity selection path is kept.

Runs at generate time on the model's device; torch is imported lazily by the caller. The
value-MODIFYING stages live here (temporal merge sums features, density-peak clustering
averages clusters). The numpy index cores in ``pre_llm_compress`` (``segment_np`` /
``attn_div_v2_select_np``) mirror the selection decisions for torch-free tests.
"""
from __future__ import annotations

import math


def pairwise_cosine_distances(image_features):
    import torch  # noqa: F401

    normed = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return 1.0 - torch.bmm(normed, normed.transpose(-1, -2))


def attn_div_v2_based_token_selection(features, cls_attention, num_retained_tokens):
    """Greedy farthest-point selection over attention-calibrated cosine distance."""
    import torch

    original_features = features
    features = features.float()
    pooled_features = features.mean(1)
    global_cls_attention = cls_attention.float() * 1e6
    bsz, num_visual_tokens, feat_dim = features.shape
    dist_matrix = pairwise_cosine_distances(features)
    calibration_term1 = global_cls_attention.unsqueeze(1)
    local_cls_attention = torch.einsum("b n d, c d -> b c n", features, pooled_features).mean(1)
    calibration_term2 = local_cls_attention.unsqueeze(1)
    dist_matrix = dist_matrix * calibration_term1 * calibration_term2
    keep_indices = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=features.device)
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)
    for i in range(1, num_retained_tokens):
        dist_sub_matrix = torch.gather(
            dist_matrix, dim=1,
            index=keep_indices[:, :i].unsqueeze(-1).expand(-1, -1, num_visual_tokens),
        )
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        keep_indices[:, i] = torch.argmax(min_dist, dim=-1)
    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(
        original_features, dim=1,
        index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim),
    )
    return selected_features, keep_indices


def flashvid_compression_torch(video_features, cls_attention, config):
    """Segment the video, then compress each segment to the per-frame token budget."""
    import torch

    num_frames, num_visual_tokens, feat_dim = video_features.shape
    if config.do_segment:
        segment_lengths = _segment(
            video_features.mean(1), config.segment_threshold,
            config.min_segment_num, config.complementary_segment,
        )
    else:
        segment_lengths = torch.tensor([num_frames], dtype=torch.long,
                                       device=video_features.device)
    num_segments = segment_lengths.shape[0]
    global_indices = torch.arange(num_frames * num_visual_tokens, dtype=torch.long,
                                  device=video_features.device)
    token_budget = math.ceil(num_visual_tokens * config.retention_ratio * config.expansion)
    num_attn_div_tokens = math.ceil(token_budget * config.alpha)
    num_sttm_tokens = token_budget - num_attn_div_tokens
    config.num_attn_div_tokens = num_attn_div_tokens
    config.num_sttm_tokens = num_sttm_tokens

    all_segment_features = []
    all_segment_indices = []
    offset = 0
    for seg_idx in range(num_segments):
        seg_len = segment_lengths[seg_idx]
        seg_features = video_features[offset : offset + seg_len]
        seg_cls = cls_attention[offset : offset + seg_len]
        seg_global = global_indices.view(num_frames, num_visual_tokens)[offset : offset + seg_len]
        seg_features, seg_global = _segment_compression(
            seg_features, seg_global, seg_cls, config,
        )
        all_segment_features.append(seg_features)
        all_segment_indices.append(seg_global)
        offset += seg_len
    final_tokens = torch.cat(all_segment_features, dim=0)
    final_global = torch.cat(all_segment_indices, dim=0)
    sorted_indices = final_global.argsort()
    return final_tokens[sorted_indices], final_global[sorted_indices]


def _segment_compression(segment_features, segment_global_indices, cls_attention, config):
    """Compress one segment: attention/diversity selection, then spatio-temporal merging."""
    import torch

    num_frames, num_visual_tokens, feat_dim = segment_features.shape
    if config.alpha > 0:
        selected_features, selected_indices = attn_div_v2_based_token_selection(
            features=segment_features, cls_attention=cls_attention,
            num_retained_tokens=config.num_attn_div_tokens,
        )
        selected_global_indices = segment_global_indices.gather(1, index=selected_indices).view(-1)
    else:
        selected_features = torch.tensor([]).to(segment_features)
        selected_indices = torch.tensor([]).to(segment_global_indices)
        selected_global_indices = torch.tensor([]).to(segment_global_indices)

    mask = torch.ones(num_frames, num_visual_tokens, dtype=torch.bool,
                      device=segment_features.device)
    mask.scatter_(1, selected_indices, False)
    num_other_tokens = config.num_sttm_tokens * num_frames

    if num_other_tokens > 0 and config.temporal_threshold < 1.0:
        if num_frames > 1:
            temp_merged_token_list, temp_merged_indices_list = _spatiotemporal_compression(
                segment_features, config.temporal_threshold, mask, config,
            )
            temp_merged_global_indices_list = [
                segment_global_indices.view(num_frames, -1)[i][idx]
                for i, idx in enumerate(temp_merged_indices_list)
            ]
        else:
            temp_merged_token_list = [segment_features[0]]
            temp_merged_global_indices_list = [segment_global_indices[0]]
    else:
        temp_merged_token_list = []
        temp_merged_global_indices_list = []

    all_tokens = [selected_features.view(-1, feat_dim)]
    all_global_indices = [selected_global_indices]
    if num_other_tokens > 0:
        num_current = sum(len(t) for t in temp_merged_token_list)
        adaptive_ratio = num_other_tokens / num_current if num_current else 0.0
        if adaptive_ratio < 1.0 and num_current:
            nfis = len(temp_merged_token_list)
            max_num = max(len(t) for t in temp_merged_token_list)
            batched = torch.zeros((nfis, max_num, feat_dim), dtype=segment_features.dtype,
                                  device=segment_features.device)
            valid_mask = torch.zeros((nfis, max_num), dtype=torch.bool,
                                     device=segment_features.device)
            num_clusters_list, k_list = [], []
            for i, toks in enumerate(temp_merged_token_list):
                nt = len(toks)
                batched[i, :nt] = toks
                valid_mask[i, :nt] = True
                nc = math.ceil(nt * adaptive_ratio)
                num_clusters_list.append(nc)
                k_list.append(min(nc, 7))
            cluster_idx_list, cluster_center_list = _dpc_knn(
                batched, num_clusters_list, k_list, valid_mask,
            )
            for i, (toks, gidx) in enumerate(
                zip(temp_merged_token_list, temp_merged_global_indices_list)
            ):
                nc = num_clusters_list[i]
                if nc > 0:
                    cidx = cluster_idx_list[i][: len(toks)]
                    ccidx = cluster_center_list[i]
                    agg = torch.zeros((nc, feat_dim), dtype=segment_features.dtype,
                                      device=segment_features.device)
                    agg.scatter_add_(0, cidx.unsqueeze(-1).expand(-1, feat_dim), toks)
                    counts = torch.bincount(cidx, minlength=nc).unsqueeze(-1).to(
                        segment_features.dtype
                    )
                    agg = agg / counts
                    gtok = gidx[ccidx]
                else:
                    agg = toks
                    gtok = gidx
                all_tokens.append(agg)
                all_global_indices.append(gtok)
        else:
            for toks, gidx in zip(temp_merged_token_list, temp_merged_global_indices_list):
                all_tokens.append(toks)
                all_global_indices.append(gidx)

    return torch.cat(all_tokens, dim=0), torch.cat(all_global_indices, dim=0)


def _segment(video_features, segment_threshold, min_segment_num, complementary_segment=True):
    """Segment lengths from frame-mean transition similarity."""
    import torch

    num_frames, feat_dim = video_features.shape
    normed = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    trans = torch.sum(normed[:-1] * normed[1:], dim=-1)
    cut = torch.where(trans < segment_threshold)[0]
    return _additional_segment(cut, num_frames, min_segment_num, trans,
                               segment_threshold, complementary_segment)


def _additional_segment(cut_indices, num_frames, min_segment_num, trans,
                        segment_threshold, complementary_segment=True):
    """Add cuts at the lowest-similarity transitions until the segment floor is met."""
    import torch
    from torch.nn import functional as F

    num_segments = cut_indices.numel() + 1
    if num_segments < min_segment_num and complementary_segment:
        remaining = min_segment_num - num_segments
        trans = trans.clone()
        trans[trans < segment_threshold] = 1.0
        extra = torch.topk(trans, k=min(remaining, trans.shape[0]), largest=False).indices
        cut_indices = torch.cat([cut_indices, extra]).sort().values
    padded = F.pad(cut_indices, (1, 1), value=0)
    padded[0] = -1
    padded[-1] = num_frames - 1
    return torch.diff(padded, n=1, dim=0)


def _dpc_knn(features, num_clusters, k=7, valid_token_mask=None):
    """Density-peak clustering with k nearest neighbours; returns cluster assignments."""
    import torch

    invalid = ~valid_token_mask if valid_token_mask is not None else None
    bsz, seq_len, feat_dim = features.shape
    dists = torch.cdist(features.float(), features.float()) / math.sqrt(feat_dim)
    if valid_token_mask is not None:
        dists = torch.masked_fill(dists, invalid.unsqueeze(1).expand(-1, seq_len, -1),
                                  dists.max() + 1)
    max_k = max(k) if isinstance(k, list) else k
    nearest = torch.topk(dists, k=max_k, dim=-1, largest=False).values
    if isinstance(k, list):
        density = torch.empty((bsz, seq_len), device=features.device, dtype=features.dtype)
        for i in range(bsz):
            density[i] = torch.mean(-(nearest[i, :, : k[i]] ** 2), dim=-1).exp()
    else:
        density = torch.mean(-(nearest ** 2), dim=-1).exp()
    if valid_token_mask is not None:
        density = torch.masked_fill(density, invalid, 0.0)
    mask = density[:, None, :] > density[:, :, None]
    max_dist = dists.view(bsz, -1).max(dim=-1)[0].view(-1, 1, 1)
    modified = torch.where(mask, dists, max_dist)
    dist, _ = torch.min(modified, dim=-1)
    score = dist * density
    if isinstance(num_clusters, int):
        cc = torch.topk(score, k=num_clusters, dim=-1).indices
        dists = torch.gather(dists, dim=-1, index=cc.unsqueeze(1).expand(-1, seq_len, -1))
        ci = torch.argmin(dists, dim=-1)
        ci.scatter_(dim=-1, index=cc,
                    src=torch.arange(num_clusters).to(ci).unsqueeze(0).expand(bsz, -1))
        return ci, cc
    cluster_idx_list, cluster_center_list = [], []
    for i in range(bsz):
        k_i = num_clusters[i]
        if k_i == 0:
            cluster_center_list.append(torch.tensor([], dtype=torch.long, device=features.device))
            cluster_idx_list.append(torch.zeros(seq_len, dtype=torch.long,
                                                device=features.device))
            continue
        cc = torch.topk(score[i], k=k_i, dim=-1).indices
        cluster_center_list.append(cc)
        di = torch.gather(dists[i], dim=-1, index=cc.unsqueeze(0).expand(seq_len, -1))
        ci = torch.argmin(di, dim=-1)
        ci.scatter_(dim=-1, index=cc, src=torch.arange(k_i).to(ci))
        cluster_idx_list.append(ci)
    return cluster_idx_list, cluster_center_list


def _spatiotemporal_compression(video_features, temporal_threshold, token_mask, config):
    """Temporal average merging, then spatial cluster-merge onto density peaks."""
    import torch
    from torch.nn import functional as F

    num_frames, num_visual_tokens, feat_dim = video_features.shape
    lower_bound = (config.num_attn_div_tokens + config.num_sttm_tokens) * num_frames
    normed = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    cos = torch.bmm(normed[1:], normed[:-1].transpose(1, 2))
    cos[~token_mask[1:].unsqueeze(-1).expand(-1, -1, num_visual_tokens)] = -1.0
    cos[~token_mask[:-1].unsqueeze(1).expand(-1, num_visual_tokens, -1)] = -1.0
    max_sims, max_sim_indices = torch.max(cos, dim=-1)
    padded_max_sims = F.pad(max_sims, (0, 0, 1, 0), value=-1)
    padded_max_sim_indices = F.pad(max_sim_indices, (0, 0, 1, 0), value=-1)
    token_counts = torch.ones(num_frames, num_visual_tokens).to(video_features)
    mask = padded_max_sims > temporal_threshold
    retaining_token_mask = ~mask
    if retaining_token_mask.int().sum() < lower_bound:
        soft = padded_max_sims.view(-1).topk(
            k=(num_frames * num_visual_tokens) - lower_bound
        ).values[-1]
        soft = max(soft, -1.0 + 1e-6)
        mask = padded_max_sims > soft
        retaining_token_mask = ~mask
    for frame_idx in range(num_frames - 1, -1, -1):
        frame_features = video_features[frame_idx]
        frame_token_counts = token_counts[frame_idx]
        frame_max_sim_indices = padded_max_sim_indices[frame_idx]
        tokens_to_merge = frame_features[~mask[frame_idx]]
        to_merge_counts = frame_token_counts[~mask[frame_idx]]
        if tokens_to_merge.numel() > 0:
            agg = tokens_to_merge / to_merge_counts.unsqueeze(-1).to(tokens_to_merge.dtype)
            video_features[frame_idx][~mask[frame_idx]] = agg
            token_counts[frame_idx][~mask[frame_idx]] = 1
        other = frame_features[mask[frame_idx]]
        if other.numel() > 0:
            anchors = frame_max_sim_indices[mask[frame_idx]]
            agg = torch.zeros((num_visual_tokens, feat_dim), dtype=video_features.dtype,
                              device=video_features.device)
            agg.scatter_add_(0, anchors.unsqueeze(-1).expand(-1, feat_dim), other)
            agg_counts = torch.bincount(anchors, minlength=num_visual_tokens).to(video_features.dtype)
            video_features[frame_idx - 1] += agg
            token_counts[frame_idx - 1] += agg_counts
            token_counts[frame_idx][mask[frame_idx]] = 0
    final_tokens, retained = [], []
    for i in range(num_frames):
        fm = retaining_token_mask[i] & token_mask[i]
        final_tokens.append(video_features[i][fm])
        retained.append(torch.where(fm)[0])
    return final_tokens, retained
