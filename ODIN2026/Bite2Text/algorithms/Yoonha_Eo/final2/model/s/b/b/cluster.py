import torch
from torchtyping import TensorType


def learned_region_cluster(
    offsets,
    sigmas,
    seeds,
    min_seed_score: float=0.9,
    min_cluster_size: int=20,
    min_unclustered: float=0.35,
):
    out = [torch.full((n,1), -1).to(seeds.F).long() for n in seeds.batch_counts]

    # determine spatial embeddings
    instance_idx = 0
    for b in range(offsets.batch_size):
        # prepare inputs
        offsets_b = torch.tanh(offsets.batch(b).F)
        sigmas_b = torch.exp(sigmas.batch(b).F * 10)
        seeds_b = torch.sigmoid(seeds.batch(b).F)
        mask_b = seeds_b >= 0.5

        # determine instances by clustering
        spatial_embeds = offsets_b + offsets.batch(b).C
        while mask_b.sum() >= min_cluster_size:
            voxel_idx = (seeds_b * mask_b).argmax()
            mask_b[voxel_idx] = False
            if seeds_b[voxel_idx] < min_seed_score:
                break

            center = spatial_embeds[voxel_idx]
            bandwidth = sigmas_b[voxel_idx]
            probs = (seeds_b >= 0.5) * torch.exp(-1 * torch.sum(
                bandwidth * torch.pow(spatial_embeds - center, 2),
                dim=1,
                keepdim=True,
            ))  # 1 x d x h x w
            proposal = probs > 0.5

            num_points = proposal.sum()
            overlap = mask_b[proposal].sum() / num_points
            if num_points >= min_cluster_size and overlap >= min_unclustered:
                out[b][proposal] = instance_idx
                instance_idx += 1
            
            mask_b[proposal] = False

    out = seeds.new_tensor(features=torch.cat(out)[:, 0])

    return out


def wdbscan(
	dmatrix: TensorType['K', 'K', torch.float32],
	epsilon: float,
	mu: float,
	weights=None,
	noise=True,
):
	"""
	Generates a density-based clustering of arbitrary shape. Returns a numpy
	array coding cluster membership with noise observations coded as 0.
	
	Positional arguments:
	dmatrix -- square dissimilarity matrix (cf. numpy.matrix, numpy.squareform,
	           and numpy.pdist).
	epsilon -- maximum reachability distance.
	mu      -- minimum reachability weight (cf. minimum number of points in the
	           classical DBSCAN).
	
	Keyword arguments:
	weights -- weight array (if None, weights default to 1).
	noise   -- Boolean indicating whether objects that do not belong to any
	           cluster should be considered as noise (if True) or assigned to
	           clusters of their own (if False).
	"""
	n = dmatrix.shape[0]
	ematrix = dmatrix <= epsilon # Epsilon-reachability matrix
	if weights is None:
		weights = torch.ones(n, dtype=torch.int64, device=dmatrix.device) # Classical DBSCAN
	status = torch.zeros(n, dtype=torch.int64, device=dmatrix.device) # Unclassified = 0
	cluster_id = 1 # Classified = 1, 2, ...
	for i in range(n):
		if status[i] != 0:  # already clustered
			continue
		
		seeds = ematrix[i].clone()
		if weights[seeds].sum() < mu:  # no dense neighborhood
			status[i] = -1 # Noise = -1
			continue
		
		status[seeds] = cluster_id
		seeds[i] = False
		while torch.any(seeds):
			j = torch.argmax(seeds.float())
			eneighborhood = torch.nonzero(ematrix[j])[:, 0]
			seeds[j] = False
			if weights[eneighborhood].sum() < mu:
				continue

			seeds[eneighborhood[status[eneighborhood] == 0]] = True
			status[eneighborhood[status[eneighborhood] <= 0]] = cluster_id
			
		cluster_id += 1
	if not noise: # Assign cluster ids to noise
		noisy = (status == -1)
		status[noisy] = torch.arange(cluster_id, cluster_id + noisy.sum()).to(status)
	return status
