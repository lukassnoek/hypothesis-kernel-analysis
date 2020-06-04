"""
N, P = X.shape
K = len(self.labels)
self.dist = np.zeros((N, K))
for i, (_, th_vec) in enumerate(self.th_vec.items()):
    if kernel.ndim > 1:
        dist = np.zeros((K, N))
        for i in range(K):
            # IDEA: use different distance metrics
            tmp = (X - kernel[i, :]) ** 2
            if self.normalize:
                # Normalize by kernel sq sum
                tmp /= np.sum(kernel[i, :])  # should I square this?
            
            # Compute euclidean distance
            dist[i, :] = np.sqrt(np.sum(tmp, axis=1))

        # Take max (or mean?) of different options
        dist = np.max(dist,  axis=0)
    else:
        dist = (X - kernel) ** 2
        if self.normalize:
            dist /= np.sum(kernel)  # should I square this?
        
        dist = np.sqrt(np.sum(dist, axis=1))
    
    self.dist[:, i] = dist

if self.scale_dist:
    minim = self.dist.min(axis=1, keepdims=True)
    maxim = self.dist.max(axis=1)
    rnge =  maxim - np.squeeze(minim)
    self.dist = (self.dist - minim) / rnge[:, np.newaxis]

"""