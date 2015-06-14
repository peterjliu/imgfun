import numpy as np
def Img2Mat(img):
    """Reshapes an RGB image with dimension(r, c, 3) to a matrix with dimension [r*c, 3].
    
    Each row of the matrix corresponds to a pixel in row-major order."""
    return np.reshape(img, (img.shape[0]*img.shape[1], 3))

def Mat2Img(m, r, c):
    """Inverse of Img2Matrix, with number of rows and columns specified."""
    return np.reshape(m, (r,c,3))

def AssignMeans(means, pmat):
    """Compute nearest means

    Args:
        means: K x d
        pmat: N x d
    
    Returns:
        N x 1 vector with indices of nearest means
    """
    K = means.shape[0]
    N = pmat.shape[0]
    kdists = np.zeros((K, N))
    for i in range(0, K):
        kdists[i, :] = np.sqrt(np.sum(np.square(np.tile(means[i, :], (N,1)) - pmat), axis=1))
    a = np.argmin(kdists, axis=0)
    return a

def ComputeMeans(clusters, pmat, K):
    """Compute means given cluster assignments
    
    Args:
        clusters: N x 1 cluster assignments for pmat, has entries from 0..K-1
        pmat: N x d
        K: number of clusters
        
    Returns:
        K cluster centers, K x d, where i's center is in row i-1"""
    [N, d] = pmat.shape
    means = np.zeros((K, d))
    for i in range(0, K):
        members = np.reshape(clusters == i, (N, 1))
        num_points = np.sum(members)
        if num_points > 0:
            means[i, :] = np.sum(np.multiply(np.tile(members, (1, d)),
                                 pmat), axis=0) / num_points
    return means

def Kmeans(pmat, K, mini, maxi):
    """Run K means on data
    
    The K means are initialized uniformly in [mini, maxi]
    
    Args:
        K: number of means
        pmat: data with N data points and d dimensions, N x d numpy array
        
    Returns:
        means: K x d numpy array representing the K means
        clusters: N x 1, cluster assignments, indexed from 0 to K-1"""
    means = np.random.rand(K, 3) * (maxi - mini) + mini
    [N, d] = pmat.shape
    clusters = np.zeros((N))
    prev_clusters = np.zeros((N))
    iterations = 0
    while True:
        iterations += 1
        clusters = AssignMeans(means, pmat)
        agreements = np.sum(np.equal(prev_clusters, clusters))
        if agreements == N:
            # This ends because K-means is guaranteed to converge
            break
        means = ComputeMeans(clusters, pmat, K)
        prev_clusters = np.copy(clusters)
    return [means, clusters]
    
def GenKmeansImage(img, K):
    """Generates an rgb-image by performing K-means on img
    
    Args:
        img: numpy image
        K: number of clusters
    Returns:
        New image"""
    imat = Img2Mat(img)
    N = imat.shape[0]
    [means, clusters] = Kmeans(imat, K, 0, 255)
    newimg = np.zeros((N, 3))
    for i in range(N):
        newimg[i, :] = means[clusters[i], :]
    return Mat2Img(newimg, img.shape[0], img.shape[1]).astype("uint8")
