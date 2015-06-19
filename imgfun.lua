local P = {}
imgfun = P

function P.img2mat(img)
    return torch.reshape(img, 3, img:size()[2]*img:size()[3]):t()
end

function P.mat2img(mat, h, w)
    return torch.reshape(mat:t(), 3, h,w)
end

function P.assign_means(means, mat)
    -- means: K x d
    -- mat -- N x d
    -- returns N x 1 assignments
    local K = means:size()[1]
    local N = mat:size()[1]
    local kdists = torch.zeros(K, N)
    for i = 1,K do
        kdists[{i, {}}] = (means[{i, {}}]:repeatTensor(N,1)-mat):pow(2):sum(2):sqrt()
    end
    local mins, indices = kdists:min(1)
    return indices:t()      
end

function P.compute_means(clusters, mat, K)
    -- clusters: N x 1 cluster assignments for pmat, has entries from 1..K
    -- mat: N x d
    -- K: Number of clusters
    local N = mat:size()[1]
    local d = mat:size()[2]
    local means = torch.zeros(K, d)
    for i=1,K do
        local members = clusters:eq(i):reshape(N, 1)
        local num_points = members:sum()
        if num_points > 0 then
            means[{i, {}}] = members:repeatTensor(1, d):double():cmul(mat):sum(1) / num_points
        end
    end
    return means
end

function P.kmeans(mat, K, mini, maxi)
    local means = torch.randn(K, 3) * (maxi - mini) + mini
    local N = mat:size()[1]
    local d = mat:size()[2]

    local prev_clusters = torch.LongTensor(N, 1)
    local clusters = prev_clusters:clone()
    local iters = 0
    while true do
        iters = iters + 1
        clusters = P.assign_means(means, mat)
        local agreements = torch.eq(prev_clusters, clusters):sum()
        if agreements == N then
            break
        end
        means = P.compute_means(clusters, mat, K)
        prev_clusters:set(clusters)
    end
    print("Kmeans iterations: ", iters)
    return means, clusters
end

function P.gen_kmeans_image(img, K)
    local imat = P.img2mat(img)
    local N = imat:size()[1]
    local m, clusters = P.kmeans(imat, K, 0, 1)
    local img2 = torch.zeros(N, 3)
    local x1 = m[{clusters[1], {}}]
    local y1 = img2[{1, {}}]
    for i=1,N do
        img2[{i, {}}] = m[{clusters[i][1], {}}]
    end
    return P.mat2img(img2, img:size()[2], img:size()[3])
end

return P
