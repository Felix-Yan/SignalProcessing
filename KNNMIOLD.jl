
using NearestNeighbors, Distances
#This is a function using K nearest neighbour to calculate mutual information between two matrices.
function KNNMI(k,M1,M2)
    k = 1
    X = M1
    N = M2
    
    number = size(M1,2)
   
    kdtreeX = KDTree(X)
    kdtreeN = KDTree(N)
    #sum total of X and N distances
    distX = 0
    distN = 0
    
    for i = 1:number
        pointX = X[:,i]
        pointN = N[:,i] 
        distancesX = colwise(Euclidean(), X , pointX)
        distancesX[i] = Inf
        distancesN = colwise(Euclidean(), N, pointN)
        distancesN[i] = Inf
        distances = max(distancesX,distancesN)
        sortedDis = sort(distances)
        distMin = sortedDis[k]
       
        idxsRangeX = inrange(kdtreeX, pointX, distMin, true)
        sizeX = length(idxsRangeX)

        idxsRangeN = inrange(kdtreeN, pointN, distMin, true)
        sizeN = length(idxsRangeN)
        distX += digamma(sizeX)
        distN += digamma(sizeN)
    end
    distX = distX/number
    distN = distN/number
   
    MI = digamma(k) - distX - distN + digamma(number)
    return MI
end
