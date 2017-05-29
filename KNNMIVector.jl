using NearestNeighbors
#This is a function using K nearest neighbour to calculate mutual information between two vectors.
function KNNMIVector(k,V1,V2)
    k = 1
    X = V1
    N = V2
    #varX = 1
    #varN = 1
    #X = randn(number)*sqrt(varX)
    #N2 = randn(number)*sqrt(varN)
    #N = X+N2
    number = length(V1)
    data = zeros(2,number)
    data[1,:] = X
    data[2,:] = N
    #Chebyshev() distance takes the maximum of the coordinate difference.
    kdtreeData = KDTree(data,Chebyshev())
    dataX = zeros(2,number)
    dataX[1,:] = X
    dataN = zeros(2,number)
    dataN[1,:] = N
    kdtreeX = KDTree(dataX)
    kdtreeN = KDTree(dataN)
    distX = 0
    distN = 0
    #println(data)
    for i = 1: number
        point = data[:,i]
        pointX = dataX[:,i]
        pointN = dataN[:,i] 
        idxs, dists = knn(kdtreeData, point, k+1, true)
        idxsRangeX = inrange(kdtreeX, pointX, dists[2], true) 
        sizeX = length(idxsRangeX)
        idxsRangeN = inrange(kdtreeN, pointN, dists[2], true) 
        sizeN = length(idxsRangeN)
        distX += digamma(sizeX)
        distN += digamma(sizeN)
    end
    distX = distX/number
    distN = distN/number
    #println("distX: ",distX)
    #println("distN: ",distN)
    #println(digamma(number))
    MI = digamma(k) - distX - distN + digamma(number)
    return MI
end
