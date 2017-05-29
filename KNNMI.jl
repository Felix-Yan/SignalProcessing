
using Distances
#This is a function using K nearest neighbour to calculate mutual information between two matrices.
#rows indicate dimensions.
function KNNMI(k,M1,M2)
    X=M1
    N=M2
    
    number = size(M1,2)

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
        
        #print("i=",i,",")
        #print(distMin,",")
        
        sizeX = count(j->j<=distMin,distancesX)
        sizeN = count(j->j<=distMin,distancesN)
            
        #print(sizeX,",")
        #println(sizeN)
        
        distX += digamma(sizeX)
        distN += digamma(sizeN)
    end
    distX = distX/number
    distN = distN/number
   
    MI = digamma(k) - distX - distN + digamma(number)
    return MI
end
