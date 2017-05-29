
using NearestNeighbors, PyPlot,StatsBase,StatsFuns, DSP, WAV, Distances

s, fs = wavread("E:\\WaveFiles\\11025\\questions_money.wav")
#Change s from 2d array to 1d array
s = vec(s)
#Change fs from float32 to float64
fs = Float64(fs)
B = var(s)
println(string("varS = ", B))
snr = db2pow(-5)
varN = B/snr
n = randn(size(s))
N = sqrt(varN)*n
println(string("varN = ", varN))
#Y is the received speech signal
Y = s+N

fs

#this transforms a vector to normal distribution
f1 = ecdf(s)
sizeS = length(s)
newV1 = zeros(sizeS)
max = maximum(s)
for i = 1:sizeS
    u = 0
    #make sure there is no infinity in newV
    if(s[i] == max)
        u = 0.9999
    else
        u = f1(s[i])
    end
    newV1[i] = norminvcdf(u)
end
newV1 = newV1.*sqrt(B)

Y1 = newV1+N
wavwrite(Y,"E:\\WaveFiles\\supperNoise.wav",Fs=fs)
wavwrite(Y1,"E:\\WaveFiles\\supperNoise1.wav",Fs=fs)

#no longer used
number = 1000
varLogM = 0.16
varLogP = 0.04
logM = randn(number)*sqrt(varLogM)
logP = randn(number)*sqrt(varLogP)
M = e.^logM
logX = logM+logP
varLogN = 1
logN = randn(number)*sqrt(varLogN)
X = e.^logX
N = e.^logN
logY = log(X+N)
Y = e.^logY
var(logX)

function KNN(k,M1,M2)
    k = 1
    X = M1
    N = M2
    #varX = 1
    #varN = 1
    #X = randn(number)*sqrt(varX)
    #N2 = randn(number)*sqrt(varN)
    #N = X+N2
    
    number = size(M1,2)
    #rows = size(M1,1)
    #data = zeros(rows,number)
    #data[1,:] = X
    #data[2,:] = N
    
    #Chebyshev() distance takes the maximum of the coordinate difference.
    
    #kdtreeData = KDTree(data,Chebyshev())
    #dataX = zeros(2,number)
    #dataX[1,:] = X
    #dataN = zeros(2,number)
    #dataN[1,:] = N
    
    kdtreeX = KDTree(X)
    kdtreeN = KDTree(N)
    #sum total of X and N distances
    distX = 0
    distN = 0
    #println(data)
    for i = 1:number
        #point = data[:,i]
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
    #println("distX: ",distX)
    #println("distN: ",distN)
    #println(digamma(number))
    MI = digamma(k) - distX - distN + digamma(number)
    return MI
end

a = [1, 2, 3]
b = [0, 4.0, 2]
sort(b)

MI1 = KNN(1,Y,s)

MI2 = KNN(1,Y1,s)

S = spectrogram(s[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)

SVec = power(S)

NSpec = spectrogram(N[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)

NVec = power(NSpec)

YSpec = spectrogram(Y[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)
YVec = power(YSpec)

MI = KNN(1,SVec,NVec)

MI = KNN(1,SVec,YVec)

-0.5*log(1-cor(Y,s)^2)

include("E:\\workspace-julia\\KNNMIVector.jl")
MI = KNNMIVector(1,logY,logM)

include("E:\\workspace-julia\\KNNMI.jl")
dataX = zeros(2,number)
dataX[1,:] = logY
dataN = zeros(2,number)
dataN[1,:] = logM
KNNMI(1,dataX,dataN)

#this transforms a vector to normal distribution
f1 = ecdf(logY)
size = length(logY)
newV1 = zeros(size)
max = maximum(logY)
for i = 1:size
    u = 0
    #make sure there is no infinity in newV
    if(logY[i] == max)
        u = 0.9999
    else
        u = f1(logY[i])
    end
    newV1[i] = norminvcdf(u)
end


MI = KNN(1,newV1,logM)

MI = KNN(1,logX,logM)

#this transforms a vector to beta distribution
f2 = ecdf(logY)
size = length(logY)
newV2 = zeros(size)
max = maximum(logY)
for i = 1:size
    u = 0
    #make sure there is no infinity in newV
    if(logY[i] == max)
        u = 0.9999
    else
        u = f1(logY[i])
    end
    newV2[i] = betainvcdf(0.5,0.5,u)
end
MI = KNN(1,newV2,logM)

#this transforms a vector to chisquare distribution
f3 = ecdf(logY)
size = length(logY)
newV3 = zeros(size)
max = maximum(logY)
for i = 1:size
    u = 0
    #make sure there is no infinity in newV
    if(logY[i] == max)
        u = 0.9999
    else
        u = f1(logY[i])
    end
    newV3[i] = chisqinvcdf(2,u)
end
MI = KNN(1,newV3,logM)

#this transforms a vector to f distribution
f4 = ecdf(logY)
size = length(logY)
newV4 = zeros(size)
max = maximum(logY)
for i = 1:size
    u = 0
    #make sure there is no infinity in newV
    if(logY[i] == max)
        u = 0.9999
    else
        u = f1(logY[i])
    end
    newV4[i] = fdistinvcdf(2,2,u)
end
MI = KNN(1,newV4,logM)

#this transforms a vector to student t distribution
f5 = ecdf(logY)
size = length(logY)
newV5 = zeros(size)
max = maximum(logY)
for i = 1:size
    u = 0
    #make sure there is no infinity in newV
    if(logY[i] == max)
        u = 0.9999
    else
        u = f1(logY[i])
    end
    newV5[i] = tdistinvcdf(2,u)
end
MI = KNN(1,newV5,logM)

function laplaceinvcdf(b,q)



end

meanX = mean(X)
varX = var(X)
meanN = mean(N)
varN = var(N)
D2 = varX + varN
miu = meanX + meanN
#meanLogY = 2*log(miu)-0.5*log(D2)
#varLogY = log(D2)-2*log(miu)
varLogY = log(D2/(miu^2)+1) 
meanLogY = log(miu) - 0.5*varLogY
println(varLogY-var(logY))
println(meanLogY-mean(logY))

#0.5*log(1+varLogM/varLogY) 
estimate = -0.5*log(1-cor(logY,logM)^2)
#cov(Y,M)/sqrt(varY)/sqrt(varM) is equal to cor(Y,M)
#sum((logY-mean(logY)).*(logM-mean(logM)))/length(logY) is equal to cov(logY,logM)

MI-estimate

-0.5*log(2*pi*e*(1-cor(logX,logM)^2)*varLogN*var(logX))

Hy=0.5*log(2*pi*e*var(logY))

0.5*log((2*pi*e)^2*(1-cor(logX,logM)^2)*varLogN*var(logX))

Ixm = -0.5*log(1-cor(logX,logM)^2)

hx = 0.5*log(2*pi*e*var(logX))

hn = 0.5*log(2*pi*e*varLogN)

HyGm = hx+hn-Ixm

cor(logY,logM)
#cov(logY,logM)/sqrt(var(logY))/sqrt(var(logM))
#cor(logY,logX)*cor(logX,logM)

varLogM = var(logM)
#sum((logY-meanLogY).*(logM-mean(logM)))/length(logY)

correlation = sum((logY-meanLogY).*(logM-mean(logM)))/length(logY)/sqrt(varLogY)/sqrt(varLogM)
-0.5*log(1-correlation^2)
#cov(logY,logM)
#cor(logY,logM)
#correlation

-0.5*log(1-cor(logX,logM)^2)

h = PyPlot.plt[:hist](logY,100)

h = PyPlot.plt[:hist](newV,100)


#test KNNMI
rows = 200
columns = 500

srand(1)
X = rand(Normal(0,0.2),rows)
for i = 1:columns
    c2 = rand(Normal(0,0.2),rows)
    X = cat(2,X,c2)
end

srand(2)
N = rand(Normal(0,1),rows)
for i = 1:columns
    c2 = rand(Normal(0,1),rows)
    N = cat(2,N,c2)
end
Y = X+N
R_y = cov(Y)
R_n = cov(N)
R_x = cov(X)

det(R_y)

det(R_n)

I = 0.5 * log(det(R_y)/det(R_n))










