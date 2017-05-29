
using StatsBase,StatsFuns, DSP, WAV, PyPlot, Distributions

include("E:\\workspace-julia\\KNNMI.jl")
include("E:\\workspace-julia\\KNNMIVector.jl")

#this transforms a vector s to normal distribution, with variance var
function normalTransform(s,var)
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
    newV1 = newV1.*sqrt(var)
end

# PyPlot includes a specgram function, but let's use the native
# implementation from DSP.jl. The function below extracts a
# spectrogram with standard parameters for speech (25ms Hanning windows
# and 10ms overlap), then plots it and returns it .

function plot_spectrogram(s, fs)
    S = spectrogram(s[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)
    t = time(S)
    f = freq(S)
    imshow(flipdim(log10(power(S)),1), extent=[first(t), last(t),
             fs*first(f), fs*last(f)], aspect="auto")
    S
end

srand(10)
varX = 0.2
b = sqrt(varX/2)
X = rand(Laplace(0,b), 100000)
snr = db2pow(-5)
varN = varX/snr
n = randn(size(X))
N = sqrt(varN)*n
println(string("varN = ", varN))
#Y is the received speech signal
Y = X+N

fs = 16000
XSpec = spectrogram(X[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)
XVec = power(XSpec)
YSpec = spectrogram(Y[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)
YVec = power(YSpec)

#only use the first two rows as they are independent.
XM = XVec[1:2,:]
YM = YVec[1:2,:]

MI = KNNMI(1,XM,YM)

newX = normalTransform(X,varX)

newY = newX+N

fs = 16000
XSpec2 = spectrogram(newX[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)
XVec2 = power(XSpec2)
YSpec2 = spectrogram(newY[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)
YVec2 = power(YSpec2)

XM2 = XVec2[1:2,:]
YM2 = YVec2[1:2,:]

MI2 = KNNMI(1,YM2,XM2)

MIVector = KNNMIVector(1,Y,X)

MIVector2 = KNNMIVector(1,newY,newX)

h = PyPlot.plt[:hist](laplace,100)

s, fs = wavread("E:\\WaveFiles\\16000\\supper.wav")
#Change s from 2d array to 1d array
s = vec(s)
#Change fs from float32 to float64
fs = Float64(fs)
B = var(s)
println(string("varS = ", B))
snr = db2pow(-8)
varN = B/snr
n = randn(size(s))
N = sqrt(varN)*n
println(string("varN = ", varN))
#Y is the received speech signal
Y = s+N

SSpec = spectrogram(s[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)
SVec = power(SSpec)
YSpec = spectrogram(Y[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)
YVec = power(YSpec)



MI = KNNMI(1,YVec,SVec)

newS = normalTransform(s,B)
newY = newS+N

SSpec2 = spectrogram(newS[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)
SVec2 = power(SSpec2)
YSpec2 = spectrogram(newY[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)
YVec2 = power(YSpec2)

MI2 = KNNMI(1,YVec2,SVec2)

MI2 = KNNMIVector(1,Y,s)

# not always used
number = 10000
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

h = PyPlot.plt[:hist](s,100)

S = spectrogram(s[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)
SMatrix = power(S)
YSpec = spectrogram(Y[:,1], floor(Int, 25e-3*fs),floor(Int, 10e-3*fs); window=hanning)
YMatrix = power(YSpec)

# this sums up the mutual information
cols = size(SMatrix,2)
MI = 0 #the sum of mutual information
for i = 1:cols
    mi = KNNMI(1,YMatrix[:,i],SMatrix[:,i])
    MI += mi
end
MI

#plot to see the shape
h = PyPlot.plt[:hist](newX,100)

wavwrite(newV1,"E:\\WaveFiles\\supperDistTran.wav",Fs=fs)

Y1 = newV1+N
wavwrite(Y,"E:\\WaveFiles\\supperNoise.wav",Fs=fs)
wavwrite(Y1,"E:\\WaveFiles\\supperNoise1.wav",Fs=fs)

rows = 10000
columns = 2
varX = 1
varN = 2

srand(1)
X = rand(Normal(0,varX),rows)
for i = 1:columns
    c2 = rand(Normal(0,varX),rows)
    X = cat(2,X,c2)
end

srand(20)
N = rand(Normal(0,varN),rows)
for i = 1:columns
    c2 = rand(Normal(0,varN),rows)
    N = cat(2,N,c2)
end
Y = X+N
R_y = cov(Y)
R_n = cov(N)
R_x = cov(X)

det(R_y)

det(R_n)

R_n

I = 0.5 * log(det(R_y)/det(R_n))

I2 =0.5 * log(det(R_x+R_n)/det(R_n))

using NearestNeighbors, Distances
#This is a function using K nearest neighbour to calculate mutual information between two matrices.
function KNNMI2(k,M1,M2)
    k = 1
    #standardize both X and N
    #X = M1/sqrt(var(M1))
    #N = M2/sqrt(var(M2))
    X=M1
    N=M2
    
    number = size(M1,2)
   
    #kdtreeX = KDTree(X)
    #kdtreeN = KDTree(N)
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
       
        #idxsRangeX = inrange(kdtreeX, pointX, distMin, true)
        #sizeX = length(idxsRangeX)

        #idxsRangeN = inrange(kdtreeN, pointN, distMin, true)
        #sizeN = length(idxsRangeN)
        
        print(sizeX,",")
        println(sizeN)
        
        distX += digamma(sizeX)
        distN += digamma(sizeN)
    end
    distX = distX/number
    distN = distN/number
   
    MI = digamma(k) - distX - distN + digamma(number)
    return MI
end

MI = KNNMI(1,transpose(X),transpose(Y))
