
using DSP, WAV, PyPlot

s, fs = wavread("E:\\WaveFiles\\supper.wav")
#Change s from 2d array to 1d array
s = vec(s)
#Change fs from float32 to float64
fs = Float64(fs)
B = var(s)
println(string("varS = ", B))
snr = db2pow(-20)
varN = B/snr
n = randn(size(s))
N = sqrt(varN)*n
numChannel = 80
bin = 400

function bandPass(s,fs,lowFreq,highFreq,bin)
    nyquist = fs/2
    range = -bin:1:bin
    hLowPass = sinc(highFreq/nyquist*range).*hanning(bin*2+1)
    hLowPass /= sqrt(sum(hLowPass.^2)/(highFreq/nyquist))
    hHighPass = sinc((1-lowFreq/nyquist)*range).*((-1.0).^range).*hanning(bin*2+1)
    hHighPass /= sqrt(sum(hHighPass.^2)/(1-lowFreq/nyquist))
    y = conv(s,hHighPass)
    y = y[bin+1:end-bin]
    y = conv(y,hLowPass)
    y = y[bin+1:end-bin]
    return y
end

#This makes N exist only from 0-1000 hz
bandN = bandPass(N,fs,0,1000,bin)
N = bandN

#this does multiple bandpass for s
function NaiveFiltering(s,fs,numChannel,bin)
    nyquist = fs/2
    bandwidth = nyquist/numChannel
    output = zeros(length(s), numChannel)
    #range = -bin:1:bin

    for k = 1 : numChannel
        #don't want highFreq to exceed the nyquist frequency
        lowFreq = 50+bandwidth*(k-1)
        if k == numChannel
            highFreq = bandwidth*k-1
        else
            highFreq = 50+bandwidth*k
        end
        #bandpass by combining highpass and lowpass
        #hLowPass = sinc(highFreq/nyquist*range).*hanning(bin*2+1)
        #hLowPass /= sqrt(sum(hLowPass.^2)/(highFreq/nyquist))
        #hHighPass = sinc((1-lowFreq/nyquist)*range).*((-1.0).^range).*hanning(bin*2+1)
        #hHighPass /= sqrt(sum(hHighPass.^2)/(1-lowFreq/nyquist))
        #y = conv(s,hHighPass)
        #y = y[bin+1:end-bin]
        #y = conv(y,hLowPass)
        #y = y[bin+1:end-bin]
        y = bandPass(s,fs,lowFreq,highFreq,bin)
        output[:, k] = y
    
        #h1 = abs(fft(hLowPass))
        #h_db1 = (abs(h1))
        #h2 = abs(fft(hHighPass))
        #h_db2 = (abs(h2))
        #plot(1:length(h_db1),h_db1)
        #plot(1:length(h_db2),h_db2)
    end
    return output
end

outputS = NaiveFiltering(s,fs,80,400)
varS = vec(var(outputS,1))
outputN = NaiveFiltering(N,fs,80,400)
varN = vec(var(outputN,1))

S = sum(outputS,2)
println(pow2db(var(s)/mean(abs(s-S).^2)))
plot(S)
plot(s)

Y = s+N
wavwrite(Y,"E:\\WaveFiles\\supperNoise.wav",Fs=fs)
wavwrite(S,"E:\\WaveFiles\\supperRecover.wav",Fs=fs)

#convex optimization of model 2015
R0k = 1
upper = 0
lower = -10000
λ = 0
powerSum = 0
diff = 100 #initialize the difference value between powerSum and B
b = zeros(numChannel)
step = 1
#B = sum(varS)
while abs(diff) > 0.001
#while step <= 100
    λ = (upper+lower)/2
    for k = 1:numChannel
        γ = 1/2*R0k^2*varS[k]*varN[k]+λ*varS[k]*varN[k]^2
        β = λ*varS[k]*(2-R0k^2)*varS[k]*varN[k]
        α = λ*varS[k]*(1-R0k^2)*varS[k]^2
        
        if α != 0
            b[k] = maximum([(-β+sqrt(β^2-4*α*γ))/(2*α), (-β-sqrt(β^2-4*α*γ))/(2*α)])
        else
            b[k] = -γ/β
        end
        if b[k] < 0
            b[k] = 0
        end
    end
    powerSum = var(outputS*sqrt(b))
    diff = powerSum - B

    if diff > 0
        upper = (upper+lower)/2
    else
        lower = (upper+lower)/2
    end
    println(string("step: ",step))
    println(diff)
    println(string("λ = ",λ))
    step += 1
end
println(string("λ = ",λ))
b

newZ = outputS*sqrt(b)+N
newY = outputS*sqrt(b)
#abs(var(newY)-var(sum(outputS,1)))
abs(var(newY)-B)

wavwrite(newY,"E:\\WaveFiles\\supperJustEnhanced.wav",Fs=fs)
wavwrite(newZ,"E:\\WaveFiles\\supperEnhancedPLUS.wav",Fs=fs)

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

figure(1)
plot_spectrogram(s, fs)
figure(2)
plot_spectrogram(Y, fs)
figure(3)
plot_spectrogram(newZ, fs)
figure(4)
plot_spectrogram(newY, fs)

plot(s)

lowpass = Lowpass(4000;fs=fs)
filter = digitalfilter(lowpass, FIRWindow(hanning(64)))
s1 = filt(filter,s)
plot(s1[(32:232)])
plot(s[(1:200)])
#ω = 0:0.01:pi # variables can have Unicode names!
              # This is typed in the notebook as \omega + tab.
#testFilter = digitalfilter(responsetype,FIRWindow(hanning(500)))

#this transforms a vector to normal distribution
using StatsBase,StatsFuns
f1 = ecdf(s)
sizeS = length(s)
newV1 = zeros(sizeS)
max = maximum(s)
for i = 1:sizeS
    u = 0
    #make sure there is no infinity in newV
    if(s[i] == max)
        u = 0.99999999
    else
        u = f1(s[i])
    end
    newV1[i] = norminvcdf(u)
end
newV1 = newV1.*sqrt(B)

h = PyPlot.plt[:hist](newV1,1000)
var(newV1)
