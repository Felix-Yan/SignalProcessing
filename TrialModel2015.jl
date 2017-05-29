
using DSP, WAV, PyPlot, AuditoryFilters

s, fs = wavread("E:\\WaveFiles\\supper.wav")

#set sample frequency to be 16kHz. The rate is desired frequency / original frequency.
#convert rate from float32 to float64
#rate = Float64(16000/fs)

#Change s from 2d array to 1d array
s = vec(s)
fs = Float64(fs)

#x = resample(s,rate)
#wavwrite(x,"E:\\WaveFiles\\resample.wav",Fs=fs)

varS = var(s)

#calculate decibel to power as the snr.
snr = db2pow(-15)

varN = varS/snr

n = randn(size(s))

n1 = sqrt(varN)*n

bank = make_erb_filterbank(fs,80,100)

#xft = filt(bank,x)
sft = filt(bank,s)

plot(1:1:80,vec(var(sft,1))) 

#sum elements in columns vertically
#newX=sum(xft,2)
newS=sum(sft,2)
var(newS)
println(var(newS))
println(var(s))
S2=newS/std(newS)
println(var(S2))
S2

wavwrite(newS,"E:\\WaveFiles\\supper2.wav",Fs=fs)
wavwrite(S2,"E:\\WaveFiles\\supper3.wav",Fs=fs)

function FIRfreqz(b::Array, w = linspace(0, π, 1024))
    n = length(w)
    h = Array{Complex64}(n)
    sw = 0
    for i = 1:n
      for j = 1:length(b)
        sw += b[j]*exp(-im*w[i])^-j
      end
      h[i] = sw
      sw = 0
    end
    return h
end

numChannel = 80
#hann = FIRWindow(hanning(500)/sqrt(sum(hanning(500).^2)))
hann = FIRWindow(rect(500)/sqrt(sum(rect(500).^2)))
#hann = FIRWindow(rect(500))
nyquist = fs/2
bandwidth = nyquist/numChannel
output = zeros(length(s), numChannel)
#initialize bandfilter
#bandfilter = zeros(500)

for k = 1:numChannel
    #don't want highFreq to exceed the nyquist frequency
    if k == numChannel
        highFreq = bandwidth*k-1
    else
        highFreq = 50+bandwidth*k
    end
    #print(string(k,","))
    #println(highFreq)
    bandpass = Bandpass(50+bandwidth*(k-1), highFreq; fs=fs)
    bandfilter = digitalfilter(bandpass,hann)
    output[:, k] = filt(bandfilter, s)    
    #H = freqz(bandfilter, ω)
end
#println(size(bandfilter))
w = linspace(0, pi, 1024)
bandpass = Bandpass(4000, 4150; fs=fs)
bandfilter = digitalfilter(bandpass,hann)
#h = FIRfreqz(bandfilter, w)
h = abs(fft(bandfilter))
h_db = (abs(h))
#ws = w/pi*(fs/2)
plot(1:length(h_db),h_db)
#plot(20*log10(abs(bandfilter)))
xlabel("Frequency [Hz]")
ylabel("Gain [dB]")
figure(2)
plot(bandfilter)

S3 = sum(output,2)

wavwrite(S3,"E:\\WaveFiles\\supper6.wav",Fs=fs)

plot(S3)
plot(s)

error = s-S3
pow2db(var(s)/var(error))

pow2db(var(s)/mean(abs(s-S3).^2))

pow2db(var(s)/mean(abs(s-newS).^2))

#f = 200
#range = -200: 1: 200
#h = sinc(f/(fs/2)*range)*(-1).^range
#y = conv(s,h)
#typeof(h)
#typeof(range)

lowpass = Lowpass(1000; fs=fs)
lowfilter = digitalfilter(lowpass,FIRWindow(rect(1024)))
o1 = filt(lowfilter, s)
w = linspace(0, pi, 1024)
h = FIRfreqz(lowfilter, w)
h_db = log10(abs(h))
ws = w/pi*(fs/2)
plot(ws,h_db)
figure(3)
plot(lowfilter)

highpass = Highpass(1000; fs=fs)
highfilter = digitalfilter(highpass,FIRWindow(rect(1025)))
o2 = filt(highfilter, s)
w = linspace(0, pi, 1024)
h = FIRfreqz(highfilter, w)
h_db = log10(abs(h))
ws = w/pi*(fs/2)
plot(ws,h_db)

o = o1+o2
pow2db(var(s)/mean(abs(s-o).^2))

wavwrite(o,"E:\\WaveFiles\\supperPass.wav",Fs=fs)

plot(s-o)

println(var(s))
println(var(o))

plot(s)

plot(o)

plot(s)
plot(o)

using Deconvolution

decon = wiener(o,s,s-o)

plot(decon)
plot(s)

pow2db(var(s)/mean(abs(s-decon).^2))
var(decon)
var(s)

wavwrite(decon,"E:\\WaveFiles\\supperDecon.wav",Fs=fs)
