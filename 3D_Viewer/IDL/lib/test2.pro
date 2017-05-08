gpuinit
nx = 1000000L
niter = 1
s = 0
x = randomu(s, nx)

x_gpu = gpuFltArr(nx)
res_gpu = gpuFltarr(nx)
gpuPutArr, x, x_gpu

gpuLGamma, x_gpu, res_gpu

t = systime(2)
for i = 0, niter do er = lngamma(x)
cputime = systime(2) - t
print, er[0:5]
 
er_gold = er

t = systime(2) 
for i=0, niter do  gpuLGamma, x_gpu, res_gpu
gputime = systime(2)  - t

gpuGetArr,res_gpu, x
print, x[0:5]

print, 'CPU Time = ', cputime
print, 'GPU Time = ', gputime
print, 'Speedup  = ', cputime/gputime
