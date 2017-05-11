;
; example for comparing gpu lib performance with plain IDL
;

; initialize the library
gpuinit 


nx = 1000000L   ; vector length
niter = 10      ; number of iterations of kernel

s = 0
x = randomu(s, nx)


; create gpu variables.
x_gpu = gpuFltArr(nx)
res_gpu = gpuFltarr(nx)
gpuPutArr, x, x_gpu

; execture kernel once, in order to make sure
; it actually is loaded onto GPU prior to timing
gpuLGamma, x_gpu, res_gpu

; --- IDL test
t = systime(2)
for i = 0, niter do er = lngamma(x)
cputime = systime(2) - t

; print out some results
print, er[0:5]
 
; ---- GPU test
t = systime(2)    

gpuPutArr, x, x_gpu
for i=0, niter do  gpuLGamma, x_gpu, res_gpu
gpuGetArr,res_gpu, x

gputime = systime(2)  - t
print, x[0:5]

print, 'CPU Time = ', cputime
print, 'GPU Time = ', gputime
print, 'Speedup  = ', cputime/gputime

;exit
