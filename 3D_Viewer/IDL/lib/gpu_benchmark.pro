; docformat = 'rst'

;+
; Benchmark routine that indicates the device it is running on and its 
; speedup over regular IDL code.
;
; :Params:
;    deviceId : in, optional, type=long
;       device ID
;-
pro gpu_benchmark, deviceId
  compile_opt strictarr, hidden
  
  gpuinit, deviceId

  if (!gpu.mode ne 0) then begin
    err = cudaGetDeviceProperties(prop, !gpu.device)
    name = prop.name
  endif else begin
    name = 'Pure IDL Emulation'
  endelse
   
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
   
  ; ---- GPU test
  t = systime(2)    
  
  gpuPutArr, x, x_gpu
  for i=0, niter do  gpuLGamma, x_gpu, res_gpu
  gpuGetArr,res_gpu, x
  
  gputime = systime(2)  - t

  print, name
  print, 'GPU time: ' + strtrim(gputime, 2) + ' seconds'
  print, 'Speedup: ' + strtrim(cputime / gputime, 2) + 'x' 
end
