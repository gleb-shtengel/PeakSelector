; docformat = 'rst'


function gputotal_ut::test_float_function1
  compile_opt strictarr
  
  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n) 
  
  a_gpu = gpuPutArr(a)
  
  t = gpuTotal(a_gpu)
  
  error = total(abs(total(a) - t))
  assert, error lt n * self.tolerance, $
          'incorrect total result: error = ' + strtrim(error, 2)            
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 1, 'memory leak'
  endif
  
  gpuFree, [a_gpu]
  
  return, 1
end


function gputotal_ut::test_cumulative_function1
  compile_opt strictarr
  
  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n) 
  
  a_gpu = gpuPutArr(a)
  
  t_gpu = gpuTotal(a_gpu, /cumulative)
  
  error = total(abs(total(a, /cumulative) - gpuGetarr(t_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect total result: error = ' + strtrim(error, 2)            
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, t_gpu]
  
  return, 1
end


;+
; Define instance variables.
;-
pro gputotal_ut__define
  compile_opt strictarr
  
  define = { gputotal_ut, inherits GPUutTestCase }
end