; docformat = 'rst'

function gpuwhere_ut::test_float_function2
  compile_opt strictarr
  
  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n) 
  b = fltarr(n) + 0.5
  
  a_gpu = gpuPutarr(a)
  b_gpu = gpuPutarr(b)
  
  ind_gpu = gpuWhere(gpuGt(a_gpu, b_gpu), count_gpu)
  
  ind = where(a gt b, count)
  
  assert, count_gpu eq count, 'incorrect number of indices in result'
  
  error = total(abs(ind - gpuGetarr(ind_gpu)))
  assert, error eq 0.0, $
          'incorrect total result: error = ' + strtrim(error, 2)            
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, ind_gpu]
  
  return, 1
end


;+
; Define instance variables.
;-
pro gpuwhere_ut__define
  compile_opt strictarr
  
  define = { gpuwhere_ut, inherits GPUutTestCase }
end