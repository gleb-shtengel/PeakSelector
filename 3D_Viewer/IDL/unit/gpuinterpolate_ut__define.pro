; docformat = 'rst'


function gpuinterpolate_ut::test_float_function3
  compile_opt strictarr
  
  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  p = findgen(4, 4)
  x = [0.5, 1.5, 2.5]
  y = [0.5, 1.5, 2.5]
  
  p_gpu = gpuPutArr(p)
  x_gpu = gpuPutArr(x)
  y_gpu = gpuPutArr(y)
  
  interp_gpu = gpuInterpolate(p_gpu, x_gpu, y_gpu)
  
  error = total(abs(interpolate(p, x, y) - gpuGetArr(interp_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect interpolate result: error = ' + strtrim(error, 2)            
  assert, interp_gpu.type eq 4, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 4, 'memory leak'
  endif
  
  gpuFree, [p_gpu, x_gpu, y_gpu, interp_gpu]
  
  return, 1
end


;+
; Define instance variables.
;-
pro gpuinterpolate_ut__define
  compile_opt strictarr
  
  define = { gpuinterpolate_ut, inherits GPUutTestCase }
end