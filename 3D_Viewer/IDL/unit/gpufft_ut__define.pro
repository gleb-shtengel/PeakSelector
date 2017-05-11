; docformat = 'rst'

function gpufft_ut::test_forward_float_function1
  compile_opt strictarr
  
  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
  
  a_gpu = gpuPutArr(a)
  
  c_gpu = gpuFFT(a_gpu)
  
  error = total(abs(fft(a) - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect FFT result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 6, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
  
  return, 1
end


;+
; Define instance variables.
;-
pro gpufft_ut__define
  compile_opt strictarr
  
  define = { gpufft_ut, inherits GPUutTestCase }
end