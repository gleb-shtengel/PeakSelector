; docformat = 'rst'

function gpucomplex_ut::test_float_function2
  compile_opt strictarr
  
  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  re = randomu(seed, n) 
  im = randomu(seed, n)   
  
  re_gpu = gpuPutArr(re)
  im_gpu = gpuPutArr(im)  
  
  complex_gpu = gpuComplex(re_gpu, im_gpu)
  
  re_gpu2 = gpuReal(complex_gpu)
  im_gpu2 = gpuImaginary(complex_gpu)
  
  error = total(abs(re - gpuGetarr(re_gpu2)))
  assert, error lt n * self.tolerance, $
          'incorrect real part result: error = ' + strtrim(error, 2) 

  error = total(abs(im - gpuGetarr(im_gpu2)))
  assert, error lt n * self.tolerance, $
          'incorrect imaginary part result: error = ' + strtrim(error, 2)

  assert, re_gpu2.type eq 4, 'incorrect type for real result'
  assert, im_gpu2.type eq 4, 'incorrect type for imaginary result'
                                 
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 5, 'memory leak'
  endif
  
  gpuFree, [re_gpu, im_gpu, complex_gpu, re_gpu2, im_gpu2]
  
  return, 1
end


;+
; Define instance variables.
;-
pro gpucomplex_ut__define
  compile_opt strictarr
  
  define = { gpucomplex_ut, inherits GPUutTestCase }
end