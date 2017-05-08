; docformat = 'rst'


function gpuSqrt_ut::test_float_function2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
  
  a_gpu = gpuPutArr(a)
  
  c_gpu = gpuSqrt(a)
  
  error = total(abs(sqrt(a) - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 4, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
  
  return, 1
end


function gpuSqrt_ut::test_double_function1
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = double(randomu(seed, n))
    
  a_gpu = gpuPutArr(a)
    
  c_gpu = gpuSqrt(a)
  
  error = total(abs(sqrt(a) - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 5, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
  
  return, 1
end


function gpuSqrt_ut::test_complex_function1
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = complex(randomu(seed, n), randomu(seed, n))
    
  a_gpu = gpuPutArr(a)
    
  c_gpu = gpuSqrt(a_gpu)
  
  error = total(abs(sqrt(a) - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 6, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
  
  return, 1
end


function gpuSqrt_ut::test_dcomplex_function1
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = dcomplex(randomu(seed, n), randomu(seed, n))
  
  a_gpu = gpuPutArr(a)
  
  c_gpu = gpuSqrt(a)
  
  error = total(abs(sqrt(a) - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 9, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
  
  return, 1
end


function gpuSqrt_ut::test_float_function5
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
    
  a_gpu = gpuPutArr(a)
  
  c_gpu = gpuSqrt(2., 3., a, 4., 5.)
  
  error = total(abs(2. * sqrt(3. * a + 4.) + 5. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect addition result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 4, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
  
  return, 1
end


function gpuSqrt_ut::test_double_function5
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = double(randomu(seed, n))
  
  a_gpu = gpuPutArr(a)
  
  c_gpu = gpuSqrt(2., 3., a, 4., 5.)
  
  error = total(abs(2. * sqrt(3. * a + 4.) + 5. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 5, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
  
  return, 1
end


function gpuSqrt_ut::test_complex_function5
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = complex(randomu(seed, n), randomu(seed, n))
  
  a_gpu = gpuPutArr(a)
    
  c_gpu = gpuSqrt(2., 3., a, 4., 5.)
  
  error = total(abs(2. * sqrt(3. * a + 4.) + 5. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 6, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
  
  return, 1
end


function gpuSqrt_ut::test_dcomplex_function5
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = dcomplex(randomu(seed, n), randomu(seed, n))
    
  a_gpu = gpuPutArr(a)
  
  c_gpu = gpuSqrt(2., 3., a_gpu, 4., 5.)
  
  error = total(abs(2. * sqrt(3. * a + 4.) + 5. - gpuGetArr(c_gpu))) 
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 9, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
  
  return, 1
end


function gpuSqrt_ut::test_lhs_function1
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
  b = randomu(seed, n)  
  
  a_gpu = gpuPutArr(a)
  c_gpu = gpuMake_array(n)
  c_gpu = gpuSqrt(a, lhs=c_gpu)
  
  error = total(abs(sqrt(a) - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
  
  return, 1
end


function gpuSqrt_ut::test_compound_function1
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
    
  a_gpu = gpuPutArr(a)
    
  c_gpu = gpuSqrt(gpuSqrt(a_gpu))
  
  error = total(abs(sqrt(sqrt(a)) - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
  
  return, 1
end


function gpuSqrt_ut::test_convert_function1
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
    
  c_gpu = gpuSqrt(gpuSqrt(a))
  
  error = total(abs(sqrt(sqrt(a)) - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect addition result: error = ' + strtrim(error, 2)            
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 1, 'memory leak'
  endif
  
  gpuFree, [c_gpu]
  
  return, 1
end


function gpuSqrt_ut::test_incorrectparams_procedure1
  compile_opt strictarr

  catch, error
  if (error ne 0L) then begin
    catch, /cancel
    
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers, 'memory leak'
    
    return, 1
  endif
  
  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
  
  gpuSqrt, a
  
  assert, 0, 'should have failed'
  
  return, 1
end


function gpuSqrt_ut::test_incorrectparams_procedure4
  compile_opt strictarr

  catch, error
  if (error ne 0L) then begin
    catch, /cancel
    
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers, 'memory leak'
    
    return, 1
  endif
  
  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
  b = randomu(seed, n)
  
  gpuSqrt, 2., a, 3., b
  
  assert, 0, 'should have failed'
  
  return, 1
end


function gpuSqrt_ut::test_float_procedure2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = randomu(seed, n)
    
  a_gpu = gpuPutArr(a)
    
  gpuSqrt, a_gpu, b_gpu
  
  error = total(abs(sqrt(a) - gpuGetArr(b_gpu))) 
  assert, error lt n * self.tolerance, $
          'incorrect addition result: error = ' + strtrim(error, 2)            
  assert, b_gpu.type eq 4, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu]
    
  return, 1
end


function gpuSqrt_ut::test_double_procedure2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = double(randomu(seed, n))
    
  a_gpu = gpuPutArr(a)
    
  gpuSqrt, a_gpu, b_gpu
  
  error = total(abs(sqrt(a) - gpuGetArr(b_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  assert, b_gpu.type eq 5, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu]
    
  return, 1
end


function gpuSqrt_ut::test_complex_procedure2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = complex(randomu(seed, n), randomu(seed, n))
  
  a_gpu = gpuPutArr(a)
  
  gpuSqrt, a_gpu, c_gpu
  
  error = total(abs(sqrt(a) - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 6, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
    
  return, 1
end


function gpuSqrt_ut::test_dcomplex_procedure2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = dcomplex(randomu(seed, n), randomu(seed, n))
    
  a_gpu = gpuPutArr(a)
    
  gpuSqrt, a_gpu, b_gpu
  
  error = total(abs(sqrt(a) - gpuGetArr(b_gpu))) 
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  assert, b_gpu.type eq 9, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu]
    
  return, 1
end


function gpuSqrt_ut::test_float_procedure6
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = randomu(seed, n)
  
  a_gpu = gpuPutArr(a)
  
  gpuSqrt, 2., 3., a_gpu, 4., 5., c_gpu
  
  error = total(abs(2. * sqrt(3. * a + 4.) + 5. - gpuGetArr(c_gpu))) 
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 4, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
    
  return, 1
end


function gpuSqrt_ut::test_double_procedure6
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = double(randomu(seed, n))
    
  a_gpu = gpuPutArr(a)
    
  gpuSqrt, 2., 3., a_gpu, 4., 5., c_gpu
  
  error =  total(abs(2. * sqrt(3. * a + 4.) + 5. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 5, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
    
  return, 1
end


function gpuSqrt_ut::test_complex_procedure6
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = complex(randomu(seed, n), randomu(seed, n))
    
  a_gpu = gpuPutArr(a)
  
  gpuSqrt, 2., 3., a_gpu, 4., 5., c_gpu
  
  error = total(abs(2. * sqrt(3. * a + 4.) + 5. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect addition result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 6, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
    
  return, 1
end


function gpuSqrt_ut::test_dcomplex_procedure6
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = dcomplex(randomu(seed, n), randomu(seed, n))
  
  a_gpu = gpuPutArr(a)
  
  gpuSqrt, 2., 3., a_gpu, 4., 5., c_gpu
  
  error = total(abs(2. * sqrt(3. * a + 4.) + 5. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 9, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, c_gpu]
    
  return, 1
end


function gpuSqrt_ut::test_compound_procedure2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = randomu(seed, n)
    
  a_gpu = gpuPutArr(a)
  
  gpuSqrt, gpuSqrt(a_gpu), b_gpu
  
  error = total(abs(sqrt(sqrt(a)) - gpuGetArr(b_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect addition result: error = ' + strtrim(error, 2)            
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 2, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu]
    
  return, 1
end


function gpuSqrt_ut::test_convert_procedure2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = randomu(seed, n) 
  
  gpuSqrt, a, c_gpu
  
  error = total(abs(sqrt(a) - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect square root result: error = ' + strtrim(error, 2)            
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 1, 'memory leak'
  endif
  
  gpuFree, [c_gpu]
    
  return, 1
end


function gpuSqrt_ut::test_incorrectparams_procedure3
  compile_opt strictarr

  catch, error
  if (error ne 0L) then begin
    catch, /cancel
    
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers, 'memory leak'
    
    return, 1
  endif
  
  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
  b = randomu(seed, n)  
  
  gpuSqrt, 2., 3., a
  
  assert, 0, 'should have failed'
  
  return, 1
end


function gpuSqrt_ut::test_incorrectparams_procedure5
  compile_opt strictarr

  catch, error
  if (error ne 0L) then begin
    catch, /cancel
    
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers, 'memory leak'
    
    return, 1
  endif
  
  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
  
  gpuSqrt, 2., 3., a, 4.
  
  assert, 0, 'should have failed'
  
  return, 1
end


;+
; Define instance variables.
;-
pro gpusqrt_ut__define
  compile_opt strictarr
  
  define = { gpuSqrt_ut, inherits GPUutTestCase }
end