; docformat = 'rst'


function gpuSub_ut::test_float_function2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
  b = randomu(seed, n)  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  c_gpu = gpuSub(a, b)
  
  error = total(abs(a - b - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)          
  assert, c_gpu.type eq 4, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
  
  return, 1
end


function gpuSub_ut::test_double_function2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = double(randomu(seed, n))
  b = double(randomu(seed, n))  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  c_gpu = gpuSub(a, b)
  
  error = total(abs(a - b - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 5, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
  
  return, 1
end


function gpuSub_ut::test_complex_function2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = complex(randomu(seed, n), randomu(seed, n))
  b = complex(randomu(seed, n), randomu(seed, n))  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  c_gpu = gpuSub(a_gpu, b_gpu)
  
  error = total(abs(a - b - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 6, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
  
  return, 1
end


function gpuSub_ut::test_dcomplex_function2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = dcomplex(randomu(seed, n), randomu(seed, n))
  b = dcomplex(randomu(seed, n), randomu(seed, n))  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  c_gpu = gpuSub(a_gpu, b_gpu)
  
  error = total(abs(a - b - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 9, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
  
  return, 1
end


function gpuSub_ut::test_float_function5
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
  b = randomu(seed, n)  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  c_gpu = gpuSub(2., a, 3., b, 4.)
  
  error = total(abs(2. * a - 3. * b + 4. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 4, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
  
  return, 1
end


function gpuSub_ut::test_double_function5
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = double(randomu(seed, n))
  b = double(randomu(seed, n))  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  c_gpu = gpuSub(2., a, 3., b, 4.)
  
  error = total(abs(2. * a - 3. * b + 4. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 5, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
  
  return, 1
end


function gpuSub_ut::test_complex_function5
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = complex(randomu(seed, n), randomu(seed, n))
  b = complex(randomu(seed, n), randomu(seed, n))  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  c_gpu = gpuSub(2., a, 3., b, 4.)
  
  error = total(abs(2. * a - 3. * b + 4. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 6, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
  
  return, 1
end


function gpuSub_ut::test_dcomplex_function5
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = dcomplex(randomu(seed, n), randomu(seed, n))
  b = dcomplex(randomu(seed, n), randomu(seed, n))  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  c_gpu = gpuSub(2., a, 3., b, 4.)
  
  error = total(abs(2. * a - 3. * b + 4. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 9, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
  
  return, 1
end


function gpuSub_ut::test_lhs_function2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
  b = randomu(seed, n)  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  c_gpu = gpuMake_array(n)
  c_gpu = gpuSub(a, b, lhs=c_gpu)
  
  error = total(abs(a - b - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
  
  return, 1
end


function gpuSub_ut::test_compound_function2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
  b = randomu(seed, n)  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  c_gpu = gpuSub(gpuSub(a, b), b)
  
  error = total(abs(a - 2 * b - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
  
  return, 1
end


function gpuSub_ut::test_convert_function2
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers
  
  n = 10
  a = randomu(seed, n)
  b = randomu(seed, n)  
  
  c_gpu = gpuSub(gpuSub(a, b), b)
  
  error = total(abs(a - 2 * b - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 1, 'memory leak'
  endif
  
  gpuFree, [c_gpu]
  
  return, 1
end


function gpuSub_ut::test_incorrectparams_procedure1
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
  
  gpuSub, a
  
  assert, 0, 'should have failed'
  
  return, 1
end


function gpuSub_ut::test_incorrectparams_procedure4
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
  
  gpuSub, 2., a, 3., b
  
  assert, 0, 'should have failed'
  
  return, 1
end


function gpuSub_ut::test_float_procedure3
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = randomu(seed, n)
  b = randomu(seed, n)  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  gpuSub, a_gpu, b_gpu, c_gpu
  
  error = total(abs(a - b - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 4, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
    
  return, 1
end


function gpuSub_ut::test_double_procedure3
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = double(randomu(seed, n))
  b = double(randomu(seed, n))  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  gpuSub, a_gpu, b_gpu, c_gpu
  
  error = total(abs(a - b - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 5, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
    
  return, 1
end


function gpuSub_ut::test_complex_procedure3
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = complex(randomu(seed, n), randomu(seed, n))
  b = complex(randomu(seed, n), randomu(seed, n))  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  gpuSub, a_gpu, b_gpu, c_gpu
  
  error = total(abs(a - b - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 6, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
    
  return, 1
end


function gpuSub_ut::test_dcomplex_procedure3
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = dcomplex(randomu(seed, n), randomu(seed, n))
  b = dcomplex(randomu(seed, n), randomu(seed, n))  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  gpuSub, a_gpu, b_gpu, c_gpu
  
  error = total(abs(a - b - gpuGetArr(c_gpu))) 
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 9, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
    
  return, 1
end


function gpuSub_ut::test_float_procedure6
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = randomu(seed, n)
  b = randomu(seed, n)  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  gpuSub, 2., a_gpu, 3., b_gpu, 4., c_gpu
  
  error = total(abs(2. * a - 3. * b + 4. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 4, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
    
  return, 1
end


function gpuSub_ut::test_double_procedure6
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = double(randomu(seed, n))
  b = double(randomu(seed, n))  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  gpuSub, 2., a_gpu, 3., b_gpu, 4., c_gpu
  
  error = total(abs(2. * a - 3. * b + 4. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 5, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
    
  return, 1
end


function gpuSub_ut::test_complex_procedure6
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = complex(randomu(seed, n), randomu(seed, n))
  b = complex(randomu(seed, n), randomu(seed, n))  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  gpuSub, 2., a_gpu, 3., b_gpu, 4., c_gpu
  
  error = total(abs(2. * a - 3. * b + 4. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 6, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
    
  return, 1
end


function gpuSub_ut::test_dcomplex_procedure6
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = dcomplex(randomu(seed, n), randomu(seed, n))
  b = dcomplex(randomu(seed, n), randomu(seed, n))  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  gpuSub, 2., a_gpu, 3., b_gpu, 4., c_gpu
  
  error = total(abs(2. * a - 3. * b + 4. - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  assert, c_gpu.type eq 9, 'incorrect type'
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
    
  return, 1
end


function gpuSub_ut::test_compound_procedure3
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = randomu(seed, n)
  b = randomu(seed, n)  
  
  a_gpu = gpuPutArr(a)
  b_gpu = gpuPutArr(b)
  
  gpuSub, gpuSub(a_gpu, b_gpu), b_gpu, c_gpu
  
  error = total(abs(a - 2 * b - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 3, 'memory leak'
  endif
  
  gpuFree, [a_gpu, b_gpu, c_gpu]
    
  return, 1
end


function gpuSub_ut::test_convert_procedure3
  compile_opt strictarr

  mg_heapinfo, n_pointers=origPointers

  n = 10
  a = randomu(seed, n)
  b = randomu(seed, n)  
  
  gpuSub, a, b, c_gpu
  
  error = total(abs(a - b - gpuGetArr(c_gpu)))
  assert, error lt n * self.tolerance, $
          'incorrect subtraction result: error = ' + strtrim(error, 2)            
  if (!gpu.mode eq 0L) then begin
    mg_heapinfo, n_pointers=afterPointers
    assert, afterPointers eq origPointers + 1, 'memory leak'
  endif
  
  gpuFree, [c_gpu]
    
  return, 1
end


function gpuSub_ut::test_incorrectparams_procedure2
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
  
  gpuSub, a, b
  
  assert, 0, 'should have failed'
  
  return, 1
end


function gpuSub_ut::test_incorrectparams_procedure5
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
  
  gpuSub, 2., a, 3., b, 4.
  
  assert, 0, 'should have failed'
  
  return, 1
end


;+
; Define instance variables.
;-
pro gpuSub_ut__define
  compile_opt strictarr
  
  define = { gpuSub_ut, inherits GPUutTestCase }
end