pro filter_test
    compile_opt idl2, logical_predicate

    restore, filepath('gpulib.sav', SUBDIR=['products'])
    gpuinit
    dlm_load, 'cu_filterit_f'
    restore, 'gp.sav'

    a = indgen(45)
    p = [9, a[18:24], 26, a[37:42]]
    index = bytarr(45)
    index[p] = 1

    ;  CUDA kernel
    s=systime(1)
    f0 = cu_filterit_f(CGroupParams, ParameterLimits, index)
    e=systime(1)
    t_cuda = e-s
    print, 'CUDA filter time = ', t_cuda

    ;  Plain IDL
    s = systime(1)
    CGPsz=size(CGroupParams)
    allind=indgen(45)
    pl0=ParameterLimits[p,0]#replicate(1,CGPsz[2])
    pl1=ParameterLimits[p,1]#replicate(1,CGPsz[2])
    filter0=(CGroupParams[p,*] ge pl0) and (CGroupParams[p,*] le pl1)
    f1=floor(total(filter0,1)/n_elements(p))
    e = systime(1)
    t_idl = e-s
    print, 'IDL filter time = ', t_idl

    ;  GPUlib
    indices = [9, a[18:24], 26, a[37:42]]

    m = (size(CGroupParams, /DIM))[0]
    n = (size(CGroupParams, /DIM))[1]
    k = n_elements(indices)

    s=systime(1)
    
    gpu_lo = gpuPutArr(reform(ParameterLimits[indices,0]))
    gpu_hi = gpuPutArr(reform(ParameterLimits[indices,1]))
    gpu_peaks = gpuPutArr(reform(CGroupParams[indices,*]))

    gpu_ones = gpuPutArr(replicate(1.0, n))

    gpu_kl = gpuMatrix_multiply(gpu_lo, gpu_ones, /BTRANSPOSE)
    gpu_kh = gpuMatrix_multiply(gpu_hi, gpu_ones, /BTRANSPOSE)

    gpu_a = gpuGTEQ(gpu_peaks, gpu_kl)
    gpu_b = gpuLTEQ(gpu_peaks, gpu_kh)
    gpu_c = gpuTotal(gpu_a, 1)
    gpu_d = gpuTotal(gpu_b, 1)
    t = replicate(k,n)
    gpu_e = gpuEq(t, gpu_c)
    gpu_f = gpuEq(t, gpu_d)
    gpu_g = gpuAdd(gpu_e, gpu_f)
    gpu_h = gpuEq(replicate(2,n), gpu_g)

    f2 = gpuGetArr(gpu_h)

    gpuFree, gpu_a
    gpuFree, gpu_b
    gpuFree, gpu_c
    gpuFree, gpu_d
    gpuFree, gpu_e
    gpuFree, gpu_f
    gpuFree, gpu_g
    gpuFree, gpu_h
    gpuFree, gpu_lo
    gpuFree, gpu_hi
    gpuFree, gpu_peaks
    gpuFree, gpu_ones
    gpuFree, gpu_kl
    gpuFree, gpu_kh
    
    e = systime(1)
    t_gpulib = e-s
    print, 'GPUlib filterit time = ', t_gpulib

    print, 'Speed up (IDL vs CUDA)   = ', t_idl / t_cuda
    print, 'Speed up (IDL vs GPUlib) = ', t_idl / t_gpulib

    print, 'Filters match (f0,f1) = ', array_equal(f0, f1)
    print, 'Filters match (f0,f2) = ', array_equal(f0, f2)
    print, 'Filters match (f1,f2) = ', array_equal(f1, f2)
end

