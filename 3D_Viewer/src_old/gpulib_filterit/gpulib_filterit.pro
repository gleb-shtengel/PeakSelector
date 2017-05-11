function gpulib_filterit
    compile_opt idl2, logical_predicate

    restore, filepath('gpulib.sav', SUBDIR=['products'])
    gpuinit

    restore, '../cu_filterit_f/gp.sav'

    a = indgen(45)
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

    ans = gpuGetArr(gpu_h)

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
    print, 'GPUlib filterit time = ', e-s

    return, ans
end

