;
;  file:  test_d.pro
;
;  Test convolution kernel
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function equal, a,b
    compile_opt idl2, logical_predicate

    m = median(abs(a-b))
    return, (m lt 1e-5)
end


function idl_convol_d, s, k
    compile_opt idl2, logical_predicate

    n = (size(s,/DIM))[2]
    ans = dblarr(size(s,/DIM))

    for i=0L,n-1 do begin
        ans[*,*,i] = convol(reform(s[*,*,i]), k, /EDGE_ZERO)
    endfor

    return, ans
end


pro test_d
    compile_opt idl2, logical_predicate

    dlm_load, 'cu_convol_d'

    ; Stack of images
    s = dblarr(256,256,2000)
    for i=0,1999 do s[*,*,i] = double(dist(256))

    ;  Kernel
    k = double(randomu(seed,17,17))
    print, 'k = ', (size(k,/DIM))[0]

    ; Convolve in IDL
    sss = systime(1)
    p0 = idl_convol_d(s,k)
    eee0 = systime(1) - sss

    ; Convolve with CUDA
    sss = systime(1)
    p1 = cu_convol_d(s,k)
    eee1 = systime(1) - sss

    print, 'IDL runtime = ', eee0
    print, 'CUDA runtime = ', eee1
    print, 'IDL/CUDA = ', eee0/eee1
    print, 'p0 == p1 = ', equal(p0,p1)
    stop
end

