;
;  file:  test_f.pro
;
;  Test convolution kernel
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function equal, a,b
    compile_opt idl2, logical_predicate

    m = median(abs(a-b))
    return, (m lt 1e-5)
end


function idl_convol_f, s, k
    compile_opt idl2, logical_predicate

    n = (size(s,/DIM))[2]
    ans = fltarr(size(s,/DIM))

    for i=0L,n-1 do begin
        ans[*,*,i] = convol(reform(s[*,*,i]), k, /EDGE_ZERO)
    endfor

    return, ans
end


pro test_f
    compile_opt idl2, logical_predicate

    dlm_load, 'cu_convol_f'

    ; Stack of images
    s = fltarr(256,256,5000)
    for i=0,4999 do s[*,*,i] = float(dist(256))

    ;  Kernel
    k = float(randomu(seed,111,111))
    print, 'k = ', 111

    ; Convolve in IDL
    sss = systime(1)
    p0 = idl_convol_f(s,k)
    eee0 = systime(1) - sss

    ; Convolve with CUDA
    sss = systime(1)
    p1 = cu_convol_f(s,k)
    eee1 = systime(1) - sss

    print, 'IDL runtime = ', eee0
    print, 'CUDA runtime = ', eee1
    print, 'IDL/CUDA = ', eee0/eee1
    print, 'p0 == p1 = ', equal(p0,p1)
    stop
end

