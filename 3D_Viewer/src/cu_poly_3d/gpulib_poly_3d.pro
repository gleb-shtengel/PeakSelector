;
;  file:  gpulib_poly_3d.pro
;
;  GPUlib version of cu_poly_3d.c
;
;  RTK, 12-Aug-2009
;  Last update: 13-Aug-2009
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;--------------------------------------------------------------
;+
;
;-
function gpulib_poly_3d, img, P, Q
    compile_opt idl2, logical_predicate

    dims = size(img, /DIM)
    nframes = dims[2]
    nx = dims[0]  ; nx,ny,nframes = size(img, /DIM), eventually
    ny = dims[1]
    nelem = nx*ny

    ;  Put the img and P,Q arrays on the GPU
    g_img = gpuPutArr(reform(img,nelem*nframes))
    g_P   = gpuPutArr(reform(P,4*nframes))
    g_Q   = gpuPutArr(reform(Q,4*nframes))

    ;  Make the output array on the GPU
    g_ans = gpuMake_Array(nx*ny*nframes, TYPE=size(img,/TYPE))

    ;  x,y point vectors on GPU
    x = reform(findgen(nx) # replicate(1.0, ny), nelem)
    y = reform(findgen(ny) ## replicate(1.0, nx), nelem)
    g_x = gpuPutArr(x)
    g_y = gpuPutArr(y)

    ;  Interpolate each frame by making the necessary views
    for i=0L, nframes-1 do begin

        ;  Make views onto the existing arrays
        gpuView, g_img, nelem*i, nelem, gImg
        gpuView, g_ans, nelem*i, nelem, gAns
        gpuView, g_P, 4*i, 4, gP
        gpuView, g_Q, 4*i, 4, gQ
        gpuReform, gImg, nx, ny
        gpuReform, gAns, nx, ny

        ;  Calculate the points at which to interpolate...


        ;  Interpolate - output written to ans array via view
        _ = gpuInterpolate(gImg, g_x, g_y, LHS=gAns)
    endfor

    ;  Pull back the new data
    ans = reform(gpuGetArr(g_ans), nx, ny, nframes)

    ;  Clean up
    gpuFree, [g_img, g_P, g_Q, g_ans, g_x, g_y]

    return, ans
end

;
;  end gpulib_poly_3d.pro
;

