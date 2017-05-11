;
;  file:  ps_test.pro
;
;  Test the GPU Gaussian fit
;
;  RTK, 25-Sep-2009
;  Last update, 06-Nov-2009
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

pro ps_display, s0, n0, gauss=gauss, im0=im0, im1=im1
    compile_opt idl2, logical_predicate

    s = (n_elements(s0)) ? s0 : 0
    n = (n_elements(n0)) ? n0 : 50

    window, 0, xs=3*30*11, ys=30*11

    for i=s, s+n-1 do begin
        tvscl, rebin(gauss[*,*,i],30*11,30*11),1
        tvscl, rebin(im0[*,*,i],30*11,30*11),0
        tvscl, rebin(im1[*,*,i],30*11,30*11),2
        wait, 2.0
    endfor
end


pro ps_test_big
    compile_opt idl2, logical_predicate

    ;  Load the DLM to not count CUDA init time
    dlm_load, 'CU_GAUSS2D'

    ;  Restore the image data
    restore, 'gauss50000.sav'

    ;  Increase the stack of images
    gg = uintarr(11,11,6000000)
    
    for i=0L, 119 do begin
        print, i
        gg[*,*,50000*i:(50000*i+50000-1)] = gauss
    endfor

    gauss = gg
    gg = 0b

    dims = size(gauss,/DIM)
    params = fltarr(7, dims[2])
    images = uintarr(11,11, dims[2])
    img = uintarr(11,11)
    constraints = float([0,500,100,1500,0,5,0,5,0,10,0,10])
    imax = 25

    sss = systime(1)
    p = cu_gauss2d(gauss, imax, constraints, 0)
    eee = systime(1)
    gauss = 0b
    p = reform(p,13, dims[2], /OVERWRITE)
    print, 'Fit time ', eee-sss

    ;x = reform(indgen(11) # replicate(1,11),11*11)
    ;y = reform(indgen(11) ## replicate(1,11), 11*11)

    ;for i=0L, dims[2]-1 do begin
    ;    img[x,y] = fix(p[0,i] + p[1,i]*exp(-0.5*(((x-p[4,i])/p[2,i])^2 + ((y-p[5,i])/p[3,i])^2)), TYPE=12)
    ;    images[*,*,i] = img
    ;endfor
    
    stop
end
