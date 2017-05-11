; docformat = 'rst'

;+
; Demonstrates 2d interpolation for CPU and GPU LIB
; 
; :History:
;   Tech-X Corp, 2007, pm
;-

;+
; Example of CPU vs. GPU calculations.
;
; :Keywords:
;    just_gpu : in, optional, type=boolean
;       do not do the IDL calculations on the CPU, only the GPU
;    nodisplay : in, optional, type=boolean
;       set to not display frames in order to get a more accurate timing
;    _extra : in, optional, type=keywords
;       keywords to GPUINIT
;-
pro swirl, just_gpu=just_gpu, nodisplay=nodisplay, _extra=e
  compile_opt strictarr

;-----
; generate some image       

  npx = 2048L
  npy = 2048L

  display_x = 640
  display_y = 640 

  if (~keyword_set(nodisplay)) then begin
    window, /free, xsize=display_x, ysize=display_y
    display_id = !d.window

    window, /free, /pixmap, xsize=display_x, ysize=display_y
    pix_id = !d.window

    device, /decompose, bypass=0
    loadct, 33
  endif

  p = reform(sin(findgen(npx * npy) * 4 * 2* !pi / float(npx * npy)), npx, npy)

  xx = findgen(npx, npy)
  yy = xx

  for i=0, npx-1 do xx[i, *] = 2*(findgen(npy)-npy/2)/float(npy)
  for i=0, npy-1 do yy[*, i] = 2*(findgen(npx)-npx/2)/float(npx)
  
  nx = n_elements(xx)
; pp = interpolate(p, x, y)

 niter = 20
;-----
; IDL test
  if (~keyword_set(just_gpu)) then begin
    print, 'Starting plain IDL display...'
    start = systime(2)
  
    r = 1/(sqrt(xx^2+yy^2) + 1)

    for t = -niter, niter do  begin
      alpha = (2*float(t)/float(niter))*2 * !pi * (r - 1/(sqrt(2)+1))/0.585786
      ca = cos(alpha)
      sa = sin(alpha)
      tx = (xx * ca + yy * sa) / sqrt(2.)
      ty = (-xx * sa+ yy * ca) / sqrt(2.)

      x = (tx + 1)/2. * npx
      y = (ty + 1)/2. * npy

      res_gold = interpolate(p, x, y)
      small = congrid(reform(res_gold, 2048, 2048), display_x, display_y)

      if (~keyword_set(nodisplay)) then begin
        wset, pix_id
        tvscl, small
        wset, display_id
        device, copy=[0, 0, display_x, display_y, 0, 0, pix_id]
      endif
    endfor

    cputime = systime(2) - start
  endif

;----
; GPU test

; for the GPU, we have to pre-allocate the arrays in main memory
  res = fltarr(nx)

; initialize the GPU
  gpuinit, _extra=e

; create vectors on the GPU 
  gpu_p = gpufltarr(npx, npy)
  gpu_x = gpufltarr(npx, npy)
  gpu_y = gpufltarr(npx, npy)
  gpu_res = gpufltarr(npx, npy)
  gpu_small = gpufltarr(display_x, display_y)
  gpu_tmp = gpufltarr(npx, npy)
  gpu_alpha = gpufltarr(npx,npy)
  gpu_ca = gpufltarr(npx,npy)
  gpu_sa = gpufltarr(npx, npy)
  gpu_tx = gpufltarr(npx, npy)
  gpu_ty = gpufltarr(npx, npy)
  gpu_r = gpufltarr(npx, npy)

; start the timer...
;  start = systime(2)

; transfer data from host memory to the GPU
  gpuPutArr, p, gpu_p
  gpuPutArr, xx, gpu_x
  gpuPutArr, yy, gpu_y

; start timer here to ignore data transfer cost
;  start = systime(2)

  gpumult, gpu_x, gpu_x, gpu_r
  gpumult, gpu_y, gpu_y, gpu_tmp
  gpuadd, gpu_r, gpu_tmp, gpu_r
  gpusqrt, 1., 1, gpu_r, 0, 1, gpu_r
 
  gpuadd, 0., gpu_tmp, 0., gpu_tmp, 1., gpu_tmp
  gpuDiv, gpu_tmp, gpu_r, gpu_r

  print, 'Starting GPU display...' 
  start = systime(2)

  ;for t = -niter, niter do begin
  for t = niter, -niter, -1 do begin
    angle = (2*float(t)/float(niter)) * 2 * !pi
    gpuAdd, angle/0.585786, gpu_r, 0, gpu_r, -angle/(sqrt(2)+1)/0.585786, gpu_alpha
    gpucos, gpu_alpha, gpu_ca
    gpusin, gpu_alpha, gpu_sa
   
    gpumult, gpu_x, gpu_ca, gpu_tx
    gpumult, gpu_x, gpu_sa, gpu_ty

    gpumult, gpu_y, gpu_sa, gpu_tmp
    gpumult, gpu_y, gpu_ca, gpu_res

  ; err = cudaThreadSynchronize()

    gpuadd, npx/(2*sqrt(2)), gpu_tx, npx/(2*sqrt(2)), gpu_tmp, npx/2, gpu_tx
    gpusub, npy/(2*sqrt(2)), gpu_res, npy/(2*sqrt(2)), gpu_ty, npy/2, gpu_ty
    
    ; perform the actual interpolation
    gpuInterpolate, gpu_p, gpu_tx, gpu_ty, gpu_res

    gpuCongrid, gpu_res, display_x, display_y, gpu_small
    gpuGetArr, gpu_small, small

    if (~keyword_set(nodisplay)) then begin
      wset, pix_id
      tvscl, small
      wset, display_id
      device, copy=[0, 0, display_x, display_y, 0, 0, pix_id]
    endif
  endfor

; stop timer here to ignore data transfer cost
  gputime =  systime(2) - start

  ; get the result back to the CPU
  gpugetarr, gpu_res, res

;  gputime =  (systime(2) - start)

  if (~keyword_set(just_gpu)) then begin
    print, res_gold[0:5]
    print, res[0:5]
    print, 'CPU     : ', cputime
    print, 'GPU     : ', gputime
    print, 'Speedup : ', cputime/gputime
  endif

  if (~keyword_set(nodisplay)) then begin
   wdelete, pix_id
  endif
end
