; docformat = 'rst'

;+
; Find craters on the moon.
; 
; :Keywords:
;    display : in, optional, type=boolean
;       set to display intermediate results
;    gpu : in, optional, type=boolean
;       set to use the GPU lib, otherwise uses standard IDL
;    _extra : in, optional, type=keywords
;       keywords to GPUINIT 
;-
pro ga_moon_craters, display=display, gpu=gpu, _extra=e
  compile_opt strictarr
  
  ; read in image
  im = read_png(filepath('moon.png', root=mg_src_root()))

  sz = size(im, /dimensions)
  nx = sz[0]
  ny = sz[1]

  if (keyword_set(gpu)) then gpuinit, _extra=e
      
  if (keyword_set(gpu)) then begin
    imGpu = gpuFltarr(nx, ny)
    gpuPutArr, float(im), imGpu
  endif else begin
    pImage = ptr_new(im)
  endelse
  
  x = fltarr(nx, ny)
  xx = (findgen(nx) - (nx / 2.)) / (nx / 2.)
  yy = (findgen(ny) - (ny / 2.)) / (ny / 2.)
  x = xx # (fltarr(ny) + 1.0)    ; copies xx as each row
  y = (fltarr(nx) + 1.0) # yy    ; copies yy as each column
  
  if (keyword_set(gpu)) then begin
    xGpu = gpuFltarr(nx, ny)
    yGpu = gpuFltarr(nx, ny)
    gpuPutArr, x, xGpu
    gpuPutArr, y, yGpu
    classname = 'GA_GPU_Crater'
  endif else begin
    xPtr = ptr_new(x)
    yPtr = ptr_new(y)
    classname = 'GA_Crater'
  endelse
  
  tournament = obj_new('GA_Tournament', classname=classname, n_rounds=5, $
                       population_size=5)
  
  if (keyword_set(display)) then begin
    window, title='Intermediate results', /free, xsize=nx, ysize=ny
    tv, im
    device, get_graphics_function=oldGF
    device, set_graphics_function=7
  endif
  
  if (keyword_set(gpu)) then begin
    xtemp = gpuFltarr(nx, ny)
    ytemp = gpuFltarr(nx, ny)
    winner = tournament->run(image=imGpu, x_gpu=xGpu, y_gpu=yGpu, $
                             xtemp=xtemp, ytemp=ytemp, time=time, $
                             display=display, _extra=e)
  endif else begin
    winner = tournament->run(image=pImage, x=xPtr, y=yPtr, $
                             time=time, $
                             display=display, _extra=e)
  endelse
  
  winner->getProperty, x_pos=xpos, y_pos=ypos, x_axis=xAxis, y_axis=yAxis
  
  window, title='The best solution', /free, xsize=nx, ysize=ny
  tv, im
  winner->display, channel=1
  device, set_graphics_function=oldGF
  
  format = '(%"%s: centered = (%0.2f, %0.2f), axes (%0.2f, %0.2f), found in %0.2f seconds")'
  print, format=format, $
         (keyword_set(gpu) ? 'GPU' : 'CPU'), xpos, ypos, xAxis, yAxis, time
  
  obj_destroy, tournament
  
  if (keyword_set(gpu)) then begin
    gpuFree, [imGpu, xGpu, yGpu, xtemp, ytemp]
  endif else begin
    ptr_free, pImage, xPtr, yPtr
  endelse

end
