; docformat = 'rst'

;+
; Deconvolution example.
; 
; Running this code should create or restore a blurred image: 
;
; .. image:: hubble-blurred-thumbnail.png 
;
; This should be fixed by using deconvolution using plain IDL and on the GPU.
; The result should look like:
;
; .. image:: hubble-fixed-thumbnail.png
;-

pro decon, devId, GPU=gpu, CPU=cpu

  if keyword_set(GPU) then begin cpu = 0 & gpu = 1 & end
  if keyword_set(CPU) then begin cpu = 1 & gpu = 0 & end

  winx = 800
  winy = 400

  view = obj_new('IDLgrView', viewplane_rect=[0, 0, winx, winy])

  model = obj_new('IDLgrModel')
  view->add, model

  image = obj_new('IDLgrImage')
  model->add, image

  text = obj_new('IDLgrText', locations=[10, 10, 0], color=[255, 255, 255])
  model->add, text
  
  win = obj_new('IDLgrWindow', dimensions=[winx, winy], graphics_tree=view, $
                title='Running Hubble image deconvolution using ' + (keyword_set(gpu) ? 'GPU' : 'CPU'))

  if (~file_test('blurred.sav')) then begin
    read_jpeg, 'hubble.jpg', img

    ; n = 3000
    n = 2000

    img_r = congrid(reform(img(0, *, *)), n, n)
    img_g = congrid(reform(img(1, *, *)), n, n) 
    img_b = congrid(reform(img(2, *, *)), n, n)

    nx = n_elements(img_r(*,0))
    ny = n_elements(img_r(0,*))

    xax = findgen(nx)/float(nx) - 0.5
    yax = findgen(ny)/float(ny) - 0.5

    xsigma = 0.01
    ysigma = 0.02

    xgauss = exp(-xax^2/(2*xsigma)^2)
    ygauss = exp(-yax^2/(2*ysigma)^2)

    ; create the point-spread function
    psf = xgauss # ygauss

    psf_fft = fft(psf)

    ; fix the psf
    psf_fft[where(abs(psf_fft) lt 1e-5)] = 1e-5

    img_r_fft = fft(img_r)
    img_g_fft = fft(img_g)
    img_b_fft = fft(img_b)

    ; create the blurred image
    fimg_r = shift(fft(img_r_fft * psf_fft, -1), [nx/2, ny/2])
    fimg_g = shift(fft(img_g_fft * psf_fft, -1), [nx/2, ny/2])
    fimg_b = shift(fft(img_b_fft * psf_fft, -1), [nx/2, ny/2])

    fimg = fltarr(3, nx, ny)
    fimg[0, *, *] = abs(fimg_r)
    fimg[1, *, *] = abs(fimg_g)
    fimg[2, *, *] = abs(fimg_b)
 
    nx_disp = 640
    ny_disp = 400

    fimg_disp = fltarr(3, nx_disp, ny_disp)

    fimg_disp[0, *, *] = congrid(abs(fimg_r), nx_disp, ny_disp)
    fimg_disp[1, *, *] = congrid(abs(fimg_g), nx_disp, ny_disp)
    fimg_disp[2, *, *] = congrid(abs(fimg_b), nx_disp, ny_disp)
  
    save, fimg, fimg_disp, psf_fft, filename='blurred.sav'
  end else begin 
    restore, filename='blurred.sav'
    
    nx = n_elements(fimg[0, *, 0])
    ny = n_elements(fimg[0, 0, *])
  end

  cputime = "N/A"
  gputime = "N/A"
 
  for nit = 0, 100 do begin

    image->setProperty, data=bytscl(congrid(fimg_disp, 3, winx, winy))
    win->draw
 
    ; now start the deconvolution
    if (keyword_set(cpu)) then begin
      t = systime(2)

      fimg_clean = fltarr(3, nx, ny)
      for c = 0, 2 do begin
        fimg_fft = fft(fimg[c, *, *]) 
        fimg_decon = fimg_fft / psf_fft
        fimg_clean[c, *, *] = float(fft(fimg_decon, -1))
      end
      cputime = systime(2) - t

      clean = shift(fimg_clean, [0, nx/2, ny/2])
      text->setProperty, strings='Deconvolution time: ' + strtrim(cputime, 2)
      image->setProperty, data=bytscl(congrid(clean, 3, winx, winy))
      win->draw
      wait, 1.
    endif 
 
   ; start gpu version
   if (keyword_set(gpu)) then begin
     gpuinit, devId

     fimg = abs(fimg)

     image->setProperty, data=congrid(fimg_disp, 3, winx, winy)

     t = systime(2)
 
     gpuPutarr, psf_fft, gpu_psf_fft
 
     fimg_clean = fltarr(3, nx, ny)
     for c = 0, 2 do begin
       gpuFFT, reform(fimg[c, *, *]), gpu_fimg, /DIM2D
       gpuDiv, gpu_fimg, gpu_psf_fft, gpu_fimg
       gpuFFt, gpu_fimg, gpu_fimg, /DIM2D, /INVERSE
       gpuFloat, gpu_fimg, gpu_fimg_clean_fix
       gpuGetArr, gpu_fimg_clean_fix, fimg_clean_gpu
       fimg_clean[c, *, *] = fimg_clean_gpu
     end

     gputime = systime(2) - t
     text->setProperty, strings='GPU deconvolution time: ' + strtrim(gputime, 2)
 
     clean = shift(fimg_clean, [0, nx / 2, ny / 2])

     gpuFree, gpu_fimg_clean_fix
     gpuFree, gpu_fimg
     gpuFree, gpu_psf_fft

     image->setProperty, data=bytscl(congrid(clean, 3,winx,winy))
     win->draw
     wait, 1
   end 
 end
end


;+
; Run the Hubble image deconvolution example.
;
; :Keywords:
;    gpu : in, optional, type=long
;       specify software (0) or hardware (1) GPU mode
;-
pro decon_hubble, GPU=gpu
  compile_opt strictarr
  
  print, 'GPULib demo: Image deconvolution'
  print, '--------------------------------'
  print, ''
  args = command_line_args(count=nargs)
  if (nargs gt 0L || n_elements(gpu) gt 0L) then begin
    _gpu = n_elements(gpu) eq 0L ? (nargs gt 0L ? long(args[0]) : 1L) : gpu
    decon, _gpu, /GPU
  end else begin
    decon, /CPU
  end
end


