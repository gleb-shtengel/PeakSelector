pro fdtd

  mpi_consts

  myrank = MPIDL_COMM_RANK()
  nprocs = MPIDL_COMM_SIZE()

  rank_up   = long((myrank + 1 ) mod nprocs)
  rank_down = long((nprocs + myrank - 1 ) mod nprocs)

  nx = 512L
  ny = 512L
  n = nx * ny
  
  gpuinit, myrank
  print, '[', myrank,'] initialized'

  ex_gpu = gpuFltarr(n)
  ey_gpu = gpuFltarr(n)
  ez_gpu = gpuFltarr(n)
  bx_gpu = gpuFltarr(n)
  by_gpu = gpuFltarr(n)
  bz_gpu = gpuFltarr(n)

  winx = 640
  winy = 400

; view = obj_new('IDLgrView')
; view->setProperty, projection=2

; model = obj_new('IDLgrModel')
; model->rotate, [1, 0., 0], -50.
; model->rotate, [0, 1, 0], -20
; view->add, model

; light = obj_new('IDLgrLight', type=2, location = [1,-1,1])
; model->add, light

; surf = obj_new('IDLgrSurface', $
;               dataz = fltarr(nx, ny), $
;               datax = 1.6*findgen(nx)/float(nx)-0.8, $
;               datay = 1.6*findgen(ny)/float(ny)-0.8, $
;               color=[20., 120., 255.], shading=1, $
;               style=2, shininess = 85.,  $
;               ambient=[120., 240., 10.])
; model->add, surf

 ; text = obj_new('IDLgrText', locations=[10, 10, 0], color=[255, 255, 255])
 ; model->add, text
  
;  win = obj_new('IDLgrWindow', dimensions=[winx, winy], graphics_tree=view, $
;                title='FDTD ')



; create conductivity field
  con = fltarr(nx, ny) + 1.
  con[0, *] = 0.
  con[nx-1, *] = 0.
  con[*,0]  = 0.
  con[*, ny-1] = 0.

  l = nx/3
  for i = 0, l do begin
     con[0:l-i, ny/4.+i] = 0.
     con[nx-i-1:*, 2*ny/4.+i] = 0.
  end

  gpuPutArr, con, gpu_con

  tim = systime(2)

  for t = 0, 5000 do begin

      ; setting the boundary condition
  
      ;ez = reform(ez, nx, ny)
      ;ez(nx/2, ny/2) = sin(t*2*!pi/20) 
      ;ez = reform(ez, nx*ny)

     ;if myrank eq 1 then begin
     ; gpuView, ez_gpu, ny/2*nx + nx/2, 1, ez_gpu_bc 
     ; gpuPutArr, [sin(t*2*!pi/100.)], ez_gpu_bc 
     ;end

;-----
;    apply BC

    ; gpuMult, ex_gpu, gpu_con, ex_gpu
    ; gpuMult, ey_gpu, gpu_con, ey_gpu
    ; gpuMult, ez_gpu, gpu_con, ez_gpu

     if (myrank eq 0) and (t lt 300) then begin
        gpuView, ez_gpu, nx + nx/2-200, 400, ez_gpu_bc
        ;gpuPutArr, sin(t*2*!pi/30.) * sin(findgen(400)*!pi/400.)^2 * sin(t*!pi/300)^2, ez_gpu_bc
        gpuPutArr, sin(t*2*!pi/(30. - t/20.)) * sin(findgen(400)*!pi/400.)^2 * sin(t*!pi/300)^2, ez_gpu_bc
     end

    ;  bx(0:n-nx-1) = bx(0:n-nx-1) - ( ez(nx:*) - ez(0:n-nx-1))
     gpuView, bx_gpu, 0,  n - nx - 1, bx_gpu_view
     gpuView, ez_gpu, nx, n - nx - 1, ez_gpu_view1
     gpuView, ez_gpu, 0,  n - nx - 1, ez_gpu_view2

     gpuSub, bx_gpu_view, ez_gpu_view1, bx_gpu_view
     gpuAdd, bx_gpu_view, ez_gpu_view2, bx_gpu_view

;      by(0:n-nx-1) = by(0:n-nx-1) - (-ez(1:n-nx) + ez(0:n-nx-1))
     gpuView, by_gpu, 0, n - nx - 1, by_gpu_view
     gpuView, ez_gpu, 1, n - nx - 1, ez_gpu_view1
     gpuView, ez_gpu, 0, n - nx - 1, ez_gpu_view2

     gpuAdd, by_gpu_view, ez_gpu_view1, by_gpu_view
     gpuSub, by_gpu_view, ez_gpu_view2, by_gpu_view

;      bz(0:n-nx-1) = bz(0:n-nx-1) - ((ey(1:n-nx) - ey(0:n-nx-1)) $
;                                   - (ex(nx:*)   - ex(0:n-nx-1)))
     gpuView, bz_gpu, 0,  n - nx - 1, bz_gpu_view
     gpuView, ey_gpu, 1,  n - nx - 1, ey_gpu_view1
     gpuView, ey_gpu, 0,  n - nx - 1, ey_gpu_view2
     gpuView, ex_gpu, nx, n - nx - 1, ex_gpu_view1
     gpuView, ex_gpu, 0,  n - nx - 1, ex_gpu_view2

     gpuSub, bz_gpu_view, ey_gpu_view1, bz_gpu_view
     gpuAdd, bz_gpu_view, ey_gpu_view2, bz_gpu_view
     gpuSub, bz_gpu_view, ex_gpu_view1, bz_gpu_view
     gpuAdd, bz_gpu_view, ex_gpu_view2, bz_gpu_view

;-----
;    Ghost cell exchange..

     gpuView, bx_gpu, nx, nx, bx_gpu_view
     gpuView, by_gpu, nx, nx, by_gpu_view
     gpuView, bz_gpu, nx, nx, bz_gpu_view
    
     gpuGetArr, bx_gpu_view, bx_buf
     gpuGetArr, by_gpu_view, by_buf
     gpuGetArr, bz_gpu_view, bz_buf

  
     buf              = fltarr(nx * 3)
     buf[0:nx-1]      = bx_buf
     buf[nx:2*nx-1]   = by_buf
     buf[2*nx:3*nx-1] = bz_buf

    ; deal with periodicity 
     if myrank eq 0 then buf = buf * 0.

     mpidl_Send, buf, dest=rank_down, tag = 42L
     cnt = nx * 3
     buf = mpidl_Recv(count=cnt, source=rank_up, /float)

     ;print, 'rank = ', myrank, ' received message'
     gpuView, bx_gpu, n-2*nx, nx, bx_gpu_view
     gpuView, by_gpu, n-2*nx, nx, by_gpu_view
     gpuView, bz_gpu, n-2*nx, nx, bz_gpu_view

     gpuPutArr, buf[0:nx-1],      bx_gpu_view
     gpuPutArr, buf[nx:2*nx-1],   by_gpu_view
     gpuPutArr, buf[2*nx:3*nx-1], bz_gpu_view
 
;-----
; end of B field update

;      ex(nx:*) = ex(nx:*) + ( bz(nx:*) - bz(0:n-nx-1))/4.
     gpuView, ex_gpu, nx, n - nx - 1, ex_gpu_view
     gpuView, bz_gpu, nx, n - nx - 1, bz_gpu_view1
     gpuView, bz_gpu, 0,  n - nx - 1, bz_gpu_view2

     gpuAdd, 1.0, ex_gpu_view, 0.25, bz_gpu_view1, 0., ex_gpu_view
     gpuSub, 1.0, ex_gpu_view, 0.25, bz_gpu_view2, 0., ex_gpu_view

    ;  ey(nx:*) = ey(nx:*) + (-bz(nx:*) + bz(nx-1:n-2))/4.
     gpuView, ey_gpu, nx, n - nx - 1, ey_gpu_view
     gpuView, bz_gpu, nx, n - nx - 1, bz_gpu_view1
     gpuView, bz_gpu, nx-1,  n - nx - 1, bz_gpu_view2

     gpuSub,1.0, ex_gpu_view, 0.25, bz_gpu_view1, 0., ex_gpu_view
     gpuAdd,1.0, ex_gpu_view, 0.25, bz_gpu_view2, 0., ex_gpu_view

;      ez(nx:*) = ez(nx:*) + ( by(nx:*) - by(nx-1:n-2))/4  $
;                           -( bx(nx:*) - bx(0:n-nx-1))/4.
     gpuView, ez_gpu, nx, n - nx - 1, ez_gpu_view
     gpuView, by_gpu, nx, n - nx - 1, by_gpu_view1
     gpuView, by_gpu, nx-1, n - nx - 1, by_gpu_view2
     gpuView, bx_gpu, nx, n - nx - 1, bx_gpu_view1
     gpuView, bx_gpu, 0,  n - nx - 1, bx_gpu_view2

     gpuAdd,1.0, ez_gpu_view, 0.25, by_gpu_view1, 0., ez_gpu_view
     gpuSub,1.0, ez_gpu_view, 0.25, by_gpu_view2, 0., ez_gpu_view
     gpuSub,1.0, ez_gpu_view, 0.25, bx_gpu_view1, 0., ez_gpu_view
     gpuAdd,1.0, ez_gpu_view, 0.25, bx_gpu_view2, 0., ez_gpu_view

;-----
;    apply BC

     gpuMult, ex_gpu, gpu_con, ex_gpu
     gpuMult, ey_gpu, gpu_con, ey_gpu
     gpuMult, ez_gpu, gpu_con, ez_gpu

;-----
;    get the upper row for messaging to neighbor

     gpuView, ex_gpu, n-2*nx, nx, ex_gpu_view
     gpuView, ey_gpu, n-2*nx, nx, ey_gpu_view
     gpuView, ez_gpu, n-2*nx, nx, ez_gpu_view
    
     gpuGetArr, ex_gpu_view, ex_buf
     gpuGetArr, ey_gpu_view, ey_buf
     gpuGetArr, ez_gpu_view, ez_buf

     buf              = fltarr(nx * 3)
     buf[0:nx-1]      = ex_buf
     buf[nx:2*nx-1]   = ey_buf
     buf[2*nx:3*nx-1] = ez_buf


     mpidl_Send, buf, dest=rank_up, tag = 42L
     cnt = nx * 3
     buf = mpidl_Recv(count=cnt, source=rank_down, /float)

    ; deal with periodicity 
     if myrank eq 0 then buf = buf * 0.

     gpuView, ex_gpu, nx, nx, ex_gpu_view
     gpuView, ey_gpu, nx, nx, ey_gpu_view
     gpuView, ez_gpu, nx, nx, ez_gpu_view
    
     gpuPutArr, buf[0:nx-1],      ex_gpu_view
     gpuPutArr, buf[nx:2*nx-1],   ey_gpu_view
     gpuPutArr, buf[2*nx:3*nx-1], ez_gpu_view

      ;rbx = reform(bx, nx, ny)
      ;rby = reform(by, nx, ny)
      ;rbz = reform(bz, nx, ny)

      ;rex = reform(ex, nx, ny)
      ;rey = reform(ey, nx, ny)
;      rez = reform(ez, nx, ny)
;      vpplothist, rez
     if t mod 10 eq 0 then begin
       gpuGetArr, ez_gpu, rez
       r= transpose(reform(rez, nx, ny))
       r[0,0:1] = -1
       r[0, nx-2:*]=1
       vpplothist, r
       ;vpplothist, transpose(reform(rez, nx, ny))
       wait, 0.1 
     end   
;     surf->setProperty, dataz=reform(rez, nx, ny)*2, max_value = 0.5
;     win->draw
     
     ;shade_surf, reform(rez, nx, ny), zrange=[-1,1], ax=70
  end
  print, systime(2) - tim
end
