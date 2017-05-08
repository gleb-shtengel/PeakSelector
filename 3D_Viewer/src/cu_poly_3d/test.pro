pro test, out0=out0, out1=out1
  compile_opt idl2

  n = 30000
  w = 128

  img = intarr(w,w,n)
  d = fix(dist(w))
  for i=0,n-1 do img[0,0,i] = d
  p = fltarr(2,2,n)
  for i=0,n-1 do p[0,0,i] = [[0,0],[0.5,0]]
  q = fltarr(2,2,n)
  for i=0,n-1 do q[0,0,i] = [[0,0.4],[0,0]]

  ;  pure IDL
  out0 = intarr(w,w,n)
  s = systime(1)
  for i=0,n-1 do out0[0,0,i] = poly_2d(img[*,*,i], p[*,*,i], q[*,*,i], 1)
  e = systime(1)
  t0 = e-s
  print, 'pure IDL = ', e-s

  ;  DLM
  dlm_load, 'cu_poly3d'
  s = systime(1)
  out1 = cu_poly3d(img, p, q)
  e = systime(1)
  t1 = e-s
  print, 'DLM = ', e-s

  print, 'IDL / DLM = ', t0/t1
end

