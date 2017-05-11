pro test_cuda,n,w, out=out1
  compile_opt idl2

  img = intarr(w,w,n)
  d = fix(dist(w))
  for i=0,n-1 do img[0,0,i] = d
  p = fltarr(2,2,n)
  for i=0,n-1 do p[0,0,i] = [[0,0],[0.5,0]]
  q = fltarr(2,2,n)
  for i=0,n-1 do q[0,0,i] = [[0,0.4],[0,0]]

  ;  DLM
  dlm_load, 'cu_poly_3d'
  s = systime(1)
  out1 = cu_poly_3d(img, p, q)
  e = systime(1)
  t1 = e-s
  print, 'DLM = ', e-s
end

