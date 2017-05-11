function oblique, vol, planeEq, PIXEL_SPACING=spc, $
    LOCATION=loc, ORIENTATION=orient, MISSING=missing, $
    OUT_SPC=outSpc, OUT_LOC=outLoc, OUT_ORIENT=outOrient
  compile_opt idl2, logical_predicate

  if (n_elements(spc) eq 0) then spc = [1,1,1]
  if (n_elements(loc) eq 0) then loc = [0,0,0]
  if (n_elements(orient) eq 0) then orient = [1,0,0, 0,1,0]
  if (n_elements(missing) eq 0) then missing = vol[0] * 0b

  tx = DicomTransform(loc, orient, spc[0:1], spc[2])
  
  ; set up output coord system
  void = min(abs(planeEq[0:2]),mindex)
  y_axis = crossp(planeEq[0:2], [0,1,2] eq mindex)
  y_axis /= sqrt(total(y_axis^2.0))
  x_axis = crossp(planeEq[0:2], y_axis)
  outOrient = [x_axis, y_axis]
  outLoc = [0,0,0]
  
  
  voldim = size(vol, /DIMENSIONS)
  
  corners = [ $
    [0,0,0], [voldim[0],0,0], $
    [voldim[0],voldim[1],0], [0, voldim[1],0], $
    [0,0,voldim[2]], [voldim[0],0,voldim[2]], $
    [voldim], [0, voldim[1], voldim[2]]] - 0.5

  c_patient = ApplyTransform(tx, corners)

    ; check polygon intersections for all 6 faces
  p = ptrarr(6, /ALLOCATE_HEAP)
  *p[0] = c_patient[0:2,[0,1,2,3]] ; bottom
  *p[1] = c_patient[0:2,[0,1,5,4]] ; front
  *p[2] = c_patient[0:2,[4,5,6,7]] ; top
  *p[3] = c_patient[0:2,[2,3,7,6]] ; back
  *p[4] = c_patient[0:2,[0,3,7,4]] ; left
  *p[5] = c_patient[0:2,[1,2,6,5]] ; right
  
  n_pts = 0
  for i=0, 5 do begin
    tmp_pts = PolylinePlaneIntersect(*p[i], planeEq, /CLOSED)
    if (n_elements(tmp_pts) gt 1) then begin
      Padd, pts, tmp_pts[*], n_pts
    endif
  endfor
  ptr_free, p
  if (n_pts eq 0) then begin
    outImg = replicate(missing, [128,128])
    return, outImg
  endif 
  if (n_pts ne n_elements(pts)) then pts = pts[0:n_pts-1]
  pts = reform(pts, 3, n_pts/3, /OVERWRITE)
  
  if (n_elements(outSpc) eq 0) then outSpc = spc[0:1]
  ; compute output geometry
  outTx = DicomTransform(outLoc, outOrient, outSpc, 1)
  ptsout = ApplyTransform(outTx, pts)
  min_range = min(ptsout, DIMENSION=2, MAX=max_range)
  dx = max_range[0] - min_range[0]
  dy = max_range[1] - min_range[1]
  nx = floor(dx / outSpc[0]) > 2
  ny = floor(dy / outSpc[1]) > 2
  outLoc = ApplyTransform(la_invert(outTx), [min_range[0:2],0])
  outTx = DicomTransform(outLoc[0:2], outOrient, outSpc, 1)
  x = lindgen(1,nx*ny) mod nx
  y = lindgen(1,nx*ny) / nx
  xy = [x,y]
  outXyz = ApplyTransform(la_invert(outTx) # tx, xy)
  ;who, reform(outXyz[0,*]), reform(outXyz[1,*]), reform(outXyz[2,*])
  ;outImg = make_array([nx, ny], TYPE=size(vol, /TYPE))
  outImg = vol[reform(outXyz[0,*]), reform(outXyz[1,*]), reform(outXyz[2,*])]  
  outImg = reform(outImg, [nx,ny])
  return, outImg    
end