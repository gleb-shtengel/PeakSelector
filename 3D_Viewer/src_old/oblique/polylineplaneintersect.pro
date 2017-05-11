;+
; Find the intersection point between a line and a plane.
;
; @Returns
;  The intersection points or scalar 0 if none
;
; @Param line {in}{required}
;  Set this to a 3x2 array of two points along the line.
;
; @Param planeEq {in}{required}
;  Set this to a 4-element array representing the plane equation:
;  [a,b,c,d] => ax + by + cz + d = 0
;
;
; @History
;  Feb-2005, AB, RSI - Original <br>
;-
function PolylinePlaneIntersect, poly, planeEq, CLOSED=closed
  compile_opt idl2, logical_predicate

  dims = size(poly, /DIMENSIONS)
  curr_xy = keyword_set(closed) ? poly[*,dims[1]-1] : poly[*,0]
  startInd = ~keyword_set(closed)
  
  n_pts = 0
  for i=startInd, dims[1]-1 do begin
    last_xy = curr_xy
    curr_xy = poly[*,i]
    length = sqrt(total((curr_xy - last_xy)^2.0))
    if (length eq 0) then continue
    lnDir = (curr_xy - last_xy) / length
    denominator = total(planeEq[0:2] * lnDir)
    ; near parallel
    if (abs(denominator) lt 1d-7) then continue

    k = -(planeEq[3] + total(planeEq[0:2] * last_xy)) / denominator
    if (k ge 0 && k lt length) then Padd, pts, last_xy + k*lnDir, n_pts
  endfor
  if (n_pts eq 0) then return, 0
  if (n_pts ne n_elements(pts)) then pts = pts[0:n_pts-1]
  pts = reform(pts, 3, n_pts/3, /OVERWRITE)
  return, pts
end