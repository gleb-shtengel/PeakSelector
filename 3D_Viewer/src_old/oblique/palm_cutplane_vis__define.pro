pro palm_cutplane_vis_popup_event, sEvent
  compile_opt idl2, logical_predicate
  widget_control, sEvent.handler, GET_UVALUE=obj
  if (obj_valid(obj)) then obj->PopupEvent, sEvent
end

pro palm_cutplane_vis::CreatePopup
  compile_opt idl2, logical_predicate
  if (obj_valid(self.oSliceImage)) then return
  self.popup = widget_base(XOFFSET=self.popup_pos[0], $
    YOFFSET=self.popup_pos[1], UVALUE=self, $
    /TLB_MOVE_EVENTS, /TLB_SIZE_EVENTS, TITLE='Oblique Slice Plane')
  w = widget_draw(self.popup, XSIZE=self.popup_size[2], $
    YSIZE=self.popup_size[3], GRAPHICS_LEVEL=2, RETAIN=0, /EXPOSE_EVENTS, $
    /BUTTON_EVENTS, /MOTION_EVENTS)
  widget_control, self.popup, /REALIZE
  widget_control, w, GET_VALUE=oWindow
  
  oView = obj_new('IDLgrView', COLOR=[0,0,0])
  oModel = obj_new('IDLgrModel')
  oView->Add, oModel
  self.oSliceImage = obj_new('IDLgrImage', LOCATION=[-1,-1,0], $
    DIMENSIONS=[2,2])
  oModel->Add, self.oSliceImage
  oWindow->SetProperty, GRAPHICS_TREE=oView
  self.oSliceWin = oWindow
  xmanager, 'palm_cutplane_vis_popup', self.popup, /NO_BLOCK
end

pro palm_cutplane_vis::Cleanup
  ;stop
  ;help, /TRACE
end

pro palm_cutplane_vis::CreateSphere
  compile_opt idl2, logical_predicate
  
  n_phi = 40
  n_rad = 5
  phi = findgen(1,n_phi) * (2*!pi/(n_phi-1))
  rad = findgen(n_rad) / (n_rad-1) > 0.1
  nv = 0
  nc = 0
  n_lin = 8
  theta = findgen(n_lin) * (!pi*2/n_lin)
  for i=0, n_rad-1 do begin
    tmp_xy = [rad[i] * cos(phi), rad[i] * sin(phi)]
    Padd, vert, reform(tmp_xy, 2*n_phi), nv
    Padd, conn, [n_phi, lindgen(n_phi)+i*n_phi], nc
  endfor
  for i=0, n_lin-1 do begin
    pt = [cos(theta[i]), sin(theta[i])]
    Padd, vert, [pt, 0,0], nv
    Padd, conn, [2, nv/2-2, nv/2-1], nc
  endfor
  if (nv ne n_elements(vert)) then vert = vert[0:nv-1]
  vert = reform(vert, 2, nv/2)
     
  oLine = obj_new('IDLgrPolyline', vert, POLYLINE=conn, $
    COLOR=[0,255,0])
    
  self.oPoint = obj_new('IDLgrPolygon', vert[*,0:n_phi-2], $
    COLOR=[255,0,0])
  self.oHemi = obj_new('IDLgrModel')
  self.oHemi->Add, [oLine, self.oPoint]
end

pro palm_cutplane_vis::Draw, oWin, oView
  compile_opt idl2, logical_predicate
  self.oWin = oWin
  self.oView = oView
  self->IDLgrModel::Draw, oWin, oView
end

pro palm_cutplane_vis::Event, sEvent, MSG=msg
  compile_opt idl2, logical_predicate

  msg = ''
  if (sEvent.type eq 0 && sEvent.press eq 1) then begin
    self->EventRotate, sEvent, MSG=msg
  endif
  if (sEvent.type eq 0 && sEvent.press eq 2) then begin
    self->EventTranslate, sEvent, MSG=msg
  endif
  if (self.mouse eq 1 && sEvent.type eq 2) then begin
    self->EventRotate, sEvent, MSG=msg
  endif
  if (sEvent.type eq 1) then begin
    if (self.mouse eq 1) then msg = 'rotate_end'
    self.msg = msg
    self.mouse = 0
    self->CreatePopup
  endif    
end

pro palm_cutplane_vis::EventRotate, sEvent, MSG=msg
  compile_opt idl2, logical_predicate
  
  ctm = self.oHemi->GetCtm(DESTINATION=self.oWin)
  self.oWin->GetProperty, DIMENSIONS=dims
  xy = [sEvent.x, sEvent.y] * (2.0/dims) - 1
  xyz = ApplyTransform(la_invert(ctm), xy)
  rad_sq = total(xyz[0:1]^2.0)
  if (sEvent.type eq 0 && rad_sq le 1) then begin
    self.mouse = 1
    self.last_pos = [xyz[0:1], sqrt(1 - rad_sq)]
    self->RotatePlane
    self.oPoint->SetProperty, XCOORD_CONV=[xyz[0],1], YCOORD_CONV=[xyz[1],1]
    msg = 'rotate_begin'
    self.msg = msg
    if (obj_valid(self.oSliceImage) && ptr_valid(self.ptr)) then begin
      data = oblique(*self.ptr, self.planeEq)
      dims = [size(data, /DIMENSIONS),1]
      if (dims[0] le 1 || dims[1] le 1) then data = bytarr(2,2)
      self.oSliceImage->SetProperty, DATA=bytscl(data)
      self.oSliceWin->Draw
    endif
  endif
  
  if (sEvent.type ne 2) then return
  if (rad_sq gt 1) then begin
    xyz /= sqrt(rad_sq)
    rad_sq = 1
  endif
  self.mouse = 1
  self.last_pos = [xyz[0:1], sqrt(1 - rad_sq)]
  self->RotatePlane
  self.oPoint->SetProperty, XCOORD_CONV=[xyz[0],1], YCOORD_CONV=[xyz[1],1]
  msg = 'rotate_motion'
  self.msg = msg
  if (obj_valid(self.oSliceImage) && ptr_valid(self.ptr)) then begin
    data = oblique(*self.ptr, self.planeEq)
    dims = [size(data, /DIMENSIONS),1]
    if (dims[0] le 1 || dims[1] le 1) then data = bytarr(2,2)
    self.oSliceImage->SetProperty, DATA=bytscl(data)
    self.oSliceWin->Draw
  endif
end

pro palm_cutplane_vis::EventTranslate, sEvent, MSG=msg
  compile_opt idl2, logical_predicate
  if (sEvent.type eq 0) then begin
    self.mouse = 2
    self.last_pos = [0,sEvent.y, 0]
    msg = 'translate_begin'
  endif
  if (sEvent.type eq 2) then begin
    delta = sEvent.y - self.last_pos[1]
    self.last_pos = [0,sEvent.y, 0]
    self->TranslatePlane, delta
    if (obj_valid(self.oSliceImage) && ptr_valid(self.ptr)) then begin
      data = oblique(*self.ptr, self.planeEq)
      dims = [size(data, /DIMENSIONS),1]
      if (dims[0] le 1 || dims[1] le 1) then data = bytarr(2,2)
      self.oSliceImage->SetProperty, DATA=bytscl(data)
      self.oSliceWin->Draw
    endif
  endif
end

pro palm_cutplane_vis::GetProperty, HEMI_SPHERE=oHemi, $
    PLANE_EQUATION=planeEq, MESSAGE=msg, $
    _REF_EXTRA=extra
  compile_opt idl2, logical_predicate
  
  if arg_present(msg) then msg = self.msg
  if arg_present(planeEq) then planeEq = self.planeEq
  if arg_present(oHemi) then oHemi = self.oHemi
  if n_elements(extra) then self->IDLgrModel::GetProperty, _STRICT_EXTRA=extra
end 

function palm_cutplane_vis::Init, _REF_EXTRA=extra
  compile_opt idl2, logical_predicate
  
  self.popup_size = [0,0,500,500]
  self.corners = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],$
    [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
  
  if (~self->IDLgrModel::Init()) then return, 0
  
  self.oPlane = obj_new('IDLgrPolygon', STYLE=2, COLOR=[0,255,0], $
    ALPHA=0.75)
  self.oBox = obj_new('IDLgrPolygon', STYLE=1, COLOR=[0,255,0], $
    THICK=2)
  self.oInter = obj_new('IDLgrPolygon', STYLE=1, COLOR=[0,255,0], $
    THICK=2)
    
  self->CreateSphere
  self->Add, [self.oPlane, self.oBox, self.oInter]
  
  if (n_elements(extra)) then self->SetProperty, _STRICT_EXTRA=extra
  return, 1
end

pro palm_cutplane_vis::OnMouseMotion, oWindow, x, y, ButtonMask, Modifiers, NumClicks
  compile_opt idl2, logical_predicate
  sEvent = { widget_draw }
  sEvent.x = x
  sEvent.y = y
  sEvent.type = 2
  self->Event, sEvent
end
pro palm_cutplane_vis::OnMouseUp, oWindow, x, y, ButtonMask, Modifiers, NumClicks
  compile_opt idl2, logical_predicate
  sEvent = { widget_draw }
  sEvent.x = x
  sEvent.y = y
  sEvent.type = 1
  self->Event, sEvent
end
pro palm_cutplane_vis::OnKeyboard, oWindow, IsASCII, Character, KeySymbol, x, y, Press, Release, Modifiers
end
pro palm_cutplane_vis::OnWheel, oWindow, X, Y, Delta, Modifiers
  compile_opt idl2, logical_predicate

  if (ptr_valid(self.ptr)) then begin
    dim = size(*self.ptr, /DIMENSIONS)
    step = (self.corners[4] - self.corners[0]) / dim[0]
    self->TranslatePlane, delta*step
  endif
end

pro palm_cutplane_vis::PopupEvent, sEvent
  compile_opt idl2, logical_predicate
  type = tag_names(sEvent, /STRUCTURE_NAME)
  if (type eq 'WIDGET_TLB_MOVE') then begin
    self.popup_pos = [sEvent.x, sEvent.y]
  endif
  if (type eq 'WIDGET_BASE') then begin
    delta = [sEvent.x, sEvent.y] - self.popup_size[0:1]
    self.oSliceWin->GetProperty, DIMENSIONS=dims
    dims += delta
    self.oSliceWin->SetProperty, DIMENSIONS=dims
    self.popup_size = [sEvent.x, sEvent.y, long(dims)]
  endif
  if (type eq 'WIDGET_DRAW') then begin
    if (sEvent.type eq 0 && sEvent.press eq 1) then begin
      self->EventTranslate, sEvent, MSG=msg
    endif
    if (self.mouse eq 2 && sEvent.type eq 2) then begin
      msg = ''
      self->EventTranslate, sEvent, MSG=msg
      self.oWin->Draw
    endif
    if (sEvent.type eq 1) then begin
      self.mouse = 0
    endif
      
  endif
end

pro palm_cutplane_vis::RotatePlane
  compile_opt idl2, logical_predicate
  ctm = self->GetCtm()
  pos = self.last_pos
  ctm[3,*] = [0,0,0,1]
  ctm[*,3] = [0,0,0,1]
  for i=0, 2 do ctm[i,*] /= sqrt(total(ctm[i,*]^2))
  xyz = ApplyTransform(transpose(ctm), [pos,0])
  
  self.planeEq[0:2] = xyz[0:2] / sqrt(total(xyz[0:2]^2.0))
  distance = total(self.planeEq * [self.refpt, 1])
  self.planeEq[3] -= distance
  self->UpdatePoly
end

pro palm_cutplane_vis::TranslatePlane, delta
  compile_opt idl2, logical_predicate

  self.planeEq[3] += delta
  self.refPt += delta * self.planeEq[0:2]
  self->UpdatePoly
end
  
pro palm_cutplane_vis::SetProperty, VOLUME_CORNERS=corners, $
    PLANE_EQUATION=planeEq, CENTER=center, RADIUS=radius, $
    REF_POINT=refpt, $
    _REF_EXTRA=extra
  compile_opt idl2, logical_predicate
  
  if n_elements(corners) then self.corners = corners
  if n_elements(planeEq) then self.planeEq = planeEq
  if n_elements(refPt) then self.refpt = refpt
  if n_elements(extra) then self->IDLgrModel::SetProperty, _STRICT_EXTRA=extra
end

pro palm_cutplane_vis::SetVolPtr, ptr
  compile_opt idl2, logical_predicate
  self.ptr = ptr
end
pro palm_cutplane_vis::UpdateDot
  compile_opt idl2, logical_predicate

  ctm = self->GetCtm()
  ctm[3,*] = [0,0,0,1]
  ctm[*,3] = [0,0,0,1]
  for i=0, 2 do ctm[i,*] /= sqrt(total(ctm[i,*]^2))
  pt = ApplyTransform(ctm, [self.planeEq[0:2],0])
  if (pt[2] lt 0) then pt = -pt
  self.oPoint->SetProperty, XCOORD_CONV=[pt[0],1], YCOORD_CONV=[pt[1],1]
end

pro palm_cutplane_vis::UpdatePoly
  compile_opt idl2, logical_predicate
  
  ; arbitrary coord system
  void = min(abs(self.planeEq[0:2]), mindex)
  y_axis = crossp(self.planeEq[0:2], [0,1,2] eq mindex)
  y_axis /= sqrt(total(y_axis^2.0))
  x_axis = crossp(self.planeEq[0:2], y_axis)
  orient = [x_axis, y_axis]
  ; pick a point in the plane
  max_ax = max(self.planeEq[0:2], /ABSOLUTE, mxindex)
  loc = ([0,1,2] eq mxindex) * (-self.planeEq[3] / max_ax)
  tx = DicomTransform(loc, orient, [1.,1], 1.)
  self.tx = tx
  
  ; project ref point to plane
  ref = ApplyTransform(tx, self.refpt)
  if (ref[2] ne 0) then begin
    ref[2] = 0
    pt = ApplyTransform(la_invert(tx), ref[0:2])
    self.refpt = pt[0:2]
  endif    
  
  ; project points
  pts = ApplyTransform(tx, self.corners)
  min_range = min(pts, DIMENSION=2, MAX=max_range)
  box = [[min_range[0:1]], [max_range[0], min_range[1]], $
    [max_range[0:1]], [min_range[0], max_range[1]]]
  box = ApplyTransform(la_invert(tx), box) 
  self.oPlane->SetProperty, DATA=box[0:2,*]
  self.oBox->SetProperty, DATA=box[0:2,*]
end

pro palm_cutplane_vis__define
  compile_opt idl2, logical_predicate
  s = { palm_cutplane_vis $
      , inherits IDLgrModel $
      , oPlane: obj_new() $
      , oBox: obj_new() $
      , oInter: obj_new() $
      , oHemi: obj_new() $
      , oPoint: obj_new() $
      , corners: fltarr(3,8) $
      , planeEq: fltarr(4) $
      , mouse: 0 $
      , oWin: obj_new() $
      , oView: obj_new() $
      , last_pos: fltarr(3) $
      , center: fltarr(2) $
      , radius: 0.0 $
      , refpt: fltarr(3) $
      , tx: fltarr(4,4) $
      , msg: '' $
      , ptr: ptr_new() $
      , popup: 0 $
      , popup_pos: [0,0] $
      , popup_size: [0,0,0,0] $
      , oSliceWin: obj_new() $
      , oSliceImage: obj_new() $
      }
end
