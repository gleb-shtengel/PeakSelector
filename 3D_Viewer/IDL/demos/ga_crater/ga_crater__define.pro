; docformat = 'rst'

;+
; Represents a crater rim.
;-


;+
; Calculates the fitness function, in this case, the cross-correlation with the
; background image.
;
; :Returns: float
;-
function ga_crater::fitness
  compile_opt strictarr
  
  return, self.fitness
end


;+
; Mutate this individual's solution slightly.
;-
pro ga_crater::mutate
  compile_opt strictarr
  
  r = randomu(seed, 3)
  if (r[0] lt 0.5) then begin
    r = 0.01 * r - 0.005
    self.xpos += r[1]  
    self.ypos += r[2]
  endif else begin
    r = 1. + 0.001 * r - 0.0005
    self.xAxis *= r[1]
    self.yAxis *= r[2]
  endelse
  
  self->generateIndices
  
  if (self.display) then self->display, channel=2
end


;+
; Create an offspring between this individual and the given individual.
;
; :Returns: 
;    individual object representing the offspring between this individual and
;    the specified individual
;-
function ga_crater::reproduce, individual
  compile_opt strictarr
    
  child = obj_new('GA_Crater', image=self.pImage, $
                  x=self.xPtr, y=self.yPtr, $
                  /no_generate_indices, display=self.display)

  child.xAxis = self.xAxis
  child.yAxis = self.yAxis
  child.thickness = self.thickness                  
  child.xpos = self.xpos
  child.ypos = self.ypos
  
  child->mutate  
  
  return, child 
end



pro ga_crater::generateIndices
  compile_opt strictarr

  outside = ((*self.xPtr - self.xpos) / self.xAxis) ^ 2 $
              + ((*self.yPtr - self.ypos) / self.yAxis) ^ 2 lt 1.
  inside = ((*self.xPtr - self.xpos) / (self.xAxis - self.thickness)) ^ 2 $
             + ((*self.yPtr - self.ypos) / (self.yAxis - self.thickness)) ^ 2 lt 1.

  rimInd = where(outside - inside gt 0, nPtsOnRim)  
  insideInd = where(inside gt 0, nPtsInside)
  
  self.indices = ptr_new(rimInd, /no_copy)
  
  ; find the ratio of rim pixel values to interior pixel values
  self.fitness = (*self.indices)[0] eq -1 || (insideInd)[0] eq -1 $
                   ? 0.0 $
                   : (mean((*self.pImage)[*self.indices]) / mean((*self.pImage)[insideInd]))
   
  ; find the mean value of the rim pixels                
  ;self.fitness = (*self.indices)[0] eq -1 ? 0.0 : mean((*self.pImage)[*self.indices])
end


pro ga_crater::display, _extra=e
  compile_opt strictarr
  
  if ((*self.indices)[0] eq -1) then return
  
  dims = size(*self.xPtr, /dimensions)
  
  answer = bytarr(dims[0], dims[1])
  answer[*self.indices] = 1B
  tvscl, answer, _strict_extra=e
end


;+
; Get properties.
;-
pro ga_crater::getProperty, indices=indices, x_pos=xpos, y_pos=ypos, $
                            x_axis=xAxis, y_axis=yAxis, _ref_extra=e
  compile_opt strictarr
  
  if (arg_present(indices)) then indices = *self.indices
  
  if (arg_present(xpos)) then xpos = self.xpos
  if (arg_present(ypos)) then ypos = self.ypos  

  if (arg_present(xAxis)) then xAxis = self.xAxis
  if (arg_present(yAxis)) then yAxis = self.yAxis
  
  if (n_elements(e) gt 0) then begin
    self->ga_individual::getproperty, _strict_extra=e
  endif
end


;+
; Set properties.
;-
pro ga_crater::setProperty, _ref_extra=e
  compile_opt strictarr
  
  if (n_elements(e) gt 0) then begin
    self->ga_individual::setproperty, _strict_extra=e
  endif
end


;+
; Free resources.
;-
pro ga_crater::cleanup
  compile_opt strictarr
  
  ptr_free, self.indices
  
  self->ga_individual::cleanup
end


;+
; Create ga_crater object.
;
; :Returns: 1 for success, 0 for failure
;
; :Keywords:
;    image : in, required, type
;       pointer to base image
;-
function ga_crater::init, image=pImage, x=xPtr, y=yPtr, $
                          display=display, no_generate_indices=noGenerateIndices
  compile_opt strictarr
  
  if (~self->ga_individual::init()) then return, 0
  
  self.pImage = pImage
  
  dims = size(*self.pImage, /dimensions)
  self.xPtr = xPtr
  self.yPtr = yPtr
  self.display = keyword_set(display)
  
  if (~keyword_set(noGenerateIndices)) then begin  
    self.thickness = 4.0 / (min([dims[0], dims[1]]))
    
    r = randomu(seed, 4)
    
    ; this should probably be some sort of normal curve, not sure of the parameters
    self.xAxis = 0.1 * r[0] * dims[1] / dims[0]
    self.yAxis = 0.1 * r[0]
    
    ; equally scattered over the image background; seems fair
    self.xpos = 2.0 * r[2] - 1.0
    self.ypos = 2.0 * r[3] - 1.0
    
    self->generateIndices
    if (self.display) then self->display, channel=3
  endif
  
  return, 1
end


;+
; Define instance variables.
;
; :Fields:
;
;-
pro ga_crater__define
  compile_opt strictarr
  
  define = { ga_crater, inherits ga_individual, $             
             pImage: ptr_new(), $
             xPtr: ptr_new(), $
             yPtr: ptr_new(), $
             xAxis: 0.0, $
             yAxis: 0.0, $             
             thickness: 0.0, $
             xpos: 0.0, $
             ypos: 0.0, $
             indices: ptr_new(), $
             display: 0B $
           }
end