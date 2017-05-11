; docformat = 'rst'

;+
; Calculates the fitness function, in this case, the cross-correlation with the
; background image.
;
; :Returns: float
;-
function ga_gpu_crater::fitness
  compile_opt strictarr
  
  return, self.fitness
end


;+
; Mutate this individual's solution slightly.
;-
pro ga_gpu_crater::mutate
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
function ga_gpu_crater::reproduce, individual
  compile_opt strictarr
    
  child = obj_new('GA_GPU_Crater', image=self.image, $
                  x_gpu=self.x, y_gpu=self.y, xtemp=self.xtemp, ytemp=self.ytemp, $
                  /no_generate_indices, display=self.display)

  child.xAxis = self.xAxis
  child.yAxis = self.yAxis
  child.thickness = self.thickness                  
  child.xpos = self.xpos
  child.ypos = self.ypos
  
  child->mutate  
  
  return, child 
end


pro ga_gpu_crater::generateIndices
  compile_opt strictarr
  
  ; xtemp = x - x0
  gpuAdd, 1., self.x, 0., self.x, - self.xpos, self.xtemp  
  
  ; xtemp = (xtemp / xAxis) ^ 2
  gpuMult, 1. / self.xAxis, self.xtemp, 1. / self.xAxis, self.xtemp, 0., self.xtemp 
  
  ; ytemp = y - y0
  gpuAdd, 1., self.y, 0., self.y, - self.yAxis, self.ytemp     
   
  ; ytemp = (ytemp / yAxis) ^ 2
  gpuMult, 1. / self.yAxis, self.ytemp, 1. / self.yAxis, self.ytemp, 0., self.ytemp 
  
  ; result = xtemp + ytemp - 1.0
  gpuAdd, 1., self.xtemp, 1., self.ytemp, - 1., self.xtemp      
  
  gpuSignbit, self.xtemp, self.xtemp
  nPixels = gpuTotal(self.xtemp)
  gpuMult, self.xtemp, self.image, self.xtemp
  
  ; total
  crosscor = gpuTotal(self.xtemp)
  
  ; find the mean value of the rim pixels                
  self.fitness = crosscor / nPixels 
end


pro ga_gpu_crater::display, _extra=e
  compile_opt strictarr
  
  gpuGetArr, self.xtemp, im
   
  tvscl, im, _strict_extra=e
end


;+
; Get properties.
;-
pro ga_gpu_crater::getProperty, x_pos=xpos, y_pos=ypos, $
                                x_axis=xAxis, y_axis=yAxis, _ref_extra=e
  compile_opt strictarr
  
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
pro ga_gpu_crater::setProperty, _ref_extra=e
  compile_opt strictarr
  
  if (n_elements(e) gt 0) then begin
    self->ga_individual::setproperty, _strict_extra=e
  endif
end


;+
; Free resources.
;-
pro ga_gpu_crater::cleanup
  compile_opt strictarr
  
  self->ga_individual::cleanup
end


;+
; Create ga_gpu_crater object.
;
; :Returns: 1 for success, 0 for failure
;-
function ga_gpu_crater::init, image=image, x_gpu=x, y_gpu=y, xtemp=xtemp, ytemp=ytemp, $
                              display=display, no_generate_indices=noGenerateIndices
  compile_opt strictarr
  
  if (~self->ga_individual::init()) then return, 0
  
  self.image = image
  self.x = x
  self.y = x
  self.xtemp = xtemp
  self.ytemp = ytemp
  dims = self.image.dimensions
  
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
pro ga_gpu_crater__define
  compile_opt strictarr
  
  define = { ga_gpu_crater, inherits ga_individual, $
             image: { GPUHANDLE }, $
             x: { GPUHANDLE }, $
             y: { GPUHANDLE }, $
             xtemp: { GPUHANDLE }, $
             ytemp: { GPUHANDLE }, $
             xAxis: 0.0, $
             yAxis: 0.0, $             
             thickness: 0.0, $
             xpos: 0.0, $
             ypos: 0.0, $
             display: 0B $
           }
end