; docformat = 'rst'

;+
; Create a test case for the GPU library.
;
; :Returns:
;    1 for success, 0 for failure
;    
; :Keywords:
;    _extra : in, optional, type=keywords
;       keyword to MGutTestCase::init
;-
function gpuuttestcase::init, _extra=e
  compile_opt strictarr
  
  if (~self->mguttestcase::init(_extra=e)) then return, 0
  
  self.tolerance = 1e-6
 
  return, 1
end


;+
; Define instance variables.
;-
pro gpuuttestcase__define
  compile_opt strictarr
  
  define = { GPUutTestCase, inherits MGutTestCase, tolerance: 0.0 }
end