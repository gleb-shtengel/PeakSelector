; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuGetArr.pro
;
; Transfers a GPU array to host memory
;
; Copyright (C) 2008 Tech-X Corporation. All rights reserved.
;
; This file is part of GPULib.
;
; This file may be distributed under the terms of the GNU Affero General Public
; License (AGPL). This file may be distributed and/or modified under the
; terms of the GNU Affero  General Public License version 3 as published by the
; Free Software Foundation.
;
; This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
; WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
;
; Licensees holding valid Tech-X commercial licenses may use this file
; in accordance with the Tech-X Commercial License Agreement provided
; with the Software.
;
; See http://gpulib.txcorp.com/ or email sales@txcorp.com for more information.
;
; This work was in part funded by NASA SBIR Phase II Grant #NNG06CA13C.
;
;-----------------------------------------------------------------------------


;+
; Transfer GPU variables to IDL.
;
; :Returns:
;     normal IDL variable of the same type/size as x_gpu
;
; :Params:
;    x_gpu : in, optional, type={ GPUHANDLE }
;       GPU variable
;
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuGetArr, x_gpu, ERROR=err
  on_error, 2

  err = 0
  nx = x_gpu.n_elements

  if (size(x, /type) eq 0) then begin
    x = make_array(nx, type=x_gpu.type, /NOZERO)
    if (x_gpu.n_dimensions eq 2) then x = reform(x, x_gpu.dimensions)
    new_alloc = 1
  endif

;  if size(x, /type) ne size(float(1.), /type) then begin
;    if n_elements(new_alloc) ne 0 then dummy = temporary(x)
;    message, level=-1, 'getGetArr: Incompatible types'
;  endif

  if (size(x, /type) ne x_gpu.type) then x = 0

  if (n_elements(x) ne nx) then begin
    x = make_array(nx, type=x_gpu.type, /NOZERO)
    ;x = fltarr(nx, /NOZERO)
    x = reform(x, x_gpu.dimensions > 1)
  endif

  if (!gpu.mode eq 0) then begin
    x = *x_gpu.data
  endif else begin
    case x_gpu.type  of
      4 : nbytes = 4L 
      5 : nbytes = 2 * 4L 
      6 : nbytes = 2 * 4L
      9 : nbytes = 2 * 2 * 4L
    end
    
    err = cublasGetVector(nx, nbytes, x_gpu.handle, 1L, x, 1L)
  endelse

  if (arg_present(x_gpu)) then x_gpu.isTemporary = 0B
  if (x_gpu.isTemporary) then gpuFree, x_gpu           
      
  return, x
end


;+
; Transfer GPU variables to IDL.
;
; :Params:
;    x_gpu : in, optional, type={ GPUHANDLE }
;       GPU variable
;    x : in, out, required, type=any
;       pre-allocated normal IDL variable of the same type/size as x_gpu
;
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuGetArr, x_gpu, x, ERROR=err
  on_error, 2

  x = gpuGetArr(x_gpu, ERROR=err)
end

