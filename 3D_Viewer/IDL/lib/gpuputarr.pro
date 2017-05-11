; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuPutArr.pro
;
; transfers an array from host memory to the GPU
;
; Copyright (C) 2008 Tech-X Corporation. All rights reserved.
;
; This file is part of GPULib.
;
; This file may be distributed under the terms of the GNU Affero General Public
; License (AGPL). This file may be distributed and/or modified under the
; terms of the GNU Affero General Public License version 3 as published by the
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
; Transfer IDL variables to the GPU.
;
; :Returns:
;    GPU handle
;
; :Params:
;    x : in, required, type=any
;       normal IDL variable to send to the GPU
;
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuPutArr, x, LHS=lhs, ERROR=err
  compile_opt strictarr
  on_error, 2
  err = 0
  
  xInfo = size(x, /structure)
  
  if (size(lhs, /TYPE) eq 8L) then begin
    if (lhs.type ne xInfo.type) then begin            
      gpuFix, lhs, _lhs, TYPE=xInfo.type
    endif else begin
      _lhs = lhs
    endelse
  endif else begin
    _lhs = gpuMake_Array(xInfo.n_elements, TYPE=xInfo.type)
    _lhs.isTemporary = 1B
    _lhs.n_dimensions = xInfo.n_dimensions
    _lhs.dimensions  = xInfo.dimensions[0:1]
  endelse
  
  ; scalars must be treated as arrays
  _x = xInfo.n_dimensions eq 0L ? [x] : x

  if (!gpu.mode eq 0) then begin
    *_lhs.data = _x
  endif else begin
    case _lhs.type  of
      4 : nbytes = 4L 
      5 : nbytes = 2 * 4L 
      6 : nbytes = 2 * 4L
      9 : nbytes = 2 * 2 * 4L
  end
    err = cublasSetVector(xInfo.n_elements, nbytes, _x, 1L, _lhs.handle, 1L)
  endelse
  
  return, _lhs
end


;+
; Transfer IDL variables to the GPU.
;
; :Params:
;    x : in, required, type=any
;       normal IDL variable to send to the GPU
;    x_gpu : in, out, optional, type={ GPUHANDLE }
;       GPU variable to fill of the same size/type as x
;
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuPutArr, x, x_gpu, ERROR=err
  compile_opt strictarr
  on_error, 2

  x_gpu = gpuPutArr(x, LHS=x_gpu, ERROR=err)
end

