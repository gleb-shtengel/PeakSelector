; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpucopy.pro
;
; Copies an array on the GPU.
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
; Copies a GPU vector
;
; :Returns:
;    { GPUHANDLE }
;
; :Params:
;    x_gpu : in, required, type=float or { GPUHANDLE }
;       GPU array to copy. If an IDL array is provided, it is first
;       copied to the GPU.
;
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuCopy, x_gpu, LHS=lhs, ERROR=err
  on_error, 2

  x = size(x_gpu, /type) eq 8L ? x_gpu : gpuPutArr(x_gpu)
  
  if (size(lhs, /TYPE) eq 8L) then begin
    if (lhs.type ne result_type) then begin            
      gpuFix, lhs, _lhs, TYPE=result_type
    endif else begin
      _lhs = lhs
    endelse
  endif else begin
    _lhs = gpuFltarr(x.n_elements, /NOZERO)
    _lhs.isTemporary = 1B
    _lhs.dimensions = x_gpu.dimensions
    _lhs.n_dimensions = x_gpu.n_dimensions
  endelse
  
  if (!gpu.mode eq 0) then begin
    *_lhs.data = *x.data
  endif else begin
    err = cudaMemcpy(_lhs.handle, x.handle, x.n_elements * 4L, 3L)
  endelse

  if (size(x_gpu, /type) ne 8) then begin
    gpuFree, x
  endif else begin
    if (arg_present(x_gpu)) then x_gpu.isTemporary = 0B
    if (x_gpu.isTemporary) then gpuFree, x_gpu        
  endelse
  
  return, _lhs
end


;+
; Copies a GPU vector
;
; :Params:
;    x_gpu : in, required, type=float or { GPUHANDLE }
;       GPU array to copy. If and IDL array is provided, it is first
;       copied to the GPU.
;    res_gpu : in, optional, type = any
;       resulting GPU vector. If res_gpu is a {GPUHANDLE} it is used
;       for holding the result, otherwise a new GPU array is created.
;
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuCopy, x_gpu, res_gpu, ERROR=err
  on_error, 2

  res_gpu = gpuCopy(arg_present(x_gpu) ? x_gpu : (size(x_gpu, /n_dimensions) gt 0 ? x_gpu[*] : x_gpu[0]), $
                    LHS=res_gpu, ERROR=err)
end

