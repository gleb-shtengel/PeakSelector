; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuprefixsum.pro
;
; Computes the prefix sum on the GPU.
;
; Copyright (C) 2008 Tech-X Corporation. All rights reserved.
;
; This file is part of GPULib.
;
; This file may be distributed under the terms of the GNU General Public
; License (GPL). This file may be distributed and/or modified under the
; terms of the GNU General Public License version 2 as published by the
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
; Computes the prefix sum on a GPU array
;
; :Returns:
;    { GPUHANDLE }
;    
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       GPU array to be summed
;
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuPrefixSum, x_gpu, LHS=lhs, ERROR=err
  compile_opt strictarr
  on_error, 2

  x = size(x_gpu, /type) eq 8 ? x_gpu : gpuPutArr(x_gpu)

  if (size(lhs, /type) eq 8) then begin
    if (lhs.type ne x.type) then begin            
      gpuFix, lhs, _lhs, TYPE=x.type
    endif else begin
      _lhs = lhs
    endelse
  endif else begin
    _lhs = gpuMake_Array(x.dimensions, TYPE=result_type, /NOZERO)
    _lhs.isTemporary = 1B
  endelse

  if (!gpu.mode eq 0) then begin
    *_lhs.data = total(*x.data, /cumulative)
  endif else begin
    err = gpuPrefixSumF(x.n_elements, x.handle, _lhs.handle)
    _lhs = gpuAdd(_lhs, x, LHS=_lhs)
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
; Computes the prefix sum on a GPU array
;
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       GPU array to be summed
;    res_gpu : out, required, type={ GPUHANDLE }
;       GPU array conaining the result
;
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuPrefixSum, x_gpu, res_gpu, ERROR=err
  on_error, 2

  res_gpu = gpuPrefixSum(arg_present(x_gpu) ? x_gpu : (size(x_gpu, /n_dimensions) gt 0 ? x_gpu[*] : x_gpu[0]), $
                         LHS=res_gpu, ERROR=err)
end


