; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuTotal.pro
;
; compute the sum of the elements in a vector
;
; Copyright (C) 2007 Tech-X Corporation.  All rights reserved.
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
;-----------------------------------------------------------------------------

;+
; Calculates the total of the elements of the GPU vector.
;
; :Returns: 
;    float or { GPUHANDLE }
;
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       GPU array to total
;
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuTotal, x_gpu, CUMULATIVE=cumulative, LHS=lhs, ERROR=err
  compile_opt strictarr
  
  if (keyword_set(cumulative)) then return, gpuPrefixSum(x_gpu, LHS=lhs, ERROR=err)
  
  err = 0
  res = 0.0

  x = size(x_gpu, /type) eq 8 ? x_gpu : gpuPutArr(x_gpu)

  if (!gpu.mode eq 0) then begin
    res = total(*x.data)
  endif else begin
    err = gpuTotalF(x.n_elements, x.handle, res)
  endelse

  if (size(x_gpu, /type) ne 8) then begin
    gpuFree, x
  endif else begin
    if (arg_present(x_gpu)) then x_gpu.isTemporary = 0B
    if (x_gpu.isTemporary) then gpuFree, x_gpu 
  endelse
  
  return, res
end


