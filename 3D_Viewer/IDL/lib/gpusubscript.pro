; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpusubscript.pro
;
; Subscripts a GPU array with another GPU array.
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
; Subscripts a GPU array with another GPU array
;
; :Returns:
;    { GPUHANDLE }
;    
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       GPU array representing the indices
;    y_gpu : in, required, type={ GPUHANDLE }
;       GPU array representing the array to be subscripted
;
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuSubscript, x_gpu, y_gpu, LHS=lhs, ERROR=err
  on_error, 2

  x = size(x_gpu, /TYPE) eq 8L ? x_gpu : gpuPutArr(x_gpu)
  y = size(y_gpu, /TYPE) eq 8L ? y_gpu : gpuPutArr(y_gpu)
  
  if (size(lhs, /TYPE) eq 8L) then begin
    _lhs = lhs
  endif else begin
    _lhs = gpuFltarr(x.n_elements, /NOZERO)
    _lhs.isTemporary = 1B
  endelse
 
  if (!gpu.mode eq 0) then begin
    *_lhs.data = (*y.data)[*x.data]
  end else begin
    err = gpuSubscriptF(x.n_elements, x.handle, y.handle, _lhs.handle)
  end

  if (size(x_gpu, /TYPE) ne 8) then begin
    gpuFree, x
  endif else begin
    if (arg_present(x_gpu)) then x_gpu.isTemporary = 0B
    if (x_gpu.isTemporary) then gpuFree, x_gpu   
  endelse
  
  if (size(y_gpu, /TYPE) ne 8) then begin
    gpuFree, y
  endif else begin
    if (arg_present(y_gpu)) then y_gpu.isTemporary = 0B
    if (y_gpu.isTemporary) then gpuFree, y_gpu   
  endelse
  
  return, _lhs
end


;+
; Subscripts a GPU array with another GPU array
;
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       GPU array representing the indices
;    y_gpu : in, required, type={ GPUHANDLE }
;       GPU array representing the array to be subscripted
;    res_gpu : out, required, type={ GPUHANDLE }
;       GPU array conaining values of y_gpu subscripted by x_gpu
;
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuSubscript, x_gpu, y_gpu, res_gpu, ERROR=err
 on_error, 2

  if size(x_gpu, /type) ne 8 then gpuPutArr, x_gpu, x else x = x_gpu
  if size(y_gpu, /type) ne 8 then gpuPutArr, y_gpu, y else y = y_gpu
  if size(res_gpu, /type) ne 8 then begin
    res_gpu = gpuFltarr(x.n_elements, /NOZERO)
    new_alloc = 1
  end

  if x.n_elements ne res_gpu.n_elements then begin
    if size(x_gpu, /type) ne 8 then gpuFree, x
    if size(y_gpu, /type) ne 8 then gpuFree, y
    if n_elements(new_alloc) ne 0 then gpuFree, res_gpu
    message, level=-1, 'gpuSubscript: vector length missmatch'
  endif
 
  if (!gpu.mode eq 0) then begin
    *res_gpu.data = (*y.data)[*x.data]
  end else begin
    err = gpuSubscriptF(x.n_elements, x.handle, y.handle, res_gpu.handle)
  end

  if (size(x_gpu, /TYPE) ne 8) then begin
    gpuFree, x
  endif else begin
    if (arg_present(x_gpu)) then x_gpu.isTemporary = 0B
    if (x_gpu.isTemporary) then gpuFree, x_gpu   
  endelse
  
  if (size(y_gpu, /TYPE) ne 8) then begin
    gpuFree, y
  endif else begin
    if (arg_present(y_gpu)) then y_gpu.isTemporary = 0B
    if (y_gpu.isTemporary) then gpuFree, y_gpu   
  endelse
end


