; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuReal.pro
;
; Extracts the real part of a complex number on the GPU
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
; Extracts the real part of a complex data object on the GPU
;
; :Returns:
;    { GPUHANDLE }
;
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       GPU array representing a complex array
;
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuReal, x_gpu, LHS=lhs, ERROR=err
  on_error, 2

  x = size(x_gpu, /TYPE) eq 8L ? x_gpu : gpuPutArr(x_gpu)
  
  if (size(lhs, /TYPE) eq 8) then begin
    _lhs = lhs
  endif else begin
    case x.type of 
      4 : _lhs = gpuFltarr(x.n_elements, /NOZERO)
      5 : _lhs = gpuDblarr(x.n_elements, /NOZERO)
      6 : _lhs = gpuFltarr(x.n_elements, /NOZERO)
      9 : _lhs = gpuDblarr(x.n_elements, /NOZERO)
    endcase
    _lhs.isTemporary = 1B
  endelse

  if (!gpu.mode eq 0) then begin
    *_lhs.data = real_part(*x.data)
  end else begin
    case x.type of
      4: err = cublasSScopy(x.n_elements, x.handle, _lhs.handle)
      5: err = cublasSDcopy(x.n_elements, x.handle, _lhs.handle)
      6: err = gpuComplexRealToFloat(x.n_elements, x.handle, _lhs.handle)
      9: err = gpuDcomplexRealToDouble(x.n_elements, x.handle, _lhs.handle)
    endcase  
  end

  if (size(x_gpu, /type) ne 8) then begin
    gpuFree, x
  endif else begin
    if (arg_present(x_gpu)) then x_gpu.isTemporary = 0B
    if (x_gpu.isTemporary) then gpuFree, x_gpu  
  endelse
  
  return, _lhs
end


;+
; Extracts the real part of a complex data object on the GPU
;
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       GPU array representing a complex array
;    res_gpu : out, required, type={ GPUHANDLE }
;       GPU array conaining the real part of the input
;
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuReal, x_gpu, res_gpu, ERROR=err
  on_error, 2

  if size(x_gpu, /type) ne 8 then gpuPutArr, x_gpu, x else x = x_gpu
  if size(res_gpu, /type) ne 8 then begin
    case x.type of 
      4 : res_gpu = gpuFltarr(x.n_elements, /NOZERO)
      5 : res_gpu = gpuDblarr(x.n_elements, /NOZERO)
      6 : res_gpu = gpuFltarr(x.n_elements, /NOZERO)
      9 : res_gpu = gpuDblarr(x.n_elements, /NOZERO)
    endcase
    new_alloc  = 1
  end

  if x.n_elements ne 2*res_gpu.n_elements then begin
    if size(x_gpu, /type) ne 8 then gpuFree, x
    if n_elements(new_alloc) ne 0 then gpuFree, res_gpu
    message, level=-1, 'gpuReal: vector length missmatch'
  endif

  if (!gpu.mode eq 0) then begin
   *res_gpu.data = (*x.data)[0:*:2]
  end else begin
    case x.type of
      4: err = cublasSScopy(x.n_elements, x.handle, res_gpu.handle)
      5: err = cublasSDcopy(x.n_elements, x.handle, res_gpu.handle)
      6: err = gpuComplexRealToFloat(x.n_elements, x.handle, res_gpu.handle)
      9: err = gpuDcomplexRealToDouble(x.n_elements, x.handle, res_gpu.handle)
    endcase 
  end

  if (size(x_gpu, /type) ne 8) then begin
    gpuFree, x
  endif else begin
    if (arg_present(x_gpu)) then x_gpu.isTemporary = 0B
    if (x_gpu.isTemporary) then gpuFree, x_gpu  
  endelse
end

