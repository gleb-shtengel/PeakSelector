; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuimaginary.pro
;
; Extracts the imaginary part of complex numbers on the GPU.
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
; Extracts the imaginary part of a complex data object on the GPU
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
function gpuImaginary, x_gpu, LHS=lhs, ERROR=err
  on_error, 2

  x = size(x_gpu, /type) eq 8L ? x_gpu : gpuPutArr(x_gpu)
  result_type = x.type eq 4 or x.type eq 6 ? 4 : 5

  if (size(lhs, /TYPE) eq 8L) then begin
    if (lhs.type ne result_type) then begin            
      gpuFix, lhs, _lhs, TYPE=result_type
    endif else begin
      _lhs = lhs
    endelse
  endif else begin
    case x.type of 
      4 : _lhs = gpuFltarr(x.n_elements)
      5 : _lhs = gpuDblarr(x.n_elements)
      6 : _lhs = gpuFltarr(x.n_elements, /NOZERO)
      9 : _lhs = gpuDblarr(x.n_elements, /NOZERO)
    endcase
    _lhs.isTemporary = 1B
  endelse

  if (!gpu.mode eq 0) then begin
    *_lhs.data = imaginary(*x.data)
  end else begin
    case x.type of
      4: gpuCopy, x, _lhs
      5: gpuCopy, x, _lhs
      6: err = gpuComplexImagToFloat(x.n_elements, x.handle, _lhs.handle)
      9: err = gpuDcomplexImagToDouble(x.n_elements, x.handle, _lhs.handle)
    endcase 
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
; Extracts the imaginary part of a complex data object on the GPU
;
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       GPU array representing a complex array
;    res_gpu : out, required, type={ GPUHANDLE }
;       GPU array conaining the imaginary part of the input
;
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuImaginary, x_gpu, res_gpu, ERROR=err
  on_error, 2

  res_gpu = gpuImaginary(x_gpu, LHS=res_gpu, ERROR=err)
end

