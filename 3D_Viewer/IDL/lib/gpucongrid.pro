; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuCongrid.pro
;
; Congrids an array on the GPU
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
; Changes the resolution of an array, either using interpolation
; or nearest grid point sampling.
;
; :Returns:
;    { GPUHANDLE }
; 
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       GPU array to change resolution of. Currently only supports 2D
;    nx : in, required, type=long
;       X resolution of resulting image
;    ny : in, required, type=long
;       Y resolution of resulting image
;
; :Keywords:
;    INTERP : in, optional, type=integer
;       if set, use bilinear interpolation. Otherwise use nearest grid point.
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuCongrid, x_gpu, nx, ny, INTERP=interp, LHS=lhs, ERROR=err
  on_error, 2

  err = 0

  if (size(lhs, /TYPE) eq 8L) then begin
    if (lhs.type ne x_gpu.type) then begin            
      gpuFix, lhs, _lhs, TYPE=result_type
    endif else begin
      _lhs = lhs
    endelse
  endif else begin
    _lhs = gpuFltarr(nx, ny, /NOZERO)
    _lhs.isTemporary = 1B
    _lhs.n_dimensions = x_gpu.n_dimensions
    _lhs.dimensions   = x_gpu.dimensions
  endelse

  if (!gpu.mode eq 0) then begin
    *_lhs.data = congrid(*x_gpu.data, nx, ny, INTERP=keyword_set(interp))
  endif else begin
    err = gpuCongrid2DF(x_gpu.dimensions[0], x_gpu.dimensions[1], x_gpu.handle, $
                        _lhs.dimensions[0], _lhs.dimensions[1], _lhs.handle, $
                        long(keyword_set(interp)))
  endelse

  if (arg_present(x_gpu)) then x_gpu.isTemporary = 0B
  if (x_gpu.isTemporary) then gpuFree, x_gpu                      
         
  return, _lhs
end


;+
; Changes the resolution of an array, either using interpolation
; or nearest grid point sampling.
;
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       GPU array to change resolution of. Currently only supports 2D
;    nx    : in, required, type=long
;       X resolution of resulting image
;    ny    : in, required, type=long
;       Y resolution of resulting image
;    res_gpu : out, required, type={ GPUHANDLE }
;       resulting array.
;
; :Keywords:
;    INTERP : in, optional, type=integer
;       if set, use bilinear interpolation. Otherwise use nearest grid point.
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuCongrid, x_gpu, nx, ny, res_gpu, INTERP=interp, ERROR=err
  on_error, 2

  res_gpu = gpuCongrid(arg_present(x_gpu) ? x_gpu : (size(x_gpu, /n_dimensions) gt 0 ? x_gpu[*] : x_gpu[0]), $
                       nx, ny, LHS=res_gpu, INTERP=interp, ERROR=err)
end

