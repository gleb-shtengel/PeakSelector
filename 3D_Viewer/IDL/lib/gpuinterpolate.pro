; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuinterpolate.pro
;
; Interpolates an array on the GPU.
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
; Calculates the linear or bilinear interpolates of the given input p_gpu,
; depending on the number of arguments presented. If three arguments are
; given::
;
;    Result = interpolate(p_gpu, x_gpu)
;
; If four arguments are given::
;
;    Result = interpolate(p_gpu, x_gpu, y_gpu)
;
; :Returns:
;    { GPUHANDLE }
; 
; :Params:
;    p_gpu : in, required, type={ GPUHANDLE }
;       the input array (1D or 2D)
;    x_gpu : in, required, type={ GPUHANDLE }
;       x-values for either form of interpolation
;    y_gpu : in, out, required, type={ GPUHANDLE }
;       the y-values for bilinear interpolation
;
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuInterpolate, p_gpu, x_gpu, y_gpu, LHS=lhs, ERROR=err
  on_error, 2

  err = 0

  if (size(lhs, /TYPE) eq 8L) then begin
    if (lhs.type ne 4) then begin            
      gpuFix, lhs, _lhs, TYPE=4
    endif else begin
      _lhs = lhs
    endelse  
  endif else begin
    _lhs = gpuMake_array(x_gpu.dimensions, TYPE=result_type, /NOZERO)
    _lhs.isTemporary = 1B
  endelse
           
  case n_params() of
    2: begin        
         if (!gpu.mode eq 0) then begin
           *_lhs.data = interpolate(*p_gpu.data, *x_gpu.data)
         endif else begin
           err = gpuInterpolate1DF(p_gpu.n_elements, p_gpu.handle,  $
                                   x_gpu.n_elements, x_gpu.handle,  $
                                   _lhs.handle)
         endelse
       end
    3: begin
         if p_gpu.n_dimensions ne 2 then begin
           message, level=-1, 'gpuInterpolate: input has to be 2D field'
         endif

         if x_gpu.n_elements ne y_gpu.n_elements then begin
           message, level=-1, 'gpuInterpolate: input vector size missmatch'
         endif

         if (!gpu.mode eq 0) then begin
           *_lhs.data = interpolate(*p_gpu.data, *x_gpu.data, *y_gpu.data)
         endif else begin
           err = gpuInterpolate2DF(p_gpu.dimensions[0], p_gpu.dimensions[1], $
                                   p_gpu.handle,  x_gpu.n_elements, $
                                   x_gpu.handle,  y_gpu.handle, _lhs.handle)
         endelse
      end
    else: message, level=-1, 'gpuInterpolate: incorrect number of arguments'
  endcase
  
  ; free p_gpu, x_gpu, y_gpu if necessary
  if (arg_present(p_gpu)) then p_gpu.isTemporary = 0B
  if (p_gpu.isTemporary) then gpuFree, p_gpu

  if (arg_present(x_gpu)) then x_gpu.isTemporary = 0B
  if (x_gpu.isTemporary) then gpuFree, x_gpu
  
  if (n_params() eq 3L) then begin
    if (arg_present(y_gpu)) then y_gpu.isTemporary = 0B
    if (y_gpu.isTemporary) then gpuFree, y_gpu
  endif

  return, _lhs
end


;+
; Calculates the linear or bilinear interpolates of the given input p_gpu,
; depending on the number of arguments presented. If three arguments are
; given::
;
;    arg3_gpu = interpolate(p_gpu, x_gpu)
;
; If four arguments are given::
;
;    arg4_gpu = interpolate(p_gpu, x_gpu, arg3_gpu)
;
; :Params:
;    p_gpu : in, required, type={ GPUHANDLE }
;       the input array (1D or 2D)
;    x_gpu : in, required, type={ GPUHANDLE }
;       x-values for either form of interpolation
;    arg3_gpu : in, out, required, type={ GPUHANDLE }
;       the return value for linear interpolation or the y-values for
;       bilinear interpolation
;    arg4_gpu : in, optional, type={ GPUHANDLE }
;       the return value for bilinear interpolation
;
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuInterpolate, p_gpu, x_gpu, arg3_gpu, arg4_gpu, ERROR=err
  on_error, 2

  case n_params() of
    3: arg3_gpu = gpuInterpolate(arg_present(p_gpu) ? p_gpu : (size(p_gpu, /n_dimensions) gt 0 ? p_gpu[*] : p_gpu[0]), $
                                 arg_present(x_gpu) ? x_gpu : (size(x_gpu, /n_dimensions) gt 0 ? x_gpu[*] : x_gpu[0]), $
                                 LHS=arg3_gpu, ERROR=err)
    4: arg4_gpu = gpuInterpolate(arg_present(p_gpu) ? p_gpu : (size(p_gpu, /n_dimensions) gt 0 ? p_gpu[*] : p_gpu[0]), $
                                 arg_present(x_gpu) ? x_gpu : (size(x_gpu, /n_dimensions) gt 0 ? x_gpu[*] : x_gpu[0]), $
                                 arg_present(arg3_gpu) ? arg3_gpu : (size(arg3_gpu, /n_dimensions) gt 0 ? arg3_gpu[*] : arg3_gpu[0]), $
                                 LHS=arg4_gpu, ERROR=err)
    else: message, level=-1, 'gpuInterpolate: incorrect number of arguments'
  endcase
end


