; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpudblarr.pro
;
; transfers an array from host memory to the GPU
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
; This routine creates an IDL structure representing a double array on the GPU
; that other FastGPU library routines can use.
;
; :Returns: structure
;
; :Params:
;    nx : in, required, type=integer
;       size of first dimenion
;    ny : in, optional, type=integer
;       size of second dimenion, if present
;
; :Keywords:
;    NOZERO: in, optional, type=boolean
;       Normally, gpuDblarr setts every element of the allocated array to
;       zero. If set, this keyword prevents zeroing the elements, running
;       slightly faster.
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuDblarr, nx, ny, NOZERO=nozero, ERROR=err

  case n_params() of
    1 : begin
          res = gpuMake_Array(nx, /DOUBLE, NOZERO=nozero, ERROR=err)
        end
    2 : begin
          res = gpuMake_Array(nx, ny, /DOUBLE, NOZERO=nozero, ERROR=err)
        end
  endcase

  return, res
end

