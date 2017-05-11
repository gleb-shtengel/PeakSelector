; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpumax.pro
;
; compute maximum of a vector
;
; Copyright (C) 2007 Tech-X Corporation.  All rights reserved.
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
;-----------------------------------------------------------------------------

;+
; Calculates the maximum of the elements of the GPU vector.
;
; :Returns: 
;    maximum value of x_gpu
;
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       GPU array to search for maximum
;    max_subscript : in, optional, type = long
;       position of the first ocurrence of the maximum element
;
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuMax, x_gpu, max_subscript, ERROR=err

  err = 0
  res = 0.0

  if (!gpu.mode eq 0) then begin
    res = max(*x_gpu.data, max_subscript)
  endif else begin
    max_subscript = 0L
    err = gpuMaxF(x_gpu.n_elements, x_gpu.handle, res, max_subscript)
  endelse

  return, res
end

