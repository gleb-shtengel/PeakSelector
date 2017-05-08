; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpureal_part.pro
;
; Extracts the real part of complex numbers on the GPU.
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
function gpuReal_part, x_gpu, LHS=lhs, ERROR=err
  on_error, 2
  
  return, gpuReal(x_gpu, LHS=lhs, ERROR=err)
end


;+
; Extracts the real part of a complex data object on the GPU
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
pro gpuReal_part, x_gpu, res_gpu, ERROR=err
  on_error, 2

  gpuReal, x_gpu, res_gpu, ERROR=err
end

