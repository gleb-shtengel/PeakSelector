; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuFree.pro
;
; Releases memory allocated on the GPU.
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
; Free resource on GPU held by given variable(s).
;
; :Params:
;    x_gpu : in, required, type=scalar or array of { GPUHANDLE }
;       GPU variable(s) to free
;
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuFree, x_gpu, ERROR=err

  err = 0

  nVars = n_elements(x_gpu)
  for i = 0L, nVars - 1L do begin
    if (!gpu.mode eq 0) then begin
       ptr_free, x_gpu[i].data
     endif else begin
       err = cudaFree(x_gpu[i].handle)
    endelse
  end

  ; undefine the variable x_gpu
  dummy = temporary(x_gpu)
end


