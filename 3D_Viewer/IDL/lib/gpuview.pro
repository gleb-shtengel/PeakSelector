; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuview.pro
;
; Creates a view on the GPU.
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
;
; Places a view onto a GPU object
;
; :Todo:
;    Available only in hardware or hardware emulation mode, not in pure IDL
;    emulation mode. 
; 
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       array to create a view on
;    px    : in, required, type = {int}
;       array element of origin of view
;    nx    : in, required, type = { int}
;       number of elements in view
;     res_gpu : out, required, type = { GPUHANDLE }
;       gpu variable of view onto x_gpu
;-
pro gpuView, x_gpu, px, nx, res_gpu
 
  res_gpu = x_gpu
  res_gpu.n_elements = nx
  res_gpu.dimensions = [nx, 1]
  if (!gpu.mode eq 0)  then begin
    message, level=-1, 'gpuView: not available in pure IDL emulation'
  end else begin
    res_gpu.handle = res_gpu.handle + px * 4
  end
end 

