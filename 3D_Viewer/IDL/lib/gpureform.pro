; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpureform.pro
;
; Changes the shape of a GPU array.
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
; Change the dimensions of a GPU variable without changing the total number of
; elements.
;
; Warning: this routine modifies the input GPU variable whereas the REFORM
; function does not unless the OVERWRITE keyword is set.
;
; :Params:
;    arr_gpu : in, out, required, type={ GPUHANDLE }
;       GPU variable to manipulate
;    xsize : in, optional, type=long
;       new xsize of arr_gpu
;    ysize : in, optional, type=long
;       new ysize of arr_gpu
;-
pro gpuReform, arr_gpu, xsize, ysize
  on_error, 2

  case n_params() of
    1: begin
         ; remove unneeded leading dimensions just like REFORM
         if (arr_gpu.n_dimensions eq 2L && arr_gpu.dimensions[0] eq 1L) then begin
           arr_gpu.n_dimensions = 1L
           arr_gpu.dimensions = [arr_gpu.dimensions[1], 0L]
         endif

         ; if IDL mode then REFORM the actuall data too
         if (!gpu.mode eq 0) then begin
           *arr_gpu.data = reform(*arr_gpu.data, /overwrite)
         endif
       end
    2: begin
         ; check to make sure the reform keeps the same number of elements
         if (xsize ne arr_gpu.n_elements) then begin
           message, level = -1, 'gpuReform: attempt to change number of elements'
         endif

         ; change n_dimensions to 1
         arr_gpu.n_dimensions = 1L

         ; change dimensions to xsize
         arr_gpu.dimensions = [xsize, 0L]

         ; if IDL mode then REFORM the actuall data too
         if (!gpu.mode eq 0) then begin
           *arr_gpu.data = reform(*arr_gpu.data, xsize, /overwrite)
         endif
       end
    3: begin
         ; check to make sure the reform keeps the same number of elements
         if (xsize * ysize ne arr_gpu.n_elements) then begin
           message, level = -1, 'gpuReform: attempt to change number of elements'
         endif

         arr_gpu.n_dimensions = 2L
         arr_gpu.dimensions = [xsize, ysize]

         ; if IDL mode then REFORM the actuall data too
         if (!gpu.mode eq 0) then begin
           *arr_gpu.data = reform(*arr_gpu.data, xsize, ysize, /overwrite)
         endif
       end
  endcase
end

