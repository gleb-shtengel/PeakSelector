; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuMatrix_multiply
;
; Multiplies two matrices on the GPU
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
; Performs multiplication of two gpu matrices or their
; transposes.
;
; :Returns:
;    { GPUHANDLE }
;
; :Params:
;    a_gpu : in, required, type={ GPUHANDLE }
;       GPU array containing the matrix A
;    b_gpu : in, required, type={ GPUHANDLE }
;       GPU array containing the matrix B
;
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    ATRANSPOSE : in, optional, type=bool
;       use A transpose of A for product
;    BTRANSPOSE : in, optional, type=bool
;       use B transpose of B for product
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuMatrix_Multiply, a_gpu, b_gpu, LHS=lhs, $
                             ATRANSPOSE=atranspose, BTRANSPOSE=btranspose, $
                             ERROR=err
  compile_opt strictarr
  on_error, 2
  
  a = size(a_gpu, /type) eq 8L ? a_gpu : gpuPutArr(a_gpu)
  b = size(b_gpu, /type) eq 8L ? b_gpu : gpuPutArr(b_gpu)
  
  if (a.dimensions[1] eq 0) then a.dimensions[1] = 1
  if (b.dimensions[1] eq 0) then b.dimensions[1] = 1

  cx = keyword_set(ATRANSPOSE) ? a.dimensions[1] : a.dimensions[0]
  cy = keyword_set(BTRANSPOSE) ? b.dimensions[0] : b.dimensions[1]

  if (size(lhs, /type) eq 8) then begin
    if (lhs.type ne result_type) then begin            
      gpuFix, lhs, _lhs, TYPE=result_type
    endif else begin
      _lhs = lhs
    endelse
  endif else begin
    _lhs = gpuFltarr(cx, cy, /NOZERO)
    _lhs.isTemporary = 1B
  endelse
  
  m = keyword_set(ATRANSPOSE) ? a.dimensions[1] : a.dimensions[0]
  n = keyword_set(BTRANSPOSE) ? b.dimensions[0] : b.dimensions[1]
  k = keyword_set(ATRANSPOSE) ? a.dimensions[0] : a.dimensions[1]

  if ((keyword_set(BTRANSPOSE) and k ne b.dimensions[1]) or $
     (~keyword_set(BTRANSPOSE) and k ne b.dimensions[0])) then begin
    if (size(a_gpu, /type) ne 8) then begin
      gpuFree, a
    endif else begin
      if (arg_present(a_gpu)) then a_gpu.isTemporary = 0B
      if (a_gpu.isTemporary) then gpuFree, a_gpu       
    endelse
    
    if (size(b_gpu, /type) ne 8) then begin
      gpuFree, b
    endif else begin
      if (arg_present(b_gpu)) then b_gpu.isTemporary = 0B
      if (b_gpu.isTemporary) then gpuFree, b_gpu             
    endelse
    
    message, level=-1, 'gpuMatrix_Multiply: Matrix size mismatch' 
  endif
   
  if (!gpu.mode eq 0) then begin 
    *_lhs.data = matrix_multiply(*a.data, *b.data, $
                                 ATRANSPOSE=atranspose, $
                                 BTRANSPOSE=btranspose)
  endif else begin
    at = keyword_set(ATRANSPOSE) ? 'T' : 'N'
    bt = keyword_set(BTRANSPOSE) ? 'T' : 'N'
    
    ; TODO: do double precision cublasDgemm
    err =  cublasSgemm(at, bt, long(m), long(n), long(k), $
                       float(1), a.handle, long(a.dimensions[0]), $
                       b.handle, long(b.dimensions[0]), $
                       float(0), _lhs.handle, long(_lhs.dimensions[0])) 
  endelse

  if (size(a_gpu, /type) ne 8) then begin
    gpuFree, a
  endif else begin
    if (arg_present(a_gpu)) then a_gpu.isTemporary = 0B
    if (a_gpu.isTemporary) then gpuFree, a_gpu       
  endelse
  
  if (size(b_gpu, /type) ne 8) then begin
    gpuFree, b
  endif else begin
    if (arg_present(b_gpu)) then b_gpu.isTemporary = 0B
    if (b_gpu.isTemporary) then gpuFree, b_gpu             
  endelse
  
  return, _lhs
end


;+
; Performs multiplication of two gpu matrices or their
; transposes.
;
; :Params:
;    a_gpu : in, required, type={ GPUHANDLE }
;       GPU array containing the matrix A
;    b_gpu : in, required, type={ GPUHANDLE }
;       GPU array containing the matrix B
;    res_gpu : out, required, type={ GPUHANDLE }
;       GPU array containing the product
;
; :Keywords:
;    ATRANSPOSE : in, optional, type=bool
;       use A transpose of A for product
;    BTRANSPOSE : in, optional, type=bool
;       use B transpose of B for product
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuMatrix_Multiply, a_gpu, b_gpu, res_gpu, $
                        ATRANSPOSE=atranspose, BTRANSPOSE=btranspose, ERROR=err
  compile_opt strictarr
  on_error, 2
  
  res_gpu = gpuMatrixMultiply(arg_present(a_gpu) ? a_gpu : (size(a_gpu, /n_dimensions) gt 0 ? a_gpu[*] : a_gpu[0]), $
                              arg_present(b_gpu) ? b_gpu : (size(b_gpu, /n_dimensions) gt 0 ? b_gpu[*] : b_gpu[0]), $
                              LHS=lhs, $
                              ATRANSPOSE=atranspose, BTRANSPOSE=btranspose, $ 
                              ERROR=err)
end

