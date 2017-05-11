; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuwhere.pro
;
; Finds array elements with value 1 and creates an array of indices.
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
; Scans the (float-boolean) input vector and stores the indices of
; the true elements in the output vector. The input vector is
; overwritten.
;
; :Returns:
;    { GPUHANDLE }
;    
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       GPU array containing the flags
;    res_gpu : out, required, type={ GPUHANDLE }
;       GPU array icontaining the indices of the true elements
;       in the input vector.
;    count : out, optional, type=long
;       number of true elements
;
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuWhere, x_gpu, count, LHS=lhs, ERROR=err
  compile_opt strictarr
  on_error, 2

  err = 0
  count = [0.]

  x = size(x_gpu, /type) eq 8L ? x_gpu : gpuPutArr(fix(x_gpu, TYPE=4))
           
  if (!gpu.mode eq 0) then begin
    ind =  where(*x.data, count)

    if (size(lhs, /type) eq 8) then begin
      _lhs.data = float(ind)
      _lhs.type = 4
      _lhs.n_dimensions = 1
      _lhs.dimensions = count
    endif else begin
      _lhs = gpuPutarr(float(ind))
      _lhs.isTemporary = 1B
    endelse
  endif else begin
    ; get the last element of the flag vector
    lastElement = x
    lastElement.handle = lastElement.handle +  4 * (lastElement.n_elements - 1)
    lastElement.n_elements = 1
    lastElement.n_dimensions = 1 
    lastElement.dimensions = [1, 0]
    lastX = fltarr(1)
    gpuGetArr, lastElement, lastX
  
    err = gpuPrefixSumF(x.n_elements, x.handle, x.handle)
  
    ; determine the size of the resulting array 
    gpuGetArr, lastElement, count 
   
    if (lastX[0] ne 0) then count = count + 1
  
    if (count eq 0) then begin
      _lhs = gpuGetHandle()
      return, _lhs
    end
  
    ; if result vector is already allocated, free it first
    if (size(lhs, /TYPE) eq 8) then gpuFree, lhs
  
    ; allocate space for the result vector
    _lhs = gpuFltarr(count, /NOZERO)
  
    ; copy elements to result vector
    err = gpuWhereF(x_gpu.n_elements, x_gpu.handle, _lhs.handle, lastX[0])
  endelse
  
  if (size(x_gpu, /TYPE) ne 8) then begin
    gpuFree, x
  endif else begin
    if (arg_present(x_gpu)) then x_gpu.isTemporary = 0B
    if (x_gpu.isTemporary) then gpuFree, x_gpu
  endelse
  
  return, _lhs       
end


;+
; Scans the (float-boolean) input vector and stores the indices of
; the true elements in the output vector. The input vector is
; overwritten.
;
; :Params:
;    x_gpu : in, required, type={ GPUHANDLE }
;       GPU array containing the flags
;    res_gpu : out, required, type={ GPUHANDLE }
;       GPU array icontaining the indices of the true elements
;       in the input vector.
;    count : out, optional, type=long
;       number of true elements
;
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuWhere, x_gpu, res_gpu, count, ERROR=err
 on_error, 2

 count = [0.]

 if (!gpu.mode eq 0) then begin
   r =  where(*x_gpu.data, count)
   res_gpu = gpuFltarr(count, /nozero)
   res_gpu.data = ptr_new(r)
   return
 end 

; get the last element of the flag vector
 lastElement = x_gpu
 lastElement.handle = lastElement.handle +  4 * (lastElement.n_elements - 1)
 lastElement.n_elements = 1
 lastElement.n_dimensions = 1 
 lastElement.dimensions = [1, 0]
 lastX = fltarr(1)
 gpuGetArr, lastElement, lastX

 err = gpuPrefixSumF(x_gpu.n_elements, x_gpu.handle, x_gpu.handle)

; determine the size of the resulting array 
 gpuGetArr, lastElement, count 
 
 if lastX[0] ne 0 then count = count + 1

 if count eq 0 then begin
   res_gpu = gpuGetHandle()
   return
 end

; if result vector is already allocated, free it first
 if size(res_gpu, /type) eq 8 then gpuFree, res_gpu

; allocate space for the result vector
 res_gpu = gpuFltarr(count, /NOZERO)

; copy elements to result vector
 err = gpuWhereF(x_gpu.n_elements, x_gpu.handle, res_gpu.handle, lastX[0])

end

