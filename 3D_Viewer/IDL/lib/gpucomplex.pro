; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuComplex.pro
;
; Converts a GPU variable to complex
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
; This routine converts an input array into a complex GPU array. 
;
; There are two forms for the arguments to this routine::
;
;   result = COMPLEX(p1)
;
; where p1 is a { GPUHANDLE } or a numeric array. Or::
;
;   result = COMPLEX(p1, p2)
;
; where p1, p2 are { GPUHANDLE } or a numeric array and real and imaginary 
; parts respectively. 
;
; :Todo:
;    Performance of this routine could be improved by a new kernel to create
;    complex arrays instead of creating temporary arrays.
;    
; :Returns: 
;    { GPUHANDLE }
;
; :Params:
;    p1 : in, required, type=numtype or { GPUHANDLE } 
;       real part of array to be converted
;    p2 : in, optional, type=numtype or { GPUHANDLE } 
;       imaginary part of array to be converted
;
; :Keywords:
;    DOUBLE : in, optional, type=boolean
;       set to get double complex values
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuComplex, p1, p2, DOUBLE=double, LHS=lhs, ERROR=err
  on_error, 2

  if (keyword_set(DOUBLE)) then return, gpuDcomplex(p1, p2, ERROR=err)
  
  case n_params() of
    1: return, gpuFix(p1, type=6, ERROR=err)
    2: begin
         if (size(p1, /type) ne 8) then begin
           re_gpu = gpuPutArr(p1)
         endif else begin
           re_gpu = p1
         endelse
             
         if (size(lhs, /TYPE) eq 8) then begin           
           gpuFix, re_gpu, _lhs, TYPE=6
         endif else begin
           _lhs = gpuFix(re_gpu, type=6, ERROR=err)
           _lhs.isTemporary = 1B         
         endelse
                 
         if (size(p2, /type) ne 8) then begin
           im_gpu = gpuPutArr(p2)
         endif else begin
           im_gpu = p2
         endelse

         if (!gpu.mode eq 0) then begin
           *_lhs.data = complex(*re_gpu.data, *im_gpu.data)
         endif else begin 
           
           case im_gpu.type of           
             4 : begin
                   tmp = gpuComplexarr(im_gpu.n_elements, /NOZERO)
                   err = gpuFloatToComplexImag(im_gpu.n_elements, $
                                               im_gpu.handle, tmp.handle)
                   _lhs = gpuAdd(_lhs, tmp, LHS=_lhs)                           
                   gpuFree, tmp
                 end
             5 : begin
                   tmp = gpuDcomplexarr(im_gpu.n_elements, /NOZERO)
                   err = gpuDoubleToComplexImag(im_gpu.n_elements, $
                                                im_gpu.handle, tmp.handle)
                   _lhs = gpuAdd(_lhs, tmp, LHS=_lhs)
                   gpuFree, tmp
                 end
             6 : begin 
                   tmp = gpuFltarr(im_gpu.n_elements, /NOZERO)
                   err = gpuComplexRealToFloat(im_gpu.n_elements, $
                                               im_gpu.handle, tmp.handle)
                   err = gpuFloatToComplexImag(tmp.n_elements, $
                                               tmp.handle, _lhs.handle)
                   gpuFree, tmp
                 end 
             9 : begin
                   tmp = gpuFltarr(im_gpu.n_elements, /NOZERO)
                   err = gpuDComplexRealToFloat(im_gpu.n_elements, $
                                                im_gpu.handle, tmp.handle)
                   err = gpuFloatToComplexImag(tmp.n_elements, $
                                               tmp.handle, _lhs.handle)
                   gpuFree, tmp
                 end
           endcase
         endelse

        if (size(p1, /type) ne 8) then begin
          gpuFree, re_gpu
        endif else begin
          if (arg_present(p1)) then p1.isTemporary = 0B
          if (p1.isTemporary) then gpuFree, p1
        endelse
                
        if (size(p2, /type) ne 8) then begin
          gpuFree, im_gpu
        endif else begin
          if (arg_present(p2)) then p2.isTemporary = 0B
          if (p2.isTemporary) then gpuFree, p2              
        endelse
      end
  endcase
       
  return, _lhs
end


;+
; This routine converts an input array into a complex GPU array. 
;
; :Params:
;    p1 : in, required, type=numtype or { GPUHANDLE } 
;       array to be converted
;    p2 : in, out, required, type=numtype or { GPUHANDLE } 
;       either imaginary part of array to be converted or output GPU array
;    p3 : out, required, type={ GPUHANDLE } 
;       GPU array of the converted input
;
; :Keywords:
;    DOUBLE : in, optional, type=boolean
;       set to get double complex values
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuComplex, p1, p2, p3, DOUBLE=double, ERROR=err
  on_error,2

  case n_params() of
    2: p2 = gpuComplex(arg_present(p1) ? p1 : (size(p1, /n_dimensions) gt 0 ? p1[*] : p1[0]), $
                       LHS=p2, DOUBLE=double, ERROR=err)
    3: p3 = gpuComplex(arg_present(p1) ? p1 : (size(p1, /n_dimensions) gt 0 ? p1[*] : p1[0]), $
                       arg_present(p2) ? p2 : (size(p2, /n_dimensions) gt 0 ? p2[*] : p2[0]), $
                       LHS=p3, DOUBLE=double, ERROR=err)
    else: message, level=-1, 'gpuComplex: incorrect number of arguments'
  endcase
end

