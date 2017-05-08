; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuDcomplex.pro
;
; Converts a GPU variable to double complex
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
;   result = DCOMPLEX(p1)
;
; where p1 is a { GPUHANDLE } or a numeric array. Or::
;
;   result = DCOMPLEX(p1, p2)
;
; where p1, p2 are { GPUHANDLE } or a numeric array and real and imaginary 
; parts respectively. 
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
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuDcomplex, p1, p2, LHS=lhs, ERROR=err
  on_error, 2

  case n_params() of
    1: return, gpuFix(p1, type=9, ERROR=err)
    2: begin
         if (size(p1, /type) ne 8) then begin
           re_gpu = gpuPutArr(p1)
         endif else begin
           re_gpu = p1
         endelse
             
         if (size(lhs, /TYPE) eq 8) then begin           
           gpuFix, re_gpu, _lhs, TYPE=9
         endif else begin
           _lhs = gpuFix(re_gpu, type=9, ERROR=err)
           _lhs.isTemporary = 1B         
         endelse
        
        if (size(p2, /type) ne 8) then begin
          im_gpu = gpuPutArr(p2)
        endif else begin
          im_gpu = p2
        endelse

        if (!gpu.mode eq 0) then begin
          *_lhs.data = complex(*re_gpu.data, *im_gpu.data)
        end else begin 
          case im_gpu.type of
            4 : err = gpuFloatToDComplexImag(im_gpu.n_elements, $
                                             im_gpu.handle, _lhs.handle)
            5 : err = gpuDoubleToDComplexImag(im_gpu.n_elements, $
                                              im_gpu.handle, _lhs.handle)
            6 : begin 
                  tmp = gpuFltarr(im_gpu.n_elements, /NOZERO)
                  err = gpuComplexRealToDouble(im_gpu.n_elements, $
                                               im_gpu.handle, tmp.handle)
                  err = gpuFloatToDComplexImag(im_gpu.n_elements, $
                                               im_gpu.handle, _lhs.handle)
                  gpuFree, tmp
                end 
            9 : begin
                  tmp = gpuFltarr(im_gpu.n_elements, /NOZERO)
                  err = gpuDComplexRealToDouble(im_gpu.n_elements, $
                                                im_gpu.handle, tmp.handle)
                  err = gpuFloatToDComplexImag(im_gpu.n_elements, $
                                               im_gpu.handle, _lhs.handle)
                  gpuFree, tmp
                end
          endcase
        endelse 

        if (size(p1, /type) eq 8) then begin
          if (arg_present(p1)) then p1.isTemporary = 0B
          if (p1.isTemporary) then gpuFree, p1
        endif
                
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
; This routine converts an input array into a double complex GPU array. 
;
; :Returns: 
;    structure
;
; :Params:
;    p1 : in, required, type=numtype or { GPUHANDLE } 
;       array of the real parts to be converted
;    p2 : in, out, required, type={ GPUHANDLE } 
;       array of the imaginary parts to be converted or the GPU array of the 
;       converted input
;    p3 : out, optional, type={ GPUHANDLE } 
;       GPU array of the converted input
;       
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuDcomplex, p1, p2, p3, ERROR=err
  on_error, 2

  case n_params() of
    2: p2 = gpuDcomplex(arg_present(p1) ? p1 : (size(p1, /n_dimensions) gt 0 ? p1[*] : p1[0]), $
                        LHS=p2, DOUBLE=double, ERROR=err)
    3: p3 = gpuDcomplex(arg_present(p1) ? p1 : (size(p1, /n_dimensions) gt 0 ? p1[*] : p1[0]), $
                        arg_present(p2) ? p2 : (size(p2, /n_dimensions) gt 0 ? p2[*] : p2[0]), $
                        LHS=p3, DOUBLE=double, ERROR=err)
    else: message, level=-1, 'gpuDcomplex: incorrect number of arguments'
  endcase
end

