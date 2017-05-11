; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuRound (generated from gpuUnaryOp.pro.sed)
;
; Template for binary operations on GPU. Used to auto-generate binary
; operation procedures using sed.
; Substitution tokens are: round, BINARY_OP_NAME and BINARY_OP_TEXT
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
; Round a vector
;
; There are two forms for the arguments to this routine::
;
;   Result = round(p1)
;
; where p1 is { GPUHANDLE } or a numeric array and p2 is { GPUHANDLE }. Or::
;
;   Result = p1 * round(p2 * p3 + p4) + p5
;
; where p2 is { GPUHANDLE } or a numerica array and p6 is { GPUHANDLE } and
; p1, p2, p4, and p5 are scalar values.
;
; :Returns: 
;    { GPUHANDLE }
; 
; :Params:
;    p1 : in, required, type=numtype or { GPUHANDLE }
;    p2 : in, out, required, type=numtype or { GPUHANDLE }
;    p3 : in, optional, type={ GPUHANDLE }
;    p4 : in, optional, type=numtype
;    p5 : in, optional, type=numtype
;
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    NONBLOCKING : in, optional, type=boolean
;       unless this keyword is set, this will block until the device has
;       completed all preceding requested tasks
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuRound, p1, p2, p3, p4, p5, LHS=lhs, $
                           NONBLOCKING=nonblocking, ERROR=err
  on_error, 2

  err = 0

  case n_params() of
    1: begin
         p1_type = size(p1, /type) ne 8 ? size(p1, /type) : p1.type

         result_type = p1_type

         if (size(p1, /type) ne 8) then begin
           gpuPutArr, p1, x 
         endif else begin
           x = p1
         endelse

         if (size(lhs, /TYPE) eq 8L) then begin
           if (lhs.type ne result_type) then begin            
             gpuFix, lhs, _lhs, TYPE=result_type
           endif else begin
             _lhs = lhs
           endelse
         endif else begin
           _lhs = gpuMake_Array(x.dimensions, TYPE=result_type, /NOZERO)  
           _lhs.isTemporary = 1B
         endelse
         
         if (!gpu.mode eq 0) then begin
           *_lhs.data = round(*x.data)
         endif else begin
           case result_type of
             4 : err = gpuRoundF(x.n_elements, x.handle, _lhs.handle)
             5 : err = gpuRoundD(x.n_elements, x.handle, _lhs.handle)
             6 : err = gpuRoundC(x.n_elements, x.handle, _lhs.handle)
             9 : err = gpuRoundZ(x.n_elements, x.handle, _lhs.handle)
           endcase
         endelse
         
         if (size(p1, /type) ne 8) then begin
           gpuFree, x
         endif else begin
           if (arg_present(p1)) then p1.isTemporary = 0B
           if (p1.isTemporary) then gpuFree, p1      
         endelse
       end
    5: begin

         p3_type = size(p3, /type) ne 8 ? size(p3, /type) : p3.type

         result_type = p3_type

         if (size(p3, /type) ne 8) then begin
           gpuPutArr, p3, x 
         endif else begin
           x = p3
         endelse

         if (size(lhs, /TYPE) eq 8L) then begin
           if (lhs.type ne result_type) then begin            
             gpuFix, lhs, _lhs, TYPE=result_type
           endif else begin
             _lhs = lhs
           endelse
         endif else begin
           _lhs = gpuMake_Array(x.dimensions, TYPE=result_type, /NOZERO)  
           _lhs.isTemporary = 1B
         endelse
         
         if (!gpu.mode eq 0) then begin
           *_lhs.data = p1 * round(p2 * *x.data + p4) + p5
         endif else begin
           case result_type of
             4 : err = gpuRoundFAT(x.n_elements, $
                                           float(p1), $
                                           float(p2), x.handle, float(p4), $
                                           float(p5), $
                                           _lhs.handle)
             5 : err = gpuRoundDAT(x.n_elements, $
                                           double(p1), $
                                           double(p2), x.handle, double(p4), $
                                           double(p5), $
                                           _lhs.handle)
             6 : err = gpuRoundCAT(x.n_elements, $
                                           complex(p1), $
                                           complex(p2), x.handle, complex(p4), $
                                           complex(p5), $
                                           _lhs.handle)
             9 : err = gpuRoundZAT(x.n_elements, $
                                           dcomplex(p1), $
                                           dcomplex(p2), x.handle, dcomplex(p4), $
                                           dcomplex(p5), $
                                           _lhs.handle)
           endcase
         endelse
         
         if (size(p3, /type) ne 8) then begin
           gpuFree, x
         endif else begin
           if (arg_present(p3)) then p3.isTemporary = 0B
           if (p3.isTemporary) then gpuFree, p3
         endelse
       end
    else: message, level=-1, 'gpuRound: incorrect number of arguments'
  endcase

  if (~keyword_set(NONBLOCKING) && (!gpu.mode ne 0)) then begin
    err OR= cudaThreadSynchronize()
  endif
          
  return, _lhs
end


;+
; Round a vector
;
; There are two forms for the arguments to this routine::
;
;   p2 = round(p1)
;
; where p1 is { GPUHANDLE } or a numeric array and p2 is { GPUHANDLE }. Or::
;
;   p6 = p1 * round(p2 * p3 + p4) + p5
;
; where p2 is { GPUHANDLE } or a numerica array and p6 is { GPUHANDLE } and
; p1, p2, p4, and p5 are scalar values.
;
; :Params:
;    p1 : in, required, type=numtype or { GPUHANDLE }
;    p2 : in, out, required, type=numtype or { GPUHANDLE }
;    p3 : in, optional, type={ GPUHANDLE }
;    p4 : in, optional, type=numtype
;    p5 : in, optional, type=numtype
;    p6 : out, optional, type={ GPUHANDLE }
;
; :Keywords:
;    NONBLOCKING : in, optional, type=boolean
;       unless this keyword is set, this will block until the device has
;       completed all preceding requested tasks
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuRound, p1, p2, p3, p4, p5, p6, NONBLOCKING=nonblocking, ERROR=err
  on_error, 2

  case n_params() of
    2: p2 = gpuRound(arg_present(p1) ? p1 : (size(p1, /n_dimensions) gt 0 ? p1[*] : p1[0]), $
                             LHS=p2, NONBLOCKING=nonblocking, ERROR=err)
    6: p6 = gpuRound(arg_present(p1) ? p1 : (size(p1, /n_dimensions) gt 0 ? p1[*] : p1[0]), $
                             arg_present(p2) ? p2 : (size(p2, /n_dimensions) gt 0 ? p2[*] : p2[0]), $
                             arg_present(p3) ? p3 : (size(p3, /n_dimensions) gt 0 ? p3[*] : p3[0]), $
                             arg_present(p4) ? p4 : (size(p4, /n_dimensions) gt 0 ? p4[*] : p4[0]), $
                             arg_present(p5) ? p5 : (size(p5, /n_dimensions) gt 0 ? p5[*] : p5[0]), $
                             LHS=p6, NONBLOCKING=nonblocking, ERROR=err)
    else: message, level=-1, 'gpuRound: incorrect number of arguments'
  endcase
end


