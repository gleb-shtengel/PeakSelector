; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuGtEq.pro (generated from gpuRelationalOp.pro.sed)
;
; Template for relational operations on GPU. Used to auto-generate binary
; operation procedures using sed.
; Substitution tokens are: GE, GtEq and Compare two vetors for GTEQ
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
; Compare two vetors for GTEQ
;
; There are two forms for the arguments to this routine::
;
;   Result = p1 GE p2
;
; where p1, p2 are { GPUHANDLE } or a numerical array and p3 is { GPUHANDLE }.
; Or::
;
;   Result = p1 * p2 GE p3 * p4 + p5
;
; where p2, p4 are { GPUHANDLE } or a numerical array, and p1, p3, and p5 are 
; scalar values.
;
; :Params:
;    p1 : in, required, type=numtype or { GPUHANDLE }
;    p2 : in, required, type={ GPUHANDLE }
;    p3 : in, out, required, type=numtype or { GPUHANDLE }
;    p4 : in, optional, type={ GPUHANDLE }
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
function gpuGtEq, p1, p2, p3, p4, p5, LHS=lhs, $
                         NONBLOCKING=nonblocking, ERROR=err
  on_error, 2

  err = 0

  case n_params() of
    2: begin
         p1_type = size(p1, /type) ne 8 ? size(p1, /type) : p1.type
         p2_type = size(p2, /type) ne 8 ? size(p2, /type) : p2.type

         result_type = 4

         if (size(p1, /type) ne 8) then begin
           gpuPutArr, fix(p1, type=result_type), x 
         endif else begin
           x = p1
         endelse
         
         if (size(p2, /type) ne 8) then begin
           gpuPutArr, fix(p2, type=result_type), y 
         endif else begin
           y = p2
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
        
         if (x.n_elements ne y.n_elements) then begin
           if (n_elements(x_tmp) ne 0) then begin
              gpuFree, x
              x = x_tmp
           end
           
           if (n_elements(y_tmp) ne 0) then begin
              gpuFree, y
              y = y_tmp
           end
           
           if (size(p1, /type) ne 8) then begin
             gpuFree, x
           endif else begin
             if (arg_present(p1)) then p1.isTemporary = 0B
             if (p1.isTemporary) then gpuFree, p1
           endelse
           
           if (size(p2, /type) ne 8) then begin
             gpuFree, y
           endif else begin
             if (arg_present(p2)) then p2.isTemporary = 0B
             if (p2.isTemporary) then gpuFree, p2           
           endelse 
           
           message, level=-1, 'gpuGtEq: input vector length missmatch'
         endif

         if (!gpu.mode eq 0) then begin
           *_lhs.data = float(*x.data GE *y.data)
         endif else begin
           case result_type of
             4 : err = gpuGtEqF(x.n_elements, x.handle, y.handle, _lhs.handle)
             5 : err = gpuGtEqD(x.n_elements, x.handle, y.handle, _lhs.handle)
             6 : err = gpuGtEqC(x.n_elements, x.handle, y.handle, _lhs.handle)
             9 : err = gpuGtEqZ(x.n_elements, x.handle, y.handle, _lhs.handle)
           endcase
         endelse

         if (n_elements(x_tmp) ne 0) then begin 
           gpuFree, x
           x = x_tmp
         end
         
         if (n_elements(y_tmp) ne 0) then begin
           gpuFree, y
           y = y_tmp
         end
         
         if (size(p1, /type) ne 8) then begin
           gpuFree, x
         endif else begin
           if (arg_present(p1)) then p1.isTemporary = 0B
           if (p1.isTemporary) then gpuFree, p1
         endelse
         
         if (size(p2, /type) ne 8) then begin
           gpuFree, y
         endif else begin
           if (arg_present(p2)) then p2.isTemporary = 0B
           if (p2.isTemporary) then gpuFree, p2           
         endelse  
       end

    5: begin
         result_type = 4

         if (size(p2, /type) ne 8) then begin
           gpuPutArr, fix(p2, type=result_type), x 
         endif else begin
           x = p2
         endelse
         
         if (size(p4, /type) ne 8) then begin
           gpuPutArr, fix(p4, type=result_type), y 
         endif else begin
           y = p4
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
         
         if (x.n_elements ne y.n_elements) then begin
           if (size(x_tmp) ne 0) then begin
             gpuFree, x
             x = x_tmp
           end
           
           if (size(y_tmp) ne 0) then begin
             gpuFree, y
             y = y_tmp
           end
           
           if (size(p2, /type) ne 8) then begin
             gpuFree, x
           endif else begin
             if (arg_present(p2)) then p2.isTemporary = 0B
             if (p2.isTemporary) then gpuFree, p2
           endelse
           
           if (size(p4, /type) ne 8) then begin
             gpuFree, y
           endif else begin
             if (arg_present(p4)) then p4.isTemporary = 0B
             if (p4.isTemporary) then gpuFree, p4      
           endelse

           message, level=-1, 'gpuGtEq: input vector length missmatch'
         endif

         if (!gpu.mode eq 0) then begin
           *_lhs.data = float(p1 * *x.data GE p3 * *y.data + p5)
         endif else begin
           case result_type of
             4 : err = gpuGtEqFAT(x.n_elements, $
                                         float(p1), x.handle, $
                                         float(p3), y.handle, $
                                         float(p5), $
                                         _lhs.handle)
             5 : err = gpuGtEqDAT(x.n_elements, $ 
                                         double(p1), x.handle, $
                                         double(p3), y.handle, $
                                         double(p5), $
                                         _lhs.handle)
             6 : err = gpuGtEqCAT(x.n_elements, $
                                         complex(p1), x.handle, $
                                         complex(p3), y.handle, $
                                         complex(p5), $
                                         _lhs.handle)
             9 : err = gpuGtEqZAT(x.n_elements, $
                                         dcomplex(p1), x.handle, $
                                         dcomplex(p3), y.handle, $
                                         dcomplex(p5), $
                                         _lhs.handle)
           endcase
         endelse
         
         if (n_elements(x_tmp) ne 0) then begin
           gpuFree, x
           x = x_tmp
         end
         
         if (n_elements(y_tmp) ne 0) then begin
           gpuFree, y
           y = y_tmp
         end
         
         if (size(p2, /type) ne 8) then begin
           gpuFree, x
         endif else begin
           if (arg_present(p2)) then p2.isTemporary = 0B
           if (p2.isTemporary) then gpuFree, p2
         endelse
         
         if (size(p4, /type) ne 8) then begin
           gpuFree, y
         endif else begin
           if (arg_present(p4)) then p4.isTemporary = 0B
           if (p4.isTemporary) then gpuFree, p4      
         endelse
       end
    else: message, level=-1, 'gpuGtEq: incorrect number of arguments'
  endcase

  if (~keyword_set(NONBLOCKING) && (!gpu.mode ne 0)) then begin
    err OR= cudaThreadSynchronize()
  endif
          
  return, _lhs
end


;+
; Compare two vetors for GTEQ
;
; There are two forms for the arguments to this routine::
;
;   p3 = p1 GE p2
;
; where p1, p2 are { GPUHANDLE } or a numerical array and p3 is { GPUHANDLE }.
; Or::
;
;   p6 = p1 * p2 GE p3 * p4 + p5
;
; where p2, p4 are { GPUHANDLE } or a numerical array , p6 is { GPUHANDLE }
; and p1, p3, and p5 are scalar values.
;
; :Params:
;    p1 : in, required, type=numtype or { GPUHANDLE }
;    p2 : in, required, type={ GPUHANDLE }
;    p3 : in, out, required, type=numtype or { GPUHANDLE }
;    p4 : in, optional, type={ GPUHANDLE }
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
pro gpuGtEq, p1, p2, p3, p4, p5, p6, NONBLOCKING=nonblocking, ERROR=err
  on_error, 2


  case n_params() of
    3: p3 = gpuGtEq(arg_present(p1) ? p1 : (size(p1, /n_dimensions) gt 0 ? p1[*] : p1[0]), $
                           arg_present(p2) ? p2 : (size(p2, /n_dimensions) gt 0 ? p2[*] : p2[0]), $
                           LHS=p3, NONBLOCKING=nonblocking, ERROR=err)
    6: p6 = gpuGtEq(arg_present(p1) ? p1 : (size(p1, /n_dimensions) gt 0 ? p1[*] : p1[0]), $
                           arg_present(p2) ? p2 : (size(p2, /n_dimensions) gt 0 ? p2[*] : p2[0]), $
                           arg_present(p3) ? p3 : (size(p3, /n_dimensions) gt 0 ? p3[*] : p3[0]), $
                           arg_present(p4) ? p4 : (size(p4, /n_dimensions) gt 0 ? p4[*] : p4[0]), $
                           arg_present(p5) ? p5 : (size(p5, /n_dimensions) gt 0 ? p5[*] : p5[0]), $
                           LHS=p6, NONBLOCKING=nonblocking, ERROR=err)
    else: message, level=-1, 'gpuGtEq: incorrect number of arguments'
  endcase
end


