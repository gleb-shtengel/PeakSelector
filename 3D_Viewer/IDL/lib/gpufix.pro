; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuFix.pro
;
; Converts a GPU array of one type into an array of another type
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
; Converts a GPU array of one type into a GPU array of another type
;
;  gpuFix, in, out, TYPE = type)
;
; :Returns: 
;    { GPUHANDLE }
; 
; :Params:
;    in : in, required, type=floatarr or { GPUHANDLE }
;
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    TYPE : in, optional, type=int
;           Type of the result vector
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuFix, in, TYPE=type, LHS=lhs, ERROR=err
  on_error, 2
  
  out_type = n_elements(type) eq 0L ? 4L : type
  in_type = size(in, /type) eq 8L ? in.type : size(in, /type)

  if (out_type eq in_type) then return, gpuCopy(in)
  x = size(in, /type) eq 8L ? in : gpuPutArr(in) 
   
  n = long(x.n_elements)

  if (size(lhs, /TYPE) eq 8L) then begin
    if (lhs.type ne out_type) then begin            
      _lhs = gpuMake_Array(x.dimensions, type=out_type, /NOZERO)
      _lhs.isTemporary = 1B
    endif else begin
      _lhs = lhs
    endelse
  endif else begin
    _lhs = gpuMake_Array(x.dimensions, type=out_type, /NOZERO)
    _lhs.isTemporary = 1B
  endelse
  
  if (!gpu.mode eq 0) then begin
    *_lhs.data = fix(*x.data, TYPE=out_type)
  end else begin
    case in_type of
      4 : case out_type of 
            5 : err = gpuFloatToDouble(n, x.handle, _lhs.handle)
            6 : err = gpuFloatToComplexReal(n, x.handle, _lhs.handle)
            9 : err = gpuFloatToDComplexReal(n, x.handle, _lhs.handle)
          endcase
      5 : case out_type of 
            4 : err = gpuDoubleToFloat(n, x.handle, _lhs.handle)
            6 : err = gpuDoubleToComplexReal(n, x.handle, _lhs.handle)
            9 : err = gpuDoubleToDComplexReal(n, x.handle, _lhs.handle)
          end 
      6 : case out_type of
            4 : err = gpuComplexRealToFloat(n, x.handle, _lhs.handle)
            5 : err = gpuComplexRealToDouble(n, x.handle, _lhs.handle)
            9 : begin
                  tmp = gpuMake_Array(n, TYPE=4, /NOZERO)
                  err = gpuComplexRealToFloat(n, x.handle, tmp.handle)
                  err = gpuFloatToDComplexReal(n, tmp.handle, _lhs.handle)
                  gpuFree, tmp
                end
          end
      9 : case out_type of
            4 : err = gpuComplexRealToFloat(n, x.handle, out.handle)
            5 : err = gpuComplexRealToDouble(n, x.handle, _lhs.handle)
            6 : begin
                  tmp = gpuMake_Array(n, TYPE=5, /NOZERO)
                  err = gpuComplexRealToDouble(n, x.handle, tmp.handle)
                  err = gpuDoubleToDComplexReal(n, tmp.handle, _lhs.handle)
                  gpuFree, tmp
                end
          end
      else: message, level=-1, "gpuFix: Unknown input type " + strtrim(in_type, 2)
    endcase
  endelse

  if (size(in, /type) ne 8) then begin
    gpuFree, x
  endif else begin
    if (arg_present(in)) then in.isTemporary = 0B
    if (in.isTemporary) then gpuFree, in       
  endelse
  
  return, _lhs
end


;+
; Converts a GPU array of one type into a GPU array of another type
;
; :Params:
;    in : in, required, type=floatarr or { GPUHANDLE }
;    out : out, required, type={ GPUHANDLE }
;
; :Keywords:
;    TYPE : in, optional, type=int
;           Type of the result vector
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuFix, in, out, TYPE=type, ERROR=err
  on_error, 2
 
  out = gpuFix(arg_present(in) ? in : (size(in, /n_dimensions) gt 0 ? in[*] : in[0]), $
               LHS=out, TYPE=type, ERROR=error)
end


