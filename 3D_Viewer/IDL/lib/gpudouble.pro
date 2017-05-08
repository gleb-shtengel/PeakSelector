; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuDouble.pro
;
; Converts a GPU variable to double
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
; This routine converts an input array into a double GPU array. 
;
; :Returns: 
;    { GPUHANDLE }
;
; :Params:
;    in : in, required, type=numtype or { GPUHANDLE } 
;       array to be converted
;
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuDouble, in, LHS=lhs, ERROR=err
  on_error, 2
  
  return, gpuFix(arg_present(in) ? in : (size(in, /n_dimensions) gt 0 ? in[*] : in[0]), $
                 type=5, LHS=lhs, ERROR=err)
end


;+
; This routine converts an input array into a double GPU array. 
;
; :Params:
;    in : in, required, type=numtype or { GPUHANDLE } 
;       array to be converted
;    out : out, required, type={ GPUHANDLE } 
;       GPU array of the converted input
;
; :Keywords:
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuDouble, in, out, ERROR=err
  on_error, 2
  
  gpuFix, arg_present(in) ? in : (size(in, /n_dimensions) gt 0 ? in[*] : in[0]), $
          out, type=5, ERROR=err
end

