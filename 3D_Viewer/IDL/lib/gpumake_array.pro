; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuMake_array.pro
;
; creates an array of arbitrary type on the GPU.
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
; This routine creates an IDL structure representing an array on the GPU
; that other FastGPU library routines can use.
;
; :Returns: 
;    structure
;
; :Params:
;    nx : in, required, type=integer
;       size of first dimenion
;    ny : in, optional, type=integer
;       size of second dimenion, if present
;
; :Keywords:
;    NOZERO : in, optional, type=boolean
;       Normally, gpuFltarr setts every element of the allocated array to
;       zero. If set, this keyword prevents zeroing the elements, running
;       slightly faster.
;    VALUE : in, optional, type=float
;       initialization value for array.
;    INDEX : in, optional, type=boolean
;       initialize array elements to their indices
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuMake_Array, nx, ny, $
                        COMPLEX=complex, DCOMPLEX=dcomplex, $
                        FLOAT=float, DOUBLE=double, $
                        TYPE=type,$
                        NOZERO=nozero, VALUE=value, INDEX=index, ERROR=err

  err = 0
  res = gpuGetHandle()   ; get the gpu handle
  
  type_id = 4

  if keyword_set(TYPE) then type_id = type 
  if keyword_set(FLOAT) then type_id =  4 
  if keyword_set(DOUBLE) then type_id = 5 
  if keyword_set(COMPLEX) then type_id = 6 
  if keyword_set(DCOMPLEX) then type_id = 9 

  case type_id of
    4:  begin
	        type_val = float(1.)
	        nbytes = 4L
        end
    5:  begin
          type_val = double(1.)
          nbytes = 8L
        end
    6:  begin
          type_val = complex(1.)
          nbytes  = 2 * 4L
        end 
    9:  begin
          type_val = dcomplex(1.)
          nbytes = 2 * 8L
        end
  endcase

  res.type = type_id
 
  ndims = n_params()

  if (ndims eq 1) then begin
    ny = 1
    if (n_elements(nx) gt 1) then begin
      ny = nx[1] 
      nx = nx[0]
      ndims = 2
    end
  end

  if (ny le 1) then ndims = 1

  case ndims of
    1 :  begin
           ln = long(nx)
           res.n_elements    = long(nx)
           res.n_dimensions  = 1L
           res.dimensions[0] = long(nx)
           res.dimensions[1] = 1L
         end
    2 :  begin
           ln = long(nx) * long(ny)
           res.n_elements    = long(nx) * long(ny)
           res.n_dimensions  = 2L
           res.dimensions[0] = long(nx)
           res.dimensions[1] = long(ny)
         end
  endcase

  if (!gpu.mode eq 0) then begin
    if (ndims eq 1) then begin
      res.data = ptr_new(make_array(nx, $
                                    COMPLEX=complex, DCOMPLEX=dcomplex, $
                                    FLOAT=float, DOUBLE=double,$
                                    NOZERO=nozero, VALUE=value, INDEX=index))
    endif else begin
      res.data = ptr_new(make_array(nx, ny, $
                                    COMPLEX=complex, DCOMPLEX=dcomplex, $
                                    FLOAT=float, DOUBLE=double,$
                                    NOZERO=nozero, VALUE=value, INDEX=index))
    endelse
    return, res
  endif

  n  = ln * nbytes
  err = cudaMalloc(handle, n)

  if (~keyword_set(NOZERO)) then begin
     if (~keyword_set(value)) then value = type_val * 0.
     if (value eq 0.) then begin
        err = cudaMemset(handle, 0L, n)
     endif else begin
        err = cudaMemset(handle, 0L, n)

        case res.type of
           4 : err = gpuAddFAT(ln, float(0), handle, float(0), handle, $
                               float(value), handle)
           5 : err = gpuAddDAT(ln, double(0), handle, double(0), handle, $
                               double(value), handle)
           6 : err = gpuAddCAT(ln, complex(0), handle, complex(0), handle, $
                               complex(value), handle)
           9 : err = gpuAddZAT(ln, dcomplex(0), handle, dcomplex(0), $
                               handle, dcomplex(value), handle)
           else: 
        endcase
     endelse
  endif

  if (keyword_set(INDEX)) then begin 
     case res.type of 
           4 : err = gpuFindgenF(ln, handle)
           5 : err = gpuDindgenD(ln, handle)
           6 : err = gpuCindgenC(ln, handle)
           9 : err = gpuDCindgenZ(ln, handle)
      endcase 
  endif
  res.handle = handle
  return, res
end

