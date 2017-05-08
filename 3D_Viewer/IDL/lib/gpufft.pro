; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuFFT.pro
;
; Performs FFT on GPU
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
; Perform FFT on the GPU
;
; The FFT can operate either on real or complex signals, and perform
; both 1D and 2D transforms. Complex number arrays use an interleaved
; memory layout and are otherwise treated as regular real arrays. Use
; the gpuReal and gpuImaginary routines to extract the real and imaginary
; component.
;
; :Todo:
;   BATCH does not work in pure IDL emulation
;   
; :Returns:
;    { GPUHANDLE }
;
; :Params:
;    p1 : in, required, type={ GPUHANDLE }
;       GPU array to perform FFT on
;    direction: in, optional, type=integer
;       -1 to perform FFT, 1 to peform inverse FFT
;       
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    PLAN: in, out, optional, type=long
;       The plan to be used for this transform. If undefined upon
;       entry, a new plan will be created and returned in this variable.
;    INVERSE: in, optional, type=boolean
;       Perform an inverse complex to complex fft. Only valid in combination
;       with the COMPLEX2COMPLEX keyword.
;    DIM2D: in, optional, type=boolean
;       Perform a 2D fft on the input data set. The dimensionality is
;       extracted from the input object.
;    DENORMALIZED: in, optional, type=boolean
;       Perform a denormalized transform. By default, the result is
;       normalized by the number of input vector elements. This option
;       allows to skip this normalization step, yielding slightly faster
;       exectution.
;    DESTROYPLAN: in, optional, type=boolen
;       Frees the GPU resoures occupied by the fft plan. This invalidates
;       the plan and requires a new plan to be generated.
;    ERROR: out, optional, type=int
;       Error code.
;-
function gpufft, p1, direction, LHS=lhs, PLAN=plan, $
                 INVERSE=inverse, DIM2D=DIM2D, DENORMALIZED=denormalized, $
                 DESTROYPLAN=destroyPlan, BATCH=batch, ERROR=err

  on_error, 2

  ; the following types are from the cufft documentation
  ; complex to complex: 0x29 = 41

  ; if user doesn't want the plan and doesn't provide one,
  ; make sure the plan gets destroyed after use.
  destroy = 0
  if (~arg_present(plan)) then destroy = 1
  if (keyword_set(destroyPlan)) then destroy = 1

  fftType = 41L

  ; by default *FORWARD* FFT
  if (n_elements(direction) eq 0) then direction = -1L

  if (keyword_set(inverse)) then direction = 1L

  x_gpu = size(p1, /type) eq 8L ? p1 : gpuPutArr(p1)

  ; must take FFT of complex variable
  if (x_gpu.type ne 6) then begin 
    gpuFix, x_gpu, x_gpu_fix , TYPE=6
    gpuFree, x_gpu
    x_gpu = x_gpu_fix
  endif

  if (size(lhs, /TYPE) eq 8L) then begin
    if (lhs.type ne result_type) then begin            
      gpuFix, lhs, _lhs, TYPE=6
    endif else begin
      _lhs = lhs
    endelse
  endif else begin
    case x_gpu.n_dimensions of 
      1: _lhs = gpuComplexArr(x_gpu.n_elements)
      2: _lhs = gpuComplexArr(x_gpu.dimensions[0], x_gpu.dimensions[1])
    endcase
    _lhs.isTemporary = 1B
  endelse
  
  if (!gpu.mode eq 0) then begin
    *_lhs.data = fft(*x_gpu.data, direction)
    if (keyword_set(denormalized) and (direction ne 1)) then begin
      *_lhs.data *= x_gpu.n_elements
    endif
  endif else begin  
    if size(plan, /type) eq 0 then begin
      plan = 0LL
  
      if (keyword_set(DIM2D)) then begin
        ; to use column major, we need to flip the dimensions
        err = cufftPlan2d(plan, x_gpu.dimensions[1], x_gpu.dimensions[0], $
                          fftType)
      endif else begin
        if (not keyword_set(BATCH)) then batch = 1L
        err = cufftPlan1d(plan, x_gpu.n_elements, fftType, long(batch))
      endelse
    endif

    err = cufftExecC2C(plan, x_gpu.handle, _lhs.handle, direction)

    if (destroy eq 1) then begin
      err = cufftDestroy(plan)
      plan = 0LL
    endif    

    if (~keyword_set(denormalized) and (direction ne 1)) then begin
      gpuAdd, 1.0 / float(x_gpu.n_elements), _lhs, 0.0, _lhs, 0.0, _lhs
    endif
  endelse
    
  if (size(p1, /type) ne 8) then begin
    gpuFree, x_gpu
  endif else begin
    if (arg_present(p1)) then p1.isTemporary = 0B
    if (p1.isTemporary) then gpuFree, p1    
  endelse
  
  return, _lhs
end


;+
; Perform FFT on the GPU
;
; The FFT can operate either on real or complex signals, and perform
; both 1D and 2D transforms. Complex number arrays use an interleaved
; memory layout and are otherwise treated as regular real arrays. Use
; the gpuReal and gpuImaginary routines to extract the real and imaginary
; component.
;
; :Params:
;    p1 : in, required, type={ GPUHANDLE }
;       GPU array to perform FFT on
;    res_gpu: out, required, type={ GPUHANDLE }
;       GPU array containing the result
;    direction: in, optional, type=integer
;       -1 to perform FFT, 1 to peform inverse FFT
;       
; :Keywords:
;    PLAN: in, out, optional, type=long
;       The plan to be used for this transform. If undefined upon
;       entry, a new plan will be created and returned in this variable.
;    INVERSE: in, optional, type=boolean
;       Perform an inverse complex to complex fft. Only valid in combination
;       with the COMPLEX2COMPLEX keyword.
;    DIM2D: in, optional, type=boolean
;       Perform a 2D fft on the input data set. The dimensionality is
;       extracted from the input object.
;    DENORMALIZED: in, optional, type=boolean
;       Perform a denormalized transform. By default, the result is
;       normalized by the number of input vector elements. This option
;       allows to skip this normalization step, yielding slightly faster
;       exectution.
;    DESTROYPLAN: in, optional, type=boolen
;       Frees the GPU resoures occupied by the fft plan. This invalidates
;       the plan and requires a new plan to be generated.
;    ERROR: out, optional, type=int
;       Error code.
;
;-
pro gpufft, p1, res_gpu, direction, PLAN=plan, $
            INVERSE=inverse, DIM2D=DIM2D, DENORMALIZED=denormalized, $
            DESTROYPLAN=destroyplan, BATCH=batch, ERROR=err

  on_error, 2

  res_gpu = gpufft(arg_present(p1) ? p1 : (size(p1, /n_dimensions) gt 0 ? p1[*] : p1[0]), $
                   direction, LHS=res_gpu, PLAN=plan, $
                   INVERSE=inverse, DIM2D=DIM2D, DENORMALIZED=denormalized, $
                   DESTROYPLAN=destroyplan, BATCH=batch, ERROR=err)
end
