; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpurandomu.pro
;
; Creates uniformly distributed random numbers on the GPU.
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
; This routine generates uniformly distributed random numbers on the GPU
;
; :Returns:
;    { GPUHANDLE }
;
; :Params:
;    seed: in, required, type=int
;       seed for RNG. Can be undefined
;    nx : in, required, type=int
;       x-dimension of resulting random array
;    ny : out, optional, type=int
;       y-dimension of resulting random array
;
; :Keywords:
;    LHS : in, optional, type={ GPUHANDLE }
;       pass GPU variable to use as return value
;    NORMAL: in, optional, type=bool
;       generate a normal distribution, rather than a uniform one
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuRandomu, seed, nx, ny, LHS=lhs, ERROR=err, NORMAL=normal
  compile_opt strictarr
   
  if (n_elements(seed) ne 0 && !gpu.mode ne 0) then err = gpuSeedMTF(long(seed))

  case n_params() of
    2 : begin
          if (size(lhs, /type) eq 8) then begin
            if (lhs.type ne 4) then begin            
              gpuFix, lhs, _lhs, TYPE=4
            endif else begin
              _lhs = lhs
            endelse
          endif else begin
            _lhs = gpuFltarr(nx, /NOZERO)
            _lhs.isTemporary = 1B
          endelse
          
          if (!gpu.mode eq 0) then begin
            *_lhs.data = randomu(seed, nx, NORMAL=keyword_set(normal))
          endif else begin
            if (n_elements(seed) ne 0) then err = gpuSeedMTF(long(seed))
            err = gpuMTF(long(nx), _lhs.handle)
            if (keyword_set(NORMAL)) then begin
              err = gpuBoxMullerF(long(nx), _lhs.handle)
            endif
          endelse
        end
    3 : begin
          nxy = long(nx) * long(ny)
          
          if (size(lhs, /type) eq 8) then begin
            if (lhs.type ne 4) then begin            
              gpuFix, lhs, _lhs, TYPE=4
            endif else begin
              _lhs = lhs
            endelse
          endif else begin
            _lhs = gpuFltarr(nxy, /NOZERO)
            _lhs.isTemporary = 1B
          endelse                     
          
          if (!gpu.mode eq 0) then begin
            *_lhs.data = randomu(seed, nx, ny, NORMAL=keyword_set(normal))            
          endif else begin
            if (n_elements(seed) ne 0) then err = gpuSeedMTF(long(seed))
            err = gpuMTF(nxy, res_gpu.handle)
            if (keyword_set(NORMAL)) then begin
              err = gpuBoxMullerF(long(nx), _lhs.handle)
            endif
          endelse
          gpuReform, _lhs, nx, ny          
        end
    else: message, level=-1, 'gpuRandomu: Invalid number of arguments'
  endcase
  
  return, _lhs
end


;+
; This routine generates uniformly distributed random numbers on the GPU
;
; :Params:
;    seed: in, required, type=int
;       seed for RNG. Can be undefined
;    nx : in, required, type=int
;       x-dimension of resulting random array
;    ny : out, optional, type=int
;       y-dimension of resulting random array
;    x_gpu : in, required, type={ GPUHANLE }
;       variable to store random numbers in
;
; :Keywords:
;    NORMAL: in, optional, type=bool
;       generate a normal distribution, rather than a uniform one
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuRandomu, seed, nx, ny, x_gpu, ERROR=err, NORMAL=normal 

  x_gpu = gpuRandomu(seed, nx, ny, LHS=x_gpu, ERROR=err, NORMAL=normal)
end

