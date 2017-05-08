; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpurandomn.pro
;
; Create normally distributed random numbers on the GPU.
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
; This routine generates normally distributed random numbers on the GPU
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
;    ERROR : out, optional, type=integer
;       error status
;-
function gpuRandomn, seed, nx, ny, LHS=lhs, ERROR=err
  return, gpuRandomu(seed, nx, ny, LHS=lhs, ERROR=err, /NORMAL)
end


;+
; This routine generates normally distributed random numbers on the GPU
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
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuRandomn, seed, nx, ny, x_gpu, ERROR=err
  gpuRandomu, seed, nx, ny, x_gpu, ERROR=err, /NORMAL
end 

