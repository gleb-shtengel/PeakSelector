; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuinit.pro
;
; IDL interface to the GPU library.
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
; Detects the installed GPU device.
; 
; :Params: 
;      devId :  in, out, optional, type=numtype 
;         id of the device to be used for GPU computations
;         
; :Returns: 
;    identifier for installed GPU device::
;    
;       1 for GPU hardware
;       0 for pure IDL emulation
;       -1 for GPU hardware emulation
;
;-
function gpuDetectDeviceMode, devId
  compile_opt strictarr
  
  catch, deviceError
  mode = 1S
  if (deviceError ne 0) then begin
    catch, /cancel
    mode = 0S
    devId = -1L
        
    return, 0
  end

  if (n_elements(devId) ne 0) then err = cudaSetDevice(devId)

  if (mode ne 0) then begin
    err = cudaGetDeviceProperties(prop, devId)
    if (prop.name eq 'Device Emulation (CPU)') then mode = -1S
  end

  return, mode 
end


;+
; Start up the IDL CUDA interface in either: hardware, emulator, or
; IDL mode. Unless a new mode is specfied, uses the previous mode if the CUDA
; interface has already been initialized. Defaults to hardware if no mode is
; specified and CUDA has not already been initialized.
;
; :Params:
;    devId: in, optional, type=numtype
;       id of the GPU device to be used for GPU computations
;       
; :Keywords:
;    HARDWARE : in, optional, type=boolean
;       set to run GPU library routine in hardware mode
;    EMULATOR : in, optional, type=boolean
;       set to run GPU library routines in emulator mode
;    IDL : in, optional, type=boolean
;       set to run GPU library routines in native IDL mode
;    ERROR : out, optional, type=integer
;       error status
;-
pro gpuinit, devId, HARDWARE=hardware, EMULATOR=emulator, IDL=idl, ERROR=err
  compile_opt strictarr
  on_error, 2

  err = 0

  ; at most one mode can be set
  if (keyword_set(hardware) + $
      keyword_set(emulator) + $
      keyword_set(idl) gt 1) then begin
    message, 'only one of HARDWARE, EMULATOR, or IDL may be set'
  endif

  _devId = n_elements(devId) eq 0L ? 0L : long(devId)
  mode = gpuDetectDeviceMode(_devId)
  defsysv, '!gpu', exists=alreadyInitialized
  
  ;  override detected device if desired
  case 1 of
    keyword_set(hardware) : mode = 1S
    keyword_set(emulator) : mode = -1S
    keyword_set(idl)      : begin
                              mode = 0S
                              _devId = -1L
                            end
    else                  : mode = (alreadyInitialized && n_elements(devId) eq 0) ? !gpu.mode : mode
  endcase

  devId = _devId

  ; don't try to compile pro/function routines if they are already compiled or 
  ; if running in the VM
  if (~alreadyInitialized && ~lmgr(/vm)) then begin
    routines = ['gpuputarr', 'gpugetarr', 'gpucopy', 'gpusubscript', $   ; access/transfer
                'gpuadd', 'gpusub', 'gpumult', 'gpudiv', $   ; binary ops
                'gpueq', 'gpuneq', 'gpult', 'gpugt', 'gpulteq', 'gpugteq', $   ; relational ops
                'gpusqrt', 'gpuexp', 'gpulog', 'gpulog10', $
                'gpusin', 'gpucos', 'gputan', 'gpuasin', 'gpuacos', 'gpuatan', $
                'gpuerf', 'gpulgamma', 'gputgamma', $
                'gputrunc', 'gpuround', 'gpurint', 'gpufloor', 'gpuceil', 'gpuabs', $   ; unary ops
                'gpucomplex', 'gpudcomplex', 'gpudouble', 'gpufloat', 'gpuFix', $   ; conversion routines
                'gpuimaginary', 'gpureal', 'gpureal_part', $
                'gpucongrid', 'gpufft', 'gpuinterpolate', 'gpumatrix_multiply', $   ; misc operations
                'gpurandomn', 'gpurandomu', 'gpuprefixsum']
                
    origQuiet = !quiet
    !quiet = 1
    resolve_routine, routines, /compile_full_file
    !quiet = origQuiet
  endif
  
  if (~alreadyInitialized) then begin
    defsysv, '!gpu', { !gpu, mode: mode, device: devId }
  endif else begin
    if (!gpu.mode * mode eq 0) then begin      
      if (!gpu.device ge 0 && devId ge 0 && !gpu.device ne devId) then begin
        message, 'hardware already initialized to device ' + strtrim(!gpu.device, 2), $
                 /informational
      endif else if (mode ne 0) then begin
        !gpu.device = devId
      endif
      !gpu.mode = mode
    endif
  endelse

  if (!gpu.mode ne 0) then err = cublasInit()
end
