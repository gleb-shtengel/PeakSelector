;
;  file:  configureenvironment.pro
;
;  Load DLMs used by PeakSelector and the 3D Viewer.
;  Also set flags for CUDA and GPUlib availability.
;
;  RTK, 30-Jun-2009
;  Last update:  02-Jul-2009
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

pro ConfigureEnvironment
    compile_opt idl2, logical_predicate


    ;  Get available DLMs
    help, /BRIEF, /DLM, OUTPUT=dlms
    t = strjoin(dlms, ' ')
    dlms = strsplit(t, ' ', /EXTRACT)

    void = where(dlms eq 'GPULIB', c)
    haveGPUlib = (c ne 0)
    void = where(dlms eq 'GPU_DEVICE_QUERY', c)
    haveGPUDeviceQuery = (c ne 0)

    if (haveGPUDeviceQuery) then begin
        void = where(dlms eq 'CU_FILTERIT_F', c)
        haveCUDAFilterIt = (c ne 0)
        void = where(dlms eq 'CU_POLY_3D', c)
        haveCUDAPoly3D = (c ne 0)
    endif else begin
        haveCUDAFilterIt = (haveCUDAPoly3D = 0)
    endelse

    ;  We need both the GPUlib DLM and .sav file
    if (haveGPUlib) then begin
        gpusav = filepath('gpulib.sav', SUBDIR=['products'])
        if (file_test(gpusav)) then begin
            restore, gpusav
            gpuinit
        endif else begin
            haveGPUlib = 0
        endelse
    endif
    
    ;  Load the DLMs found
    catch, err
    if (err) then begin
        haveGPUlib = 0  ;  error initializing GPUlib
    endif
    if (haveGPUlib) then dlm_load, 'GPULIB'
    catch, /CANCEL

    catch,err
    if (err) then begin
        haveGPUDeviceQuery = 0  ;  error initializing CUDA (no card)
        haveCUDAFilterIt = 0
        haveCUDAPoly3D = 0
    endif
    haveCUDA = 0
    if (haveGPUDeviceQuery) then begin
        dlm_load, 'GPU_DEVICE_QUERY'
        if (gpu_device_count() ne 0) then begin
            haveCUDA = 1
            if (haveCUDAFilterIt) then dlm_load, 'CU_FILTERIT_F'
            if (haveCUDAPoly3D) then dlm_load, 'CU_POLY_3D'
        endif
    endif
    catch, /CANCEL

    ;  Set global level flags
    (scope_varfetch('haveGPUlib', LEVEL=1, /ENTER)) = haveGPUlib
    (scope_varfetch('haveGPUDeviceQuery', LEVEL=1, /ENTER)) = haveGPUDeviceQuery
    (scope_varfetch('haveCUDA', LEVEL=1, /ENTER)) = haveCUDA
    (scope_varfetch('haveCUDAFilterIt', LEVEL=1, /ENTER)) = haveCUDAFilterIt
    (scope_varfetch('haveCUDAPoly3D', LEVEL=1, /ENTER)) = haveCUDAPoly3D
end

;  end configureenvironment.pro

