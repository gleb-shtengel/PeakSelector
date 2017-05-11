;
;  file:  palm_filterit.pro
;
;  Generic peak filtering routine.
;
;  RTK, 25-Jun-2009
;  Last update:  25-Jun-2009
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;--------------------------------------------------------------
;+
;  Predicate to check for CUDA availability
;
;  @returns  True (1) if CUDA available, false (0) otherwise.
;-
;--------------------------------------------------------------
function haveCUDA
    compile_opt idl2, logical_predicate

    ;  N.B. assumes that the DLMs have been loaded already

    err = 0
    catch, err
    if (err) then begin
        catch, /CANCEL
        return, 0b
    endif

    nc = gpu_device_count()
    return, (nc ne 0)
end


;--------------------------------------------------------------
;+
;  Re-implementation of original PeakSelector version.
;-
;--------------------------------------------------------------
pro FilterIt
    compile_opt idl2, logical_predicate

    mask = bytarr(45)
    mask[[indgen(18), 26,27,28,29,30,31,32,33,34,35]] = 1b
    PALM_FilterIt, mask
end


;--------------------------------------------------------------
;+
;  Re-implementation of original PeakSelector version.
;-
;--------------------------------------------------------------
pro GroupFilterIt
    compile_opt idl2, logical_predicate

    mask = bytarr(45)
    mask[[9,18,19,20,21,22,23,24,26,37,38,39,40,41,42]] = 1b
    PALM_FilterIt, mask
end


;--------------------------------------------------------------
;+
;  Filter the peaks according to the mask in filterIndex
;
;  @param filterIndex {in}{type=mask vector}{required}
;    A mask vector with a 1 in every place where the limits
;    should be checked.
;-
;--------------------------------------------------------------
pro PALM_FilterIt, filterIndex
    compile_opt idl2, logical_predicate

    common SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image,        $
                         b_set, xydsz, TotalRawData, RawFilenames, GuideStarDrift,  $
                         FiducialCoeff, FlipRotate
    common managed, ids, names, modalList

    if (n_elements(CGroupParams) eq 0) then begin
        void = dialog_message('Please load a data file')
        return
    endif

    topID = ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
  
    if (haveCUDA()) then begin
        ;  Use the GPU
        filter = cu_FilterIt_f(float(CGroupParams), ParamLimits, byte(filterIndex)) 
    endif else begin
        ;  IDL version - based on original
        CGPsz = size(CGroupParams)
        indices0 = where(filterIndex, count)
        if (count eq 0) then begin
            void = dialog_message('Warning: empty filter index')
            return
        endif
        check = where(ParamLimits[indices0,1] ne 0)
        indices = indices0[check]
        pl0 = ParamLimits[indices,0]#replicate(1,CGPsz[2])
        pl1 = ParamLimits[indices,1]#replicate(1,CGPsz[2])
        filter0 = (CGroupParams[indices,*] ge pl0) and (CGroupParams[indices,*] le pl1)
        filter = byte(floor(total(filter0,1)/n_elements(indices)))
    endelse

    peakCount = total(filter)
    if peakcount ge 1 then begin
        vp=finite(CGroupParams[*,where(filter)])
        vpcnt=round(total(vp)/CGrpSize)
        vpcnt1=round(total(vp)/50)
        if (peakcount ne vpcnt) and (peakcount ne vpcnt1) then begin
            print,'invalid peak parameters encountered'
            print,'total number of peaks=',peakcount,'   valid=',vpcnt
        endif
    endif

    wlabel = widget_info(topID, find_by_uname='WID_LABEL_NumberSelected')
    widget_control,wlabel,set_value='Peak Count = '+string(peakcount)
    print,'total filtering time (sec):  ',(systime(/seconds)-tstart),'     total filtered peaks:  ',peakcount
end

;  end palm_filterit.pro

