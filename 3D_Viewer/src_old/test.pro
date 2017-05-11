pro test

    common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image,  $
            b_set, xydsz, TotalRawData, RawFilenames, GuideStarDrift,          $
            FiducialCoeff, FlipRotate

    dataFile = 'test_data.sav'
    if ~file_test(dataFile) then begin
        dataFile = dialog_pickfile( $
            FILTER='*.sav', $
            TITLE='Select a data file')
        if dataFile EQ '' then $
            return
    endif
    restore, dataFile, /VERBOSE

    PALM_3DViewer, filt_item, func_item, $
        ACCUMULATION=Accumulation, $
        LABEL=label, $
        NANOS_PER_CCD_PIXEL=16000./2/60., $
        PARAMETER_LIMITS=ParamLimits, $
        /VERBOSE

end