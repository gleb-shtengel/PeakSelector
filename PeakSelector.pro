; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	12/14/2017 12:51.54
; 
pro WID_BASE_0_PeakSelector_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_0_PeakSelector'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TABLE_0'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        InsertChar, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Peak_Centers'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPeakCentersButton, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DRAW_0'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DRAW' )then $
        if( Event.type eq 0 )then $
          OnButtonDraw0, Event
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DRAW' )then $
        if( Event.type eq 1 )then $
          OnButtonDraw0, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_32'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnExtractPeaks, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Extract_Peaks_multiple_TIFFs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnExtractPeaks_Multiple_TIFFs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_iPALM_Macro'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        iPALM_Macro, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_53'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnExtractPeaksML, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_40'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        ReExtractPeaksMultiLabel, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_5'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OpenTheFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_1'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        AddtotheFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Save_Data_ASCII_txt'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Save_Data_ASCII_txt, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_2'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        SavetheCommon, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_8'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        RecalltheCommon, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_39'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        AddNextLabelData, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_25'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        MergeLabelsConsecutively, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Load_DIC'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        LoadDIC, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_50'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPurgeButton, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_reassign_peak_indecis'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Reassign_Peak_Indecis_within_Frames, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_51'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Purge_Filtered, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_52'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Purge_Group_Filtered, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Purge_current'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Purge_current_filter, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Purge_selected'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Purge_selected_peaks, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Purge_by_XY_coords'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Purge_by_XY_coords, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_0'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        SaveImageJPEG, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_22'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        SaveImageBMP, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_58'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        SaveImageTIFF, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_23'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        SaveImageIDLData, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_63'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Export_Hist_ASCII, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_55'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        PrintImage, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_ImportUserASCII'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        ImportUserASCII, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_ImportZeissTXT'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        ImportZeissTXt, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_3'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        StopthePeakSelect, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Edit_Preferences'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        EditPeakSelectorPreferences, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_6'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        ExittheFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_27'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnTranspose, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_28'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnFlipHorizontal, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_29'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnFlipVertical, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_30'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnRotate90CW, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_31'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnRotate90CCW, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_14'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnTwist, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_47'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        TestWriteGuideStar, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_XY_Interp'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        XY_Interpolation, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_36'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAnchorFiducialMenu, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_34'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        SaveTransformCoeffs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_38'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        TransformRaw_Save_SaveSum_MenuItem, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_35'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        TransformRaw_Save_UserDefined, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Transform_SpAnalysis'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Transform_SpAnalysis, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Convert_Pixels_to_NM'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Convert_X_to_wavelength, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_46'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnGroupPeaks, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_44'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        ZCoordinateOperations, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Zastig'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        ZCoordinateOperations_Astig, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_ProcessMultiplePalmSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Process_Multiple_Palm_Slabs_call, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_Polarization_Analysis'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Polarization_Analysis, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_64'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Recalculate_XpkwYpkw, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Reprocess_Palm_Set'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Reprocess_Palm_Set, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_ApplyFilterSelectively'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Change_Filter_Select, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_7'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAutoCorrelPeak, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_12'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAutoCorrelateGroups, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_16'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSmooth3, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_21'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSmooth5, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_24'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSmooth9, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_17'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Analyze1, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_SwapXZ'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSwapXZ, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_SwapYZ'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSwapYZ, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_65'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Swap_Z_Unwrapped_Z, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_18'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAnalyzePeakDistribution, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_GroupDistribution'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAnalyzeGroupDistribution, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_20'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAnalyze3, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_menu_Analyze_Fiducial_Alignement_2Colors'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Analyze_Fiducial_Alignement_2Colors, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_62'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAnalyze4, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_AnalyzePhaseUnwrap'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_AnalyzePhaseUnwrap, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_process_SRM'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Process_SRM, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Renumber_GP'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Renumber_Group_Peaks, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Correct_GroupSigmaXYZ'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Correct_GroupSigmaXYZ, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Calculate_MSD'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Calculate_MSD, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_10'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        ColorTheTable, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_54'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        DoToggleZtoHue, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_59'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        ToggleLegends, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_MoveColorBar'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnColorBar_Move, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_ColorBarExtend'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnColorBar_Extend, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_XYZ_diamonds'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_XYZ_use_diamonds, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Set_nm_per_pixel_scale'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_nm_per_pixel_scale, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_DisplayThisFitCond'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        DisplayThisFitCond, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_66W_MENU_set_Z_scale_multiplier'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_Z_scale_multiplier, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_vbar_top'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_vbar_top, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_SetMaxProbability_2DPALM'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        SetMaxProbability_2DPALM, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Force_MaxProb_2DPALM'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnForce_MaxProbability_2DPALM, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_IgnoreLblsHist'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnIgnoreLblsHist, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Save_Histograms_BMP'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnSaveHistBMP, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='W_MENU_Replace_TotalRaw_with_Rendered'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Replace_TotalRaw_with_Rendered, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Group_Centers'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnGroupCentersButton, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_UnZoom'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnUnZoomButton, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_TotalRawData'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnTotalRawDataButton, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PloyXY'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPlotXYButton, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_Bot'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_SLIDER' )then $
        OnStretchBottom, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_Top'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_SLIDER' )then $
        OnStretchTop, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_Gamma'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_SLIDER' )then $
        OnGamma, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Render'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnRenderButton, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_ScaleBar'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAddScaleBarButton, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_ColorBar'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAddColorBarButton, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_ImageLabel'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAddLabelButton, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_RawFrameNumber'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_SLIDER' )then $
        OnRawFrameNumber, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_RawPeakIndex'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_SLIDER' )then $
        OnRawPeakIndex, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_ScaleBar2'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAddScaleBarButton2, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_Label'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        OnLabelDropList, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_RawFileName'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        SetRawSliders, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Back1Step'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On1StepBack, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_UnZoom_2X'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnUnZoom2X, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Plot_XgrYgrZgr'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPlotXgrYgrZgr, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Plot_XgrZgrYgr'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPlotXgrZgrXgr, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Plot_YgrZgrXgr'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPlotYgrZgrXgr, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Plot_FrameZX'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPlotFrameZX, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Stat3DViewer'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnStatic3DViewer, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Reload_Paramlimits'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        ReloadParamlists, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Plot_XgrUnwZgrYgr'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPlotXgrUnwZgrXgr, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Plot_YgrUnwZgrXgr'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPlotYgrUnwZgrXgr, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Plot_XgrUnwZgrLbl'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPlotXgrUnwZgrLbl, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_FilterLabel'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAddSigmaFilterButton, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_AddAllLabels'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnAddAllLabels, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Plot_XZLbl'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPlotXZLbl, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Overlay_All_Centers'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPeakOverlayAllCentersButton, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Overlay_Frame_Centers'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPeakOverlayFrameCentersButton, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Overlay_DIC'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OverlayDIC, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_CustomTIFF'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        SaveCustomTIFF_menu, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Plot_FrameUnwrZX'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPlotFrameUnwrZX, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Plot_spectrum'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPeak_Plot_Spectrum, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Hist_Log_X'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_Hist_Log_X, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Hist_Log_Y'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_Hist_Log_Y, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Allow_Bridge'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_Allow_Bridge, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Tie_RGB'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_Tie_RGB, Event
    end
    else:
  endcase

end

pro WID_BASE_0_PeakSelector, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'PeakSelector_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_0_PeakSelector = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_0_PeakSelector' ,XOFFSET=5 ,YOFFSET=1  $
      ,SCR_XSIZE=1622 ,SCR_YSIZE=1050  $
      ,NOTIFY_REALIZE='Initialization_PeakSelector_Main' ,/SCROLL  $
      ,XSIZE=1622 ,YSIZE=1070 ,TITLE='Peak Selector v9.6© 2017,'+ $
      ' Howard Hughes Medical Institute.  All rights reserved.'  $
      ,SPACE=3 ,XPAD=3 ,YPAD=3 ,MBAR=WID_BASE_0_PeakSelector_MBAR)

  
  WID_TABLE_0 = Widget_Table(WID_BASE_0_PeakSelector,  $
      UNAME='WID_TABLE_0' ,FRAME=1 ,XOFFSET=4 ,YOFFSET=65  $
      ,SCR_XSIZE=550 ,SCR_YSIZE=490 ,/EDITABLE ,/RESIZEABLE_COLUMNS  $
      ,COLUMN_LABELS=[ 'Lower Bound', 'Upper Bound', 'Center',  $
      'Range', 'Peak' ] ,ROW_LABELS=[ 'Offset', 'Amplitude', 'X'+ $
      ' Position', 'Y Position', 'X Peak Width', 'Y Peak Width', '6 N'+ $
      ' Photons', 'ChiSquared', 'FitOK', 'Frame Number', 'Peak Index'+ $
      ' of Frame', 'Peak Global Index', '12 Sigma Offset', 'Sigma'+ $
      ' Amplitude', 'Sigma X Pos rtNph', 'Sigma Y Pos rtNph', 'Sigma'+ $
      ' X Pos Full', 'Sigma Y Pos Full', '18 Grouped Index', 'Group X'+ $
      ' Position', 'Group Y Position', 'Group Sigma X Pos', 'Group'+ $
      ' Sigma Y Pos', 'Group N Photons', '24 Group Size', 'Frame'+ $
      ' Index in Grp', 'Label Set' ] ,XSIZE=5 ,YSIZE=50)

  
  WID_BUTTON_Peak_Centers = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Peak_Centers' ,XOFFSET=15 ,YOFFSET=605  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Peak'+ $
      ' Centers')

  
  WID_DRAW_0 = Widget_Draw(WID_BASE_0_PeakSelector,  $
      UNAME='WID_DRAW_0' ,FRAME=1 ,XOFFSET=567 ,SCR_XSIZE=1024  $
      ,SCR_YSIZE=1024 ,NOTIFY_REALIZE='OnDraw0Realize'  $
      ,/BUTTON_EVENTS)

  
  W_MENU_4 = Widget_Button(WID_BASE_0_PeakSelector_MBAR,  $
      UNAME='W_MENU_4' ,/MENU ,VALUE='File')

  
  W_MENU_32 = Widget_Button(W_MENU_4, UNAME='W_MENU_32'  $
      ,VALUE='Extract Peaks .sif/ .tif/ .txt/.dat to .pks')

  
  W_MENU_Extract_Peaks_multiple_TIFFs = Widget_Button(W_MENU_4,  $
      UNAME='W_MENU_Extract_Peaks_multiple_TIFFs' ,VALUE='Extract'+ $
      ' Peaks: Multiple TIFF files')

  
  W_MENU_iPALM_Macro = Widget_Button(W_MENU_4,  $
      UNAME='W_MENU_iPALM_Macro' ,VALUE='iPALM Macro: Transform'+ $
      ' Extract Reextract Group GetZ')

  
  W_MENU_53 = Widget_Button(W_MENU_4, UNAME='W_MENU_53'  $
      ,VALUE='Extract Peaks Multiple Labels')

  
  W_MENU_40 = Widget_Button(W_MENU_4, UNAME='W_MENU_40'  $
      ,VALUE='ReExtract Peaks Multi-Label')

  
  W_MENU_5 = Widget_Button(W_MENU_4, UNAME='W_MENU_5' ,VALUE='Open'+ $
      ' .pks File')

  
  W_MENU_1 = Widget_Button(W_MENU_4, UNAME='W_MENU_1' ,VALUE='Add'+ $
      ' .pks File')

  
  W_MENU_Save_Data_ASCII_txt = Widget_Button(W_MENU_4,  $
      UNAME='W_MENU_Save_Data_ASCII_txt' ,VALUE='Export Processed'+ $
      ' Data as ASCII (.txt)')

  
  W_MENU_2 = Widget_Button(W_MENU_4, UNAME='W_MENU_2' ,VALUE='Save'+ $
      ' Proccessed as IDL(.sav)')

  
  W_MENU_8 = Widget_Button(W_MENU_4, UNAME='W_MENU_8' ,VALUE='Recall'+ $
      ' Proccessed IDL(.sav)')

  
  W_MENU_39 = Widget_Button(W_MENU_4, UNAME='W_MENU_39' ,VALUE='Add'+ $
      ' Next Label IDL(.sav)')

  
  W_MENU_25 = Widget_Button(W_MENU_4, UNAME='W_MENU_25' ,VALUE='Merge'+ $
      ' Labels Consecutively')

  
  W_MENU_Load_DIC = Widget_Button(W_MENU_4, UNAME='W_MENU_Load_DIC'  $
      ,VALUE='Load DIC / EM')

  
  W_MENU_42 = Widget_Button(W_MENU_4, UNAME='W_MENU_42' ,/MENU  $
      ,VALUE='Purge Peaks')

  
  W_MENU_50 = Widget_Button(W_MENU_42, UNAME='W_MENU_50' ,VALUE='Not'+ $
      ' OK Peaks')

  
  W_MENU_reassign_peak_indecis = Widget_Button(W_MENU_42,  $
      UNAME='W_MENU_reassign_peak_indecis' ,VALUE='Reassign Peak'+ $
      ' Indecis within Each Frame')

  
  W_MENU_51 = Widget_Button(W_MENU_42, UNAME='W_MENU_51'  $
      ,VALUE='Filtered Peaks')

  
  W_MENU_52 = Widget_Button(W_MENU_42, UNAME='W_MENU_52'  $
      ,VALUE='Group Filtered Peaks')

  
  W_MENU_Purge_current = Widget_Button(W_MENU_42,  $
      UNAME='W_MENU_Purge_current' ,VALUE='Purge with current'+ $
      ' filter')

  
  W_MENU_Purge_selected = Widget_Button(W_MENU_42,  $
      UNAME='W_MENU_Purge_selected' ,VALUE='Purge Selected Peaks'+ $
      ' (inverted filter)')

  
  W_MENU_Purge_by_XY_coords = Widget_Button(W_MENU_42,  $
      UNAME='W_MENU_Purge_by_XY_coords' ,VALUE='Filtered by X-Y'+ $
      ' Coordinates')

  
  W_MENU_15 = Widget_Button(W_MENU_4, UNAME='W_MENU_15' ,/MENU  $
      ,VALUE='Save Image')

  
  W_MENU_0 = Widget_Button(W_MENU_15, UNAME='W_MENU_0' ,VALUE='Save'+ $
      ' Image .jpeg')

  
  W_MENU_22 = Widget_Button(W_MENU_15, UNAME='W_MENU_22' ,VALUE='Save'+ $
      ' Image .bmp')

  
  W_MENU_58 = Widget_Button(W_MENU_15, UNAME='W_MENU_58' ,VALUE='Save'+ $
      ' Image .tiff')

  
  W_MENU_23 = Widget_Button(W_MENU_15, UNAME='W_MENU_23' ,VALUE='Save'+ $
      ' Image IDL Data (.sav)')

  
  W_MENU_63 = Widget_Button(W_MENU_4, UNAME='W_MENU_63'  $
      ,VALUE='Export Histogram ASCII')

  
  W_MENU_55 = Widget_Button(W_MENU_4, UNAME='W_MENU_55' ,VALUE='Print'+ $
      ' Window')

  
  W_MENU_ImportASCII = Widget_Button(W_MENU_4,  $
      UNAME='W_MENU_ImportASCII' ,/MENU ,VALUE='Import ASCII')

  
  W_MENU_ImportUserASCII = Widget_Button(W_MENU_ImportASCII,  $
      UNAME='W_MENU_ImportUserASCII' ,VALUE='Import User ASCII')

  
  W_MENU_ImportZeissTXT = Widget_Button(W_MENU_ImportASCII,  $
      UNAME='W_MENU_ImportZeissTXT' ,VALUE='Import Zeiss TXT')

  
  W_MENU_3 = Widget_Button(W_MENU_4, UNAME='W_MENU_3'  $
      ,VALUE='StopPeakSelect')

  
  W_MENU_Edit_Preferences = Widget_Button(W_MENU_4,  $
      UNAME='W_MENU_Edit_Preferences' ,VALUE='Edit Preferences')

  
  W_MENU_6 = Widget_Button(W_MENU_4, UNAME='W_MENU_6' ,VALUE='Exit')
  
  W_MENU_41 = Widget_Button(WID_BASE_0_PeakSelector_MBAR,  $
      UNAME='W_MENU_41' ,/MENU ,VALUE='Image Transformations')

  
  W_MENU_26 = Widget_Button(W_MENU_41, UNAME='W_MENU_26' ,/MENU  $
      ,VALUE='Orientation')

  
  W_MENU_27 = Widget_Button(W_MENU_26, UNAME='W_MENU_27'  $
      ,VALUE='Transpose')

  
  W_MENU_28 = Widget_Button(W_MENU_26, UNAME='W_MENU_28' ,VALUE='Flip'+ $
      ' Horizontal')

  
  W_MENU_29 = Widget_Button(W_MENU_26, UNAME='W_MENU_29' ,VALUE='Flip'+ $
      ' Vertical')

  
  W_MENU_30 = Widget_Button(W_MENU_26, UNAME='W_MENU_30'  $
      ,VALUE='Rotate 90 CW')

  
  W_MENU_31 = Widget_Button(W_MENU_26, UNAME='W_MENU_31'  $
      ,VALUE='Rotate 90 CCW')

  
  W_MENU_14 = Widget_Button(W_MENU_41, UNAME='W_MENU_14'  $
      ,VALUE='Rotate')

  
  W_MENU_47 = Widget_Button(W_MENU_41, UNAME='W_MENU_47'  $
      ,VALUE='Test/Write Guide Star')

  
  W_MENU_XY_Interp = Widget_Button(W_MENU_41,  $
      UNAME='W_MENU_XY_Interp' ,VALUE='Interpolate / Subtract XY'+ $
      ' trend')

  
  W_MENU_36 = Widget_Button(W_MENU_41, UNAME='W_MENU_36'  $
      ,VALUE='Anchor Fiducial Pnts')

  
  W_MENU_34 = Widget_Button(W_MENU_41, UNAME='W_MENU_34' ,VALUE='Save'+ $
      ' TransformCoeffs (.sav)')

  
  W_MENU_38 = Widget_Button(W_MENU_41, UNAME='W_MENU_38'  $
      ,VALUE='Transform Raw, Save and Save Sum (.dat)')

  
  W_MENU_35 = Widget_Button(W_MENU_41, UNAME='W_MENU_35'  $
      ,VALUE='Transform Raw and Save User-Defined (.dat)')

  
  W_MENU_Transform_SpAnalysis = Widget_Button(W_MENU_41,  $
      UNAME='W_MENU_Transform_SpAnalysis' ,VALUE='Transform Spectral'+ $
      ' Data and Analyze Spectra')

  
  W_MENU_Convert_Pixels_to_NM = Widget_Button(W_MENU_41,  $
      UNAME='W_MENU_Convert_Pixels_to_NM' ,VALUE='Convert X-Y data to'+ $
      ' Wavelength-Y data')

  
  W_MENU_11 = Widget_Button(WID_BASE_0_PeakSelector_MBAR,  $
      UNAME='W_MENU_11' ,/MENU ,VALUE='Special Functions')

  
  W_MENU_46 = Widget_Button(W_MENU_11, UNAME='W_MENU_46'  $
      ,VALUE='Group Peaks')

  
  W_MENU_44 = Widget_Button(W_MENU_11, UNAME='W_MENU_44' ,VALUE=''+ $
      ' Z-coordinate Operations')

  
  W_MENU_Zastig = Widget_Button(W_MENU_11, UNAME='W_MENU_Zastig'  $
      ,VALUE='Z-coordinate Operations - Astig Only ')

  
  W_MENU_ProcessMultiplePalmSlabs = Widget_Button(W_MENU_11,  $
      UNAME='W_MENU_ProcessMultiplePalmSlabs' ,VALUE='Process'+ $
      ' Multiple PALM Slabs')

  
  W_Polarization_Analysis = Widget_Button(W_MENU_11,  $
      UNAME='W_Polarization_Analysis' ,VALUE='Polarization Analysis')

  
  W_MENU_64 = Widget_Button(W_MENU_11, UNAME='W_MENU_64'  $
      ,VALUE='Recalculate CGroupParams[12,*]')

  
  W_MENU_Reprocess_Palm_Set = Widget_Button(W_MENU_11,  $
      UNAME='W_MENU_Reprocess_Palm_Set' ,VALUE='Reprocess Palm Set')

  
  W_MENU_ApplyFilterSelectively = Widget_Button(W_MENU_11,  $
      UNAME='W_MENU_ApplyFilterSelectively' ,/CHECKED_MENU  $
      ,VALUE='Apply Filter Only to Selected Label')

  
  W_MENU_19 = Widget_Button(W_MENU_11, UNAME='W_MENU_19' ,/MENU  $
      ,VALUE='Autocorrelate')

  
  W_MENU_7 = Widget_Button(W_MENU_19, UNAME='W_MENU_7' ,VALUE='Auto'+ $
      ' Correlate Peaks')

  
  W_MENU_12 = Widget_Button(W_MENU_19, UNAME='W_MENU_12' ,VALUE='Auto'+ $
      ' Correlate Group Start')

  
  W_MENU_16 = Widget_Button(W_MENU_11, UNAME='W_MENU_16'  $
      ,VALUE='Smooth 3')

  
  W_MENU_21 = Widget_Button(W_MENU_11, UNAME='W_MENU_21'  $
      ,VALUE='Smooth 5')

  
  W_MENU_24 = Widget_Button(W_MENU_11, UNAME='W_MENU_24'  $
      ,VALUE='Smooth 9')

  
  W_MENU_17 = Widget_Button(W_MENU_11, UNAME='W_MENU_17'  $
      ,VALUE='Analyze1 Intensity Triplets')

  
  W_MENU_SwapXZ = Widget_Button(W_MENU_11, UNAME='W_MENU_SwapXZ'  $
      ,/CHECKED_MENU ,VALUE='Swap X-Z')

  
  W_MENU_SwapYZ = Widget_Button(W_MENU_11, UNAME='W_MENU_SwapYZ'  $
      ,/CHECKED_MENU ,VALUE='Swap Y-Z')

  
  W_MENU_65 = Widget_Button(W_MENU_11, UNAME='W_MENU_65'  $
      ,/CHECKED_MENU ,VALUE='Swap Z with Unwrapped Z')

  
  W_MENU_18 = Widget_Button(W_MENU_11, UNAME='W_MENU_18'  $
      ,VALUE='Analyze2 Peak Distribution Statistics')

  
  W_MENU_GroupDistribution = Widget_Button(W_MENU_11,  $
      UNAME='W_MENU_GroupDistribution' ,VALUE='Analyze2 Group'+ $
      ' Distribution Statistics')

  
  W_MENU_20 = Widget_Button(W_MENU_11, UNAME='W_MENU_20'  $
      ,VALUE='Analyze3 Peak Statistics for Multiple Areas')

  
  W_menu_Analyze_Fiducial_Alignement_2Colors =  $
      Widget_Button(W_MENU_11,  $
      UNAME='W_menu_Analyze_Fiducial_Alignement_2Colors'  $
      ,VALUE='Analyze Fiducial Alignement for 2Colors')

  
  W_MENU_62 = Widget_Button(W_MENU_11, UNAME='W_MENU_62'  $
      ,VALUE='Analyze4 Show iPALM PSF Raw and Fits')

  
  W_MENU_AnalyzePhaseUnwrap = Widget_Button(W_MENU_11,  $
      UNAME='W_MENU_AnalyzePhaseUnwrap' ,VALUE='Analyze Phase Unwrap'+ $
      ' and Localizations')

  
  W_MENU_process_SRM = Widget_Button(W_MENU_11,  $
      UNAME='W_MENU_process_SRM' ,VALUE='Process SRM')

  
  W_MENU_Renumber_GP = Widget_Button(W_MENU_11,  $
      UNAME='W_MENU_Renumber_GP' ,VALUE='Re-number Group Peaks')

  
  W_MENU_Correct_GroupSigmaXYZ = Widget_Button(W_MENU_11,  $
      UNAME='W_MENU_Correct_GroupSigmaXYZ' ,VALUE='Correct'+ $
      ' GroupSigmaXYZ (*1.414)')

  
  W_MENU_Calculate_MSD = Widget_Button(W_MENU_11,  $
      UNAME='W_MENU_Calculate_MSD' ,VALUE='Calculate Mean Squared'+ $
      ' Displacement')

  
  W_MENU_9 = Widget_Button(WID_BASE_0_PeakSelector_MBAR,  $
      UNAME='W_MENU_9' ,/MENU ,VALUE='Display')

  
  W_MENU_10 = Widget_Button(W_MENU_9, UNAME='W_MENU_10' ,VALUE='Color'+ $
      ' Table')

  
  W_MENU_54 = Widget_Button(W_MENU_9, UNAME='W_MENU_54'  $
      ,/CHECKED_MENU ,VALUE='Z using Color Table -> Hue')

  
  W_MENU_59 = Widget_Button(W_MENU_9, UNAME='W_MENU_59'  $
      ,/CHECKED_MENU ,VALUE='Render Without Legends')

  
  W_MENU_MoveColorBar = Widget_Button(W_MENU_9,  $
      UNAME='W_MENU_MoveColorBar' ,/CHECKED_MENU ,VALUE='Move Color'+ $
      ' Bar to the Top')

  
  W_MENU_ColorBarExtend = Widget_Button(W_MENU_9,  $
      UNAME='W_MENU_ColorBarExtend' ,/CHECKED_MENU ,VALUE='Extend'+ $
      ' Color Bar')

  
  W_MENU_XYZ_diamonds = Widget_Button(W_MENU_9,  $
      UNAME='W_MENU_XYZ_diamonds' ,/CHECKED_MENU ,VALUE='XYZ Plot:'+ $
      ' Use Diamonds')

  
  W_MENU_Set_nm_per_pixel_scale = Widget_Button(W_MENU_9,  $
      UNAME='W_MENU_Set_nm_per_pixel_scale' ,VALUE='Set nm/pixel'+ $
      ' Scale')

  
  W_MENU_DisplayThisFitCond = Widget_Button(W_MENU_9,  $
      UNAME='W_MENU_DisplayThisFitCond' ,VALUE='Display ThisFitCond')

  
  W_MENU_66W_MENU_set_Z_scale_multiplier = Widget_Button(W_MENU_9,  $
      UNAME='W_MENU_66W_MENU_set_Z_scale_multiplier' ,VALUE='Set Z'+ $
      ' scale multiplier (X-Y swaps)')

  
  W_vbar_top = Widget_Button(W_MENU_9, UNAME='W_vbar_top' ,VALUE='Set'+ $
      ' Z Color Bar Scale')

  
  W_MENU_SetMaxProbability_2DPALM = Widget_Button(W_MENU_9,  $
      UNAME='W_MENU_SetMaxProbability_2DPALM' ,VALUE='Set Max'+ $
      ' Probability for 2D PALM images')

  
  W_MENU_Force_MaxProb_2DPALM = Widget_Button(W_MENU_9,  $
      UNAME='W_MENU_Force_MaxProb_2DPALM' ,/CHECKED_MENU  $
      ,VALUE='Force Max Probability for 2D PALM images')

  
  W_MENU_IgnoreLblsHist = Widget_Button(W_MENU_9,  $
      UNAME='W_MENU_IgnoreLblsHist' ,/CHECKED_MENU ,VALUE='Ignore'+ $
      ' Labels For Histograms')

  
  W_MENU_Save_Histograms_BMP = Widget_Button(W_MENU_9,  $
      UNAME='W_MENU_Save_Histograms_BMP' ,VALUE='Save Histograms as'+ $
      ' BMP')

  
  W_MENU_Replace_TotalRaw_with_Rendered = Widget_Button(W_MENU_9,  $
      UNAME='W_MENU_Replace_TotalRaw_with_Rendered' ,VALUE='Replace'+ $
      ' TotalRaw with Rendered Image')

  
  WID_BUTTON_Group_Centers = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Group_Centers' ,XOFFSET=15 ,YOFFSET=577  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Group'+ $
      ' Centers')

  
  WID_BUTTON_UnZoom = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_UnZoom' ,XOFFSET=400 ,YOFFSET=597  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='UnZoomed'+ $
      ' Centers')

  
  WID_LABEL_NumberSelected = Widget_Label(WID_BASE_0_PeakSelector,  $
      UNAME='WID_LABEL_NumberSelected' ,XOFFSET=8 ,YOFFSET=559  $
      ,SCR_XSIZE=258 ,SCR_YSIZE=12 ,/ALIGN_LEFT ,VALUE='Number of'+ $
      ' Selected Peaks')

  
  WID_BUTTON_TotalRawData = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_TotalRawData' ,XOFFSET=15 ,YOFFSET=633  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Total Raw'+ $
      ' Data')

  
  WID_BUTTON_PloyXY = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_PloyXY' ,XOFFSET=10 ,YOFFSET=742  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Plot X vs Y'+ $
      ' - color  Z')

  
  WID_DROPLIST_X = Widget_Droplist(WID_BASE_0_PeakSelector,  $
      UNAME='WID_DROPLIST_X' ,XOFFSET=5 ,YOFFSET=780 ,SCR_XSIZE=170  $
      ,SCR_YSIZE=23 ,TITLE='X Axis' ,VALUE=[ 'Offset', 'Amplitude',  $
      'X Position', 'Y Position', 'X Peak Width', 'Y Peak Width', '6'+ $
      ' N Photons', 'ChiSquared', 'FitOK', 'Frame Number', 'Peak'+ $
      ' Index of Frame', 'Peak Global Index', '12 Sigma Offset',  $
      'Sigma Amplitude', 'Sigma X Pos rtNph', 'Sigma Y Pos rtNph',  $
      'Sigma X Pos Full', 'Sigma Y Pos Full', '18 Grouped Index',  $
      'Group X Position', 'Group Y Position', 'Group Sigma X Pos',  $
      'Group Sigma Y Pos', 'Group N Photons', '24 Group Size', 'Frame'+ $
      ' Index in Grp', 'Label Set' ])

  
  WID_SLIDER_Bot = Widget_Slider(WID_BASE_0_PeakSelector,  $
      UNAME='WID_SLIDER_Bot' ,XOFFSET=385 ,YOFFSET=957 ,SCR_XSIZE=160  $
      ,SCR_YSIZE=46 ,TITLE='Stretch Bottom' ,MAXIMUM=1000)

  
  WID_SLIDER_Top = Widget_Slider(WID_BASE_0_PeakSelector,  $
      UNAME='WID_SLIDER_Top' ,XOFFSET=385 ,YOFFSET=847 ,SCR_XSIZE=160  $
      ,SCR_YSIZE=46 ,TITLE='Stretch Top' ,MAXIMUM=1000 ,VALUE=500)

  
  WID_SLIDER_Gamma = Widget_Slider(WID_BASE_0_PeakSelector,  $
      UNAME='WID_SLIDER_Gamma' ,XOFFSET=385 ,YOFFSET=902  $
      ,SCR_XSIZE=160 ,SCR_YSIZE=46 ,TITLE='Gamma' ,MAXIMUM=1000  $
      ,VALUE=500)

  
  WID_DROPLIST_Accumulate = Widget_Droplist(WID_BASE_0_PeakSelector,  $
      UNAME='WID_DROPLIST_Accumulate' ,XOFFSET=180 ,YOFFSET=632  $
      ,SCR_XSIZE=170 ,SCR_YSIZE=25 ,TITLE='Accumulation' ,VALUE=[  $
      'Envelope', 'Sum' ])

  
  WID_DROPLIST_Function = Widget_Droplist(WID_BASE_0_PeakSelector,  $
      UNAME='WID_DROPLIST_Function' ,XOFFSET=180 ,YOFFSET=572  $
      ,SCR_XSIZE=170 ,SCR_YSIZE=25 ,TITLE='Function' ,VALUE=[ 'Center'+ $
      ' Locations', 'Gaussian Normalized', 'Gaussian Amplitude' ])

  
  WID_DROPLIST_Filter = Widget_Droplist(WID_BASE_0_PeakSelector,  $
      UNAME='WID_DROPLIST_Filter' ,XOFFSET=180 ,YOFFSET=602  $
      ,SCR_XSIZE=170 ,SCR_YSIZE=25 ,TITLE='Filter' ,VALUE=[ 'Frame'+ $
      ' Peaks', 'Grouped Peaks' ])

  
  WID_BUTTON_Render = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Render' ,XOFFSET=400 ,YOFFSET=560  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Render')

  
  WID_BUTTON_ScaleBar = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_ScaleBar' ,XOFFSET=196 ,YOFFSET=902  $
      ,SCR_XSIZE=95 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Add Scale'+ $
      ' Bar1')

  
  WID_BUTTON_ColorBar = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_ColorBar' ,XOFFSET=176 ,YOFFSET=934  $
      ,SCR_XSIZE=120 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Add Color'+ $
      ' Bar')

  
  WID_BUTTON_ImageLabel = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_ImageLabel' ,XOFFSET=175 ,YOFFSET=966  $
      ,SCR_XSIZE=95 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Add Image'+ $
      ' Lbl.')

  
  WID_SLIDER_RawFrameNumber = Widget_Slider(WID_BASE_0_PeakSelector,  $
      UNAME='WID_SLIDER_RawFrameNumber' ,XOFFSET=21 ,YOFFSET=898  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=46 ,TITLE='Raw Frame Number')

  
  WID_SLIDER_RawPeakIndex = Widget_Slider(WID_BASE_0_PeakSelector,  $
      UNAME='WID_SLIDER_RawPeakIndex' ,XOFFSET=20 ,YOFFSET=948  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=46 ,TITLE='Raw Peak Index of Frame')

  
  WID_BUTTON_ScaleBar2 = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_ScaleBar2' ,XOFFSET=301 ,YOFFSET=902  $
      ,SCR_XSIZE=35 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Bar2')

  
  WID_DROPLIST_Label = Widget_Droplist(WID_BASE_0_PeakSelector,  $
      UNAME='WID_DROPLIST_Label' ,XOFFSET=450 ,YOFFSET=812  $
      ,SCR_XSIZE=90 ,SCR_YSIZE=23 ,TITLE='Label' ,VALUE=[ '', 'Red',  $
      'Green', 'Blue', 'DIC / EM' ])

  
  WID_DROPLIST_Y = Widget_Droplist(WID_BASE_0_PeakSelector,  $
      UNAME='WID_DROPLIST_Y' ,XOFFSET=190 ,YOFFSET=780 ,SCR_XSIZE=170  $
      ,SCR_YSIZE=23 ,TITLE='Y Axis' ,VALUE=[ 'Offset', 'Amplitude',  $
      'X Position', 'Y Position', 'X Peak Width', 'Y Peak Width', '6'+ $
      ' N Photons', 'ChiSquared', 'FitOK', 'Frame Number', 'Peak'+ $
      ' Index of Frame', 'Peak Global Index', '12 Sigma Offset',  $
      'Sigma Amplitude', 'Sigma X Pos rtNph', 'Sigma Y Pos rtNph',  $
      'Sigma X Pos Full', 'Sigma Y Pos Full', '18 Grouped Index',  $
      'Group X Position', 'Group Y Position', 'Group Sigma X Pos',  $
      'Group Sigma Y Pos', 'Group N Photons', '24 Group Size', 'Frame'+ $
      ' Index in Grp', 'Label Set' ])

  
  WID_DROPLIST_Z = Widget_Droplist(WID_BASE_0_PeakSelector,  $
      UNAME='WID_DROPLIST_Z' ,XOFFSET=375 ,YOFFSET=780 ,SCR_XSIZE=170  $
      ,SCR_YSIZE=23 ,TITLE='Color' ,VALUE=[ 'Offset', 'Amplitude', 'X'+ $
      ' Position', 'Y Position', 'X Peak Width', 'Y Peak Width', '6 N'+ $
      ' Photons', 'ChiSquared', 'FitOK', 'Frame Number', 'Peak Index'+ $
      ' of Frame', 'Peak Global Index', '12 Sigma Offset', 'Sigma'+ $
      ' Amplitude', 'Sigma X Pos rtNph', 'Sigma Y Pos rtNph', 'Sigma'+ $
      ' X Pos Full', 'Sigma Y Pos Full', '18 Grouped Index', 'Group X'+ $
      ' Position', 'Group Y Position', 'Group Sigma X Pos', 'Group'+ $
      ' Sigma Y Pos', 'Group N Photons', '24 Group Size', 'Frame'+ $
      ' Index in Grp', 'Label Set' ])

  
  WID_DROPLIST_RawFileName = Widget_Droplist(WID_BASE_0_PeakSelector,  $
      UNAME='WID_DROPLIST_RawFileName' ,XOFFSET=11 ,YOFFSET=872  $
      ,SCR_XSIZE=343 ,SCR_YSIZE=26 ,TITLE='Raw Data File')

  
  WID_BUTTON_Back1Step = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Back1Step' ,XOFFSET=399 ,YOFFSET=671  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Back 1'+ $
      ' Step')

  
  WID_BUTTON_UnZoom_2X = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_UnZoom_2X' ,XOFFSET=400 ,YOFFSET=634  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='UnZoom 2X')

  
  WID_BUTTON_Plot_XgrYgrZgr = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Plot_XgrYgrZgr' ,XOFFSET=20 ,YOFFSET=812  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=25 ,/ALIGN_CENTER  $
      ,VALUE='Xgr,Ygr,Zgr')

  
  WID_BUTTON_Plot_XgrZgrYgr = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Plot_XgrZgrYgr' ,XOFFSET=105 ,YOFFSET=812  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=25 ,/ALIGN_CENTER  $
      ,VALUE='Xgr,Zgr,Ygr')

  
  WID_BUTTON_Plot_YgrZgrXgr = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Plot_YgrZgrXgr' ,XOFFSET=190 ,YOFFSET=812  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=25 ,/ALIGN_CENTER  $
      ,VALUE='Ygr,Zgr,Xgr')

  
  WID_BUTTON_Plot_FrameZX = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Plot_FrameZX' ,XOFFSET=285 ,YOFFSET=812  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Frame, Z,'+ $
      ' X')

  
  WID_BUTTON_Stat3DViewer = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Stat3DViewer' ,XOFFSET=310 ,YOFFSET=742  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Static 3D'+ $
      ' Viewer')

  
  WID_LABEL_0 = Widget_Text(WID_BASE_0_PeakSelector,  $
      UNAME='WID_LABEL_0' ,FRAME=1 ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=540 ,SCR_YSIZE=45 ,/WRAP ,VALUE=[ ' Select Attribute'+ $
      ' Row --> Histogram' ] ,XSIZE=20 ,YSIZE=3)

  
  WID_TABLE_StartReadSkip = Widget_Table(WID_BASE_0_PeakSelector,  $
      UNAME='WID_TABLE_StartReadSkip' ,XOFFSET=175 ,YOFFSET=667  $
      ,SCR_XSIZE=211 ,SCR_YSIZE=70 ,/EDITABLE ,/COLUMN_MAJOR  $
      ,COLUMN_LABELS=[ 'Start', 'Read', 'Skip' ] ,XSIZE=3 ,YSIZE=1)

  
  WID_BUTTON_Reload_Paramlimits =  $
      Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Reload_Paramlimits' ,XOFFSET=160 ,YOFFSET=742  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Reload'+ $
      ' Limits Table')

  
  WID_BASE_0 = Widget_Base(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BASE_0' ,XOFFSET=395 ,YOFFSET=705 ,SCR_XSIZE=150  $
      ,SCR_YSIZE=22 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_StartReadSkip = Widget_Button(WID_BASE_0,  $
      UNAME='WID_BUTTON_StartReadSkip' ,SCR_XSIZE=148 ,SCR_YSIZE=22  $
      ,/ALIGN_LEFT ,VALUE='Apply Read/Skip Filter')

  
  WID_BUTTON_Plot_XgrUnwZgrYgr =  $
      Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Plot_XgrUnwZgrYgr' ,XOFFSET=85 ,YOFFSET=842  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=25 ,/ALIGN_CENTER  $
      ,VALUE='Xgr,UnwZgr,Ygr')

  
  WID_BUTTON_Plot_YgrUnwZgrXgr =  $
      Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Plot_YgrUnwZgrXgr' ,XOFFSET=190 ,YOFFSET=842  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=25 ,/ALIGN_CENTER  $
      ,VALUE='Ygr,UnwZgr,Xgr')

  
  WID_SLIDER_FractionHistAnal =  $
      Widget_Slider(WID_BASE_0_PeakSelector,  $
      UNAME='WID_SLIDER_FractionHistAnal' ,XOFFSET=10 ,YOFFSET=692  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=46 ,TITLE='Fraction (Hist. Analysis)'  $
      ,MAXIMUM=100 ,VALUE=85)

  
  WID_BUTTON_Plot_XgrUnwZgrLbl =  $
      Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Plot_XgrUnwZgrLbl' ,XOFFSET=300 ,YOFFSET=842  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=25 ,/ALIGN_CENTER  $
      ,VALUE='Xgr,UZgr,Lbl')

  
  WID_BUTTON_FilterLabel = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_FilterLabel' ,XOFFSET=274 ,YOFFSET=966  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Add Filter'+ $
      ' Lbl.')

  
  WID_BUTTON_AddAllLabels = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_AddAllLabels' ,XOFFSET=300 ,YOFFSET=934  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Add All')

  
  WID_BUTTON_Plot_XZLbl = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Plot_XZLbl' ,XOFFSET=375 ,YOFFSET=812  $
      ,SCR_XSIZE=65 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='X,Z,Lbl')

  
  WID_BUTTON_Overlay_All_Centers =  $
      Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Overlay_All_Centers' ,XOFFSET=250  $
      ,YOFFSET=997 ,SCR_XSIZE=125 ,SCR_YSIZE=25 ,/ALIGN_CENTER  $
      ,VALUE='Overlay All Centers')

  
  WID_BUTTON_Overlay_Frame_Centers =  $
      Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Overlay_Frame_Centers' ,XOFFSET=120  $
      ,YOFFSET=997 ,SCR_XSIZE=125 ,SCR_YSIZE=25 ,/ALIGN_CENTER  $
      ,VALUE='Overlay Fr. Centers')

  
  WID_BUTTON_Overlay_DIC = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Overlay_DIC' ,XOFFSET=15 ,YOFFSET=662  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Overlay'+ $
      ' DIC/EM')

  
  WID_BUTTON_CustomTIFF = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_CustomTIFF' ,XOFFSET=450 ,YOFFSET=742  $
      ,SCR_XSIZE=110 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Cust.'+ $
      ' TIFF')

  
  WID_BUTTON_Plot_FrameUnwrZX =  $
      Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Plot_FrameUnwrZX' ,XOFFSET=20 ,YOFFSET=842  $
      ,SCR_XSIZE=60 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Fr, UnwZ')

  
  WID_BASE_Redraw = Widget_Base(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BASE_Redraw' ,XOFFSET=16 ,YOFFSET=42 ,TITLE='IDL'  $
      ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Redraw = Widget_Button(WID_BASE_Redraw,  $
      UNAME='WID_BUTTON_Redraw' ,/ALIGN_LEFT ,VALUE='Redraw')

  
  WID_BUTTON_Plot_spectrum = Widget_Button(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BUTTON_Plot_spectrum' ,XOFFSET=10 ,YOFFSET=997  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Plot'+ $
      ' Spectra')

  
  WID_BASE_Redraw_0 = Widget_Base(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BASE_Redraw_0' ,XOFFSET=160 ,YOFFSET=42 ,TITLE='IDL'  $
      ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Hist_Log_X = Widget_Button(WID_BASE_Redraw_0,  $
      UNAME='WID_BUTTON_Hist_Log_X' ,/ALIGN_LEFT ,VALUE='Hist. X'+ $
      ' Log')

  
  WID_BASE_Redraw_1 = Widget_Base(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BASE_Redraw_1' ,XOFFSET=280 ,YOFFSET=42 ,TITLE='IDL'  $
      ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Hist_Log_Y = Widget_Button(WID_BASE_Redraw_1,  $
      UNAME='WID_BUTTON_Hist_Log_Y' ,/ALIGN_LEFT ,VALUE='Hist. Y'+ $
      ' Log')

  
  WID_BASE_Redraw_2 = Widget_Base(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BASE_Redraw_2' ,XOFFSET=430 ,YOFFSET=42 ,TITLE='IDL'  $
      ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Allow_Bridge = Widget_Button(WID_BASE_Redraw_2,  $
      UNAME='WID_BUTTON_Allow_Bridge' ,/ALIGN_LEFT ,VALUE='Allow'+ $
      ' Bridge')

  
  WID_BASE_Tie_RGB = Widget_Base(WID_BASE_0_PeakSelector,  $
      UNAME='WID_BASE_Tie_RGB' ,XOFFSET=415 ,YOFFSET=1005  $
      ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Tie_RGB = Widget_Button(WID_BASE_Tie_RGB,  $
      UNAME='WID_BUTTON_Tie_RGB' ,/ALIGN_LEFT ,VALUE='Tie RGB')

  Widget_Control, /REALIZE, WID_BASE_0_PeakSelector

  XManager, 'WID_BASE_0_PeakSelector', WID_BASE_0_PeakSelector, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro PeakSelector, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_0_PeakSelector, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
