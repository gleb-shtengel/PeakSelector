HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	11/15/2017 22:23.35
VERSION 1
END

WID_BASE_0_PeakSelector BASE 5 1 1622 1050
REALIZE "Initialization_PeakSelector_Main"
SCROLLWIDTH = 1622
SCROLLHEIGHT = 1070
TLB
CAPTION "Peak Selector v9.6© 2017, Howard Hughes Medical Institute.  All rights reserved."
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_TABLE_0 TABLE 4 65 550 490
  FRAME = 1
  SCROLL
  RESIZECOLUMNS
  N_ROWS = 50
  N_COLS = 5
  NUMCOLLABELS = 5
  COLLABEL "Lower Bound"
  COLLABEL "Upper Bound"
  COLLABEL "Center"
  COLLABEL "Range"
  COLLABEL "Peak"
  NUMROWLABELS = 27
  ROWLABEL "Offset"
  ROWLABEL "Amplitude"
  ROWLABEL "X Position"
  ROWLABEL "Y Position"
  ROWLABEL "X Peak Width"
  ROWLABEL "Y Peak Width"
  ROWLABEL "6 N Photons"
  ROWLABEL "ChiSquared"
  ROWLABEL "FitOK"
  ROWLABEL "Frame Number"
  ROWLABEL "Peak Index of Frame"
  ROWLABEL "Peak Global Index"
  ROWLABEL "12 Sigma Offset"
  ROWLABEL "Sigma Amplitude"
  ROWLABEL "Sigma X Pos rtNph"
  ROWLABEL "Sigma Y Pos rtNph"
  ROWLABEL "Sigma X Pos Full"
  ROWLABEL "Sigma Y Pos Full"
  ROWLABEL "18 Grouped Index"
  ROWLABEL "Group X Position"
  ROWLABEL "Group Y Position"
  ROWLABEL "Group Sigma X Pos"
  ROWLABEL "Group Sigma Y Pos"
  ROWLABEL "Group N Photons"
  ROWLABEL "24 Group Size"
  ROWLABEL "Frame Index in Grp"
  ROWLABEL "Label Set"
  EDITABLE
  ONINSERTCHAR "InsertChar"
  END
  WID_BUTTON_Peak_Centers PUSHBUTTON 15 605 130 25
  VALUE "Peak Centers"
  ALIGNCENTER
  ONACTIVATE "OnPeakCentersButton"
  END
  WID_DRAW_0 DRAW 567 0 1024 1024
  FRAME = 1
  REALIZE "OnDraw0Realize"
  ONBUTTON "OnButtonDraw0"
  END
  WID_BASE_0_PeakSelector_MBAR MENUBAR 0 0 0 0
  BEGIN
    W_MENU_4 PUSHBUTTON 0 0 0 0
    VALUE "File"
    MENU
    BEGIN
      W_MENU_32 PUSHBUTTON 0 0 0 0
      VALUE "Extract Peaks .sif/ .tif/ .txt/.dat to .pks"
      ONACTIVATE "OnExtractPeaks"
      END
      W_MENU_Extract_Peaks_multiple_TIFFs PUSHBUTTON 0 0 0 0
      VALUE "Extract Peaks: Multiple TIFF files"
      ONACTIVATE "OnExtractPeaks_Multiple_TIFFs"
      END
      W_MENU_iPALM_Macro PUSHBUTTON 0 0 0 0
      VALUE "iPALM Macro: Transform Extract Reextract Group GetZ"
      ONACTIVATE "iPALM_Macro"
      END
      W_MENU_53 PUSHBUTTON 0 0 0 0
      VALUE "Extract Peaks Multiple Labels"
      ONACTIVATE "OnExtractPeaksML"
      END
      W_MENU_40 PUSHBUTTON 0 0 0 0
      VALUE "ReExtract Peaks Multi-Label"
      ONACTIVATE "ReExtractPeaksMultiLabel"
      END
      W_MENU_5 PUSHBUTTON 0 0 0 0
      VALUE "Open .pks File"
      ONACTIVATE "OpenTheFile"
      END
      W_MENU_1 PUSHBUTTON 0 0 0 0
      VALUE "Add .pks File"
      ONACTIVATE "AddtotheFile"
      END
      W_MENU_Save_Data_ASCII_txt PUSHBUTTON 0 0 0 0
      VALUE "Export Processed Data as ASCII (.txt)"
      ONACTIVATE "On_Save_Data_ASCII_txt"
      END
      W_MENU_2 PUSHBUTTON 0 0 0 0
      VALUE "Save Proccessed as IDL(.sav)"
      ONACTIVATE "SavetheCommon"
      END
      W_MENU_8 PUSHBUTTON 0 0 0 0
      VALUE "Recall Proccessed IDL(.sav)"
      ONACTIVATE "RecalltheCommon"
      END
      W_MENU_39 PUSHBUTTON 0 0 0 0
      VALUE "Add Next Label IDL(.sav)"
      ONACTIVATE "AddNextLabelData"
      END
      W_MENU_25 PUSHBUTTON 0 0 0 0
      VALUE "Merge Labels Consecutively"
      ONACTIVATE "MergeLabelsConsecutively"
      END
      W_MENU_Load_DIC PUSHBUTTON 0 0 0 0
      VALUE "Load DIC / EM"
      ONACTIVATE "LoadDIC"
      END
      W_MENU_42 PUSHBUTTON 0 0 0 0
      VALUE "Purge Peaks"
      MENU
      BEGIN
        W_MENU_50 PUSHBUTTON 0 0 0 0
        VALUE "Not OK Peaks"
        ONACTIVATE "OnPurgeButton"
        END
        W_MENU_reassign_peak_indecis PUSHBUTTON 0 0 0 0
        VALUE "Reassign Peak Indecis within Each Frame"
        ONACTIVATE "Reassign_Peak_Indecis_within_Frames"
        END
        W_MENU_51 PUSHBUTTON 0 0 0 0
        VALUE "Filtered Peaks"
        ONACTIVATE "Purge_Filtered"
        END
        W_MENU_52 PUSHBUTTON 0 0 0 0
        VALUE "Group Filtered Peaks"
        ONACTIVATE "Purge_Group_Filtered"
        END
        W_MENU_Purge_current PUSHBUTTON 0 0 0 0
        VALUE "Purge with current filter"
        ONACTIVATE "Purge_current_filter"
        END
        W_MENU_Purge_by_XY_coords PUSHBUTTON 0 0 0 0
        VALUE "Filtered by X-Y Coordinates"
        ONACTIVATE "Purge_by_XY_coords"
        END
      END
      W_MENU_15 PUSHBUTTON 0 0 0 0
      VALUE "Save Image"
      MENU
      BEGIN
        W_MENU_0 PUSHBUTTON 0 0 0 0
        VALUE "Save Image .jpeg"
        ONACTIVATE "SaveImageJPEG"
        END
        W_MENU_22 PUSHBUTTON 0 0 0 0
        VALUE "Save Image .bmp"
        ONACTIVATE "SaveImageBMP"
        END
        W_MENU_58 PUSHBUTTON 0 0 0 0
        VALUE "Save Image .tiff"
        ONACTIVATE "SaveImageTIFF"
        END
        W_MENU_23 PUSHBUTTON 0 0 0 0
        VALUE "Save Image IDL Data (.sav)"
        ONACTIVATE "SaveImageIDLData"
        END
      END
      W_MENU_63 PUSHBUTTON 0 0 0 0
      VALUE "Export Histogram ASCII"
      ONACTIVATE "Export_Hist_ASCII"
      END
      W_MENU_55 PUSHBUTTON 0 0 0 0
      VALUE "Print Window"
      ONACTIVATE "PrintImage"
      END
      W_MENU_ImportASCII PUSHBUTTON 0 0 0 0
      VALUE "Import ASCII"
      MENU
      BEGIN
        W_MENU_ImportUserASCII PUSHBUTTON 0 0 0 0
        VALUE "Import User ASCII"
        ONACTIVATE "ImportUserASCII"
        END
        W_MENU_ImportZeissTXT PUSHBUTTON 0 0 0 0
        VALUE "Import Zeiss TXT"
        ONACTIVATE "ImportZeissTXt"
        END
      END
      W_MENU_3 PUSHBUTTON 0 0 0 0
      VALUE "StopPeakSelect"
      ONACTIVATE "StopthePeakSelect"
      END
      W_MENU_Edit_Preferences PUSHBUTTON 0 0 0 0
      VALUE "Edit Preferences"
      ONACTIVATE "EditPeakSelectorPreferences"
      END
      W_MENU_6 PUSHBUTTON 0 0 0 0
      VALUE "Exit"
      ONACTIVATE "ExittheFile"
      END
    END
    W_MENU_41 PUSHBUTTON 0 0 0 0
    VALUE "Image Transformations"
    MENU
    BEGIN
      W_MENU_26 PUSHBUTTON 0 0 0 0
      VALUE "Orientation"
      MENU
      BEGIN
        W_MENU_27 PUSHBUTTON 0 0 0 0
        VALUE "Transpose"
        ONACTIVATE "OnTranspose"
        END
        W_MENU_28 PUSHBUTTON 0 0 0 0
        VALUE "Flip Horizontal"
        ONACTIVATE "OnFlipHorizontal"
        END
        W_MENU_29 PUSHBUTTON 0 0 0 0
        VALUE "Flip Vertical"
        ONACTIVATE "OnFlipVertical"
        END
        W_MENU_30 PUSHBUTTON 0 0 0 0
        VALUE "Rotate 90 CW"
        ONACTIVATE "OnRotate90CW"
        END
        W_MENU_31 PUSHBUTTON 0 0 0 0
        VALUE "Rotate 90 CCW"
        ONACTIVATE "OnRotate90CCW"
        END
      END
      W_MENU_14 PUSHBUTTON 0 0 0 0
      VALUE "Rotate"
      ONACTIVATE "OnTwist"
      END
      W_MENU_47 PUSHBUTTON 0 0 0 0
      VALUE "Test/Write Guide Star"
      ONACTIVATE "TestWriteGuideStar"
      END
      W_MENU_XY_Interp PUSHBUTTON 0 0 0 0
      VALUE "Interpolate / Subtract XY trend"
      ONACTIVATE "XY_Interpolation"
      END
      W_MENU_36 PUSHBUTTON 0 0 0 0
      VALUE "Anchor Fiducial Pnts"
      ONACTIVATE "OnAnchorFiducialMenu"
      END
      W_MENU_34 PUSHBUTTON 0 0 0 0
      VALUE "Save TransformCoeffs (.sav)"
      ONACTIVATE "SaveTransformCoeffs"
      END
      W_MENU_38 PUSHBUTTON 0 0 0 0
      VALUE "Transform Raw, Save and Save Sum (.dat)"
      ONACTIVATE "TransformRaw_Save_SaveSum_MenuItem"
      END
      W_MENU_35 PUSHBUTTON 0 0 0 0
      VALUE "Transform Raw and Save User-Defined (.dat)"
      ONACTIVATE "TransformRaw_Save_UserDefined"
      END
      W_MENU_Transform_SpAnalysis PUSHBUTTON 0 0 0 0
      VALUE "Transform Spectral Data and Analyze Spectra"
      ONACTIVATE "Transform_SpAnalysis"
      END
      W_MENU_Convert_Pixels_to_NM PUSHBUTTON 0 0 0 0
      VALUE "Convert X-Y data to Wavelength-Y data"
      ONACTIVATE "Convert_X_to_wavelength"
      END
    END
    W_MENU_11 PUSHBUTTON 0 0 0 0
    VALUE "Special Functions"
    MENU
    BEGIN
      W_MENU_46 PUSHBUTTON 0 0 0 0
      VALUE "Group Peaks"
      ONACTIVATE "OnGroupPeaks"
      END
      W_MENU_44 PUSHBUTTON 0 0 0 0
      VALUE " Z-coordinate Operations"
      ONACTIVATE "ZCoordinateOperations"
      END
      W_MENU_Zastig PUSHBUTTON 0 0 0 0
      VALUE "Z-coordinate Operations - Astig Only "
      ONACTIVATE "ZCoordinateOperations_Astig"
      END
      W_MENU_ProcessMultiplePalmSlabs PUSHBUTTON 0 0 0 0
      VALUE "Process Multiple PALM Slabs"
      ONACTIVATE "Process_Multiple_Palm_Slabs_call"
      END
      W_Polarization_Analysis PUSHBUTTON 0 0 0 0
      VALUE "Polarization Analysis"
      ONACTIVATE "Polarization_Analysis"
      END
      W_MENU_64 PUSHBUTTON 0 0 0 0
      VALUE "Recalculate CGroupParams[12,*]"
      ONACTIVATE "Recalculate_XpkwYpkw"
      END
      W_MENU_Reprocess_Palm_Set PUSHBUTTON 0 0 0 0
      VALUE "Reprocess Palm Set"
      ONACTIVATE "Reprocess_Palm_Set"
      END
      W_MENU_ApplyFilterSelectively PUSHBUTTON 0 0 0 0
      VALUE "Apply Filter Only to Selected Label"
      CHECKED_MENU
      ONACTIVATE "On_Change_Filter_Select"
      END
      W_MENU_19 PUSHBUTTON 0 0 0 0
      VALUE "Autocorrelate"
      MENU
      BEGIN
        W_MENU_7 PUSHBUTTON 0 0 0 0
        VALUE "Auto Correlate Peaks"
        ONACTIVATE "OnAutoCorrelPeak"
        END
        W_MENU_12 PUSHBUTTON 0 0 0 0
        VALUE "Auto Correlate Group Start"
        ONACTIVATE "OnAutoCorrelateGroups"
        END
      END
      W_MENU_16 PUSHBUTTON 0 0 0 0
      VALUE "Smooth 3"
      ONACTIVATE "OnSmooth3"
      END
      W_MENU_21 PUSHBUTTON 0 0 0 0
      VALUE "Smooth 5"
      ONACTIVATE "OnSmooth5"
      END
      W_MENU_24 PUSHBUTTON 0 0 0 0
      VALUE "Smooth 9"
      ONACTIVATE "OnSmooth9"
      END
      W_MENU_17 PUSHBUTTON 0 0 0 0
      VALUE "Analyze1 Intensity Triplets"
      ONACTIVATE "Analyze1"
      END
      W_MENU_SwapXZ PUSHBUTTON 0 0 0 0
      VALUE "Swap X-Z"
      CHECKED_MENU
      ONACTIVATE "OnSwapXZ"
      END
      W_MENU_SwapYZ PUSHBUTTON 0 0 0 0
      VALUE "Swap Y-Z"
      CHECKED_MENU
      ONACTIVATE "OnSwapYZ"
      END
      W_MENU_65 PUSHBUTTON 0 0 0 0
      VALUE "Swap Z with Unwrapped Z"
      CHECKED_MENU
      ONACTIVATE "Swap_Z_Unwrapped_Z"
      END
      W_MENU_18 PUSHBUTTON 0 0 0 0
      VALUE "Analyze2 Peak Distribution Statistics"
      ONACTIVATE "OnAnalyzePeakDistribution"
      END
      W_MENU_GroupDistribution PUSHBUTTON 0 0 0 0
      VALUE "Analyze2 Group Distribution Statistics"
      ONACTIVATE "OnAnalyzeGroupDistribution"
      END
      W_MENU_20 PUSHBUTTON 0 0 0 0
      VALUE "Analyze3 Peak Statistics for Multiple Areas"
      ONACTIVATE "OnAnalyze3"
      END
      W_menu_Analyze_Fiducial_Alignement_2Colors PUSHBUTTON 0 0 0 0
      VALUE "Analyze Fiducial Alignement for 2Colors"
      ONACTIVATE "On_Analyze_Fiducial_Alignement_2Colors"
      END
      W_MENU_62 PUSHBUTTON 0 0 0 0
      VALUE "Analyze4 Show iPALM PSF Raw and Fits"
      ONACTIVATE "OnAnalyze4"
      END
      W_MENU_AnalyzePhaseUnwrap PUSHBUTTON 0 0 0 0
      VALUE "Analyze Phase Unwrap and Localizations"
      ONACTIVATE "On_AnalyzePhaseUnwrap"
      END
      W_MENU_process_SRM PUSHBUTTON 0 0 0 0
      VALUE "Process SRM"
      ONACTIVATE "Process_SRM"
      END
      W_MENU_Renumber_GP PUSHBUTTON 0 0 0 0
      VALUE "Re-number Group Peaks"
      ONACTIVATE "Renumber_Group_Peaks"
      END
      W_MENU_Correct_GroupSigmaXYZ PUSHBUTTON 0 0 0 0
      VALUE "Correct GroupSigmaXYZ (*1.414)"
      ONACTIVATE "Correct_GroupSigmaXYZ"
      END
      W_MENU_Calculate_MSD PUSHBUTTON 0 0 0 0
      VALUE "Calculate Mean Squared Displacement"
      ONACTIVATE "Calculate_MSD"
      END
    END
    W_MENU_9 PUSHBUTTON 0 0 0 0
    VALUE "Display"
    MENU
    BEGIN
      W_MENU_10 PUSHBUTTON 0 0 0 0
      VALUE "Color Table"
      ONACTIVATE "ColorTheTable"
      END
      W_MENU_54 PUSHBUTTON 0 0 0 0
      VALUE "Z using Color Table -> Hue"
      CHECKED_MENU
      ONACTIVATE "DoToggleZtoHue"
      END
      W_MENU_59 PUSHBUTTON 0 0 0 0
      VALUE "Render Without Legends"
      CHECKED_MENU
      ONACTIVATE "ToggleLegends"
      END
      W_MENU_MoveColorBar PUSHBUTTON 0 0 0 0
      VALUE "Move Color Bar to the Top"
      CHECKED_MENU
      ONACTIVATE "OnColorBar_Move"
      END
      W_MENU_ColorBarExtend PUSHBUTTON 0 0 0 0
      VALUE "Extend Color Bar"
      CHECKED_MENU
      ONACTIVATE "OnColorBar_Extend"
      END
      W_MENU_XYZ_diamonds PUSHBUTTON 0 0 0 0
      VALUE "XYZ Plot: Use Diamonds"
      CHECKED_MENU
      ONACTIVATE "On_XYZ_use_diamonds"
      END
      W_MENU_Set_nm_per_pixel_scale PUSHBUTTON 0 0 0 0
      VALUE "Set nm/pixel Scale"
      ONACTIVATE "Set_nm_per_pixel_scale"
      END
      W_MENU_DisplayThisFitCond PUSHBUTTON 0 0 0 0
      VALUE "Display ThisFitCond"
      ONACTIVATE "DisplayThisFitCond"
      END
      W_MENU_66W_MENU_set_Z_scale_multiplier PUSHBUTTON 0 0 0 0
      VALUE "Set Z scale multiplier (X-Y swaps)"
      ONACTIVATE "Set_Z_scale_multiplier"
      END
      W_vbar_top PUSHBUTTON 0 0 0 0
      VALUE "Set Z Color Bar Scale"
      ONACTIVATE "Set_vbar_top"
      END
      W_MENU_SetMaxProbability_2DPALM PUSHBUTTON 0 0 0 0
      VALUE "Set Max Probability for 2D PALM images"
      ONACTIVATE "SetMaxProbability_2DPALM"
      END
      W_MENU_Force_MaxProb_2DPALM PUSHBUTTON 0 0 0 0
      VALUE "Force Max Probability for 2D PALM images"
      CHECKED_MENU
      ONACTIVATE "OnForce_MaxProbability_2DPALM"
      END
      W_MENU_IgnoreLblsHist PUSHBUTTON 0 0 0 0
      VALUE "Ignore Labels For Histograms"
      CHECKED_MENU
      ONACTIVATE "OnIgnoreLblsHist"
      END
      W_MENU_Save_Histograms_BMP PUSHBUTTON 0 0 0 0
      VALUE "Save Histograms as BMP"
      ONACTIVATE "OnSaveHistBMP"
      END
      W_MENU_Replace_TotalRaw_with_Rendered PUSHBUTTON 0 0 0 0
      VALUE "Replace TotalRaw with Rendered Image"
      ONACTIVATE "Replace_TotalRaw_with_Rendered"
      END
    END
  END
  WID_BUTTON_Group_Centers PUSHBUTTON 15 577 130 25
  VALUE "Group Centers"
  ALIGNCENTER
  ONACTIVATE "OnGroupCentersButton"
  END
  WID_BUTTON_UnZoom PUSHBUTTON 400 597 140 30
  VALUE "UnZoomed Centers"
  ALIGNCENTER
  ONACTIVATE "OnUnZoomButton"
  END
  WID_LABEL_NumberSelected LABEL 8 559 258 12
  VALUE "Number of Selected Peaks"
  ALIGNLEFT
  END
  WID_BUTTON_TotalRawData PUSHBUTTON 15 633 130 25
  VALUE "Total Raw Data"
  ALIGNCENTER
  ONACTIVATE "OnTotalRawDataButton"
  END
  WID_BUTTON_PloyXY PUSHBUTTON 10 742 140 30
  VALUE "Plot X vs Y - color  Z"
  ALIGNCENTER
  ONACTIVATE "OnPlotXYButton"
  END
  WID_DROPLIST_X DROPLIST 5 780 170 23
  CAPTION "X Axis"
  NUMITEMS = 27
  ITEM "Offset"
  ITEM "Amplitude"
  ITEM "X Position"
  ITEM "Y Position"
  ITEM "X Peak Width"
  ITEM "Y Peak Width"
  ITEM "6 N Photons"
  ITEM "ChiSquared"
  ITEM "FitOK"
  ITEM "Frame Number"
  ITEM "Peak Index of Frame"
  ITEM "Peak Global Index"
  ITEM "12 Sigma Offset"
  ITEM "Sigma Amplitude"
  ITEM "Sigma X Pos rtNph"
  ITEM "Sigma Y Pos rtNph"
  ITEM "Sigma X Pos Full"
  ITEM "Sigma Y Pos Full"
  ITEM "18 Grouped Index"
  ITEM "Group X Position"
  ITEM "Group Y Position"
  ITEM "Group Sigma X Pos"
  ITEM "Group Sigma Y Pos"
  ITEM "Group N Photons"
  ITEM "24 Group Size"
  ITEM "Frame Index in Grp"
  ITEM "Label Set"
  END
  WID_SLIDER_Bot SLIDER 385 957 160 46
  CAPTION "Stretch Bottom"
  MAXIMUM = 1000
  ONVALUECHANGED "OnStretchBottom"
  END
  WID_SLIDER_Top SLIDER 385 847 160 46
  CAPTION "Stretch Top"
  VALUE = 500
  MAXIMUM = 1000
  ONVALUECHANGED "OnStretchTop"
  END
  WID_SLIDER_Gamma SLIDER 385 902 160 46
  CAPTION "Gamma"
  VALUE = 500
  MAXIMUM = 1000
  ONVALUECHANGED "OnGamma"
  END
  WID_DROPLIST_Accumulate DROPLIST 180 632 170 25
  CAPTION "Accumulation"
  NUMITEMS = 2
  ITEM "Envelope"
  ITEM "Sum"
  END
  WID_DROPLIST_Function DROPLIST 180 572 170 25
  CAPTION "Function"
  NUMITEMS = 3
  ITEM "Center Locations"
  ITEM "Gaussian Normalized"
  ITEM "Gaussian Amplitude"
  END
  WID_DROPLIST_Filter DROPLIST 180 602 170 25
  CAPTION "Filter"
  NUMITEMS = 2
  ITEM "Frame Peaks"
  ITEM "Grouped Peaks"
  END
  WID_BUTTON_Render PUSHBUTTON 400 560 140 30
  VALUE "Render"
  ALIGNCENTER
  ONACTIVATE "OnRenderButton"
  END
  WID_BUTTON_ScaleBar PUSHBUTTON 196 902 95 25
  VALUE "Add Scale Bar1"
  ALIGNCENTER
  ONACTIVATE "OnAddScaleBarButton"
  END
  WID_BUTTON_ColorBar PUSHBUTTON 176 934 120 25
  VALUE "Add Color Bar"
  ALIGNCENTER
  ONACTIVATE "OnAddColorBarButton"
  END
  WID_BUTTON_ImageLabel PUSHBUTTON 175 966 95 25
  VALUE "Add Image Lbl."
  ALIGNCENTER
  ONACTIVATE "OnAddLabelButton"
  END
  WID_SLIDER_RawFrameNumber SLIDER 21 898 150 46
  CAPTION "Raw Frame Number"
  ONVALUECHANGED "OnRawFrameNumber"
  END
  WID_SLIDER_RawPeakIndex SLIDER 20 948 150 46
  CAPTION "Raw Peak Index of Frame"
  ONVALUECHANGED "OnRawPeakIndex"
  END
  WID_BUTTON_ScaleBar2 PUSHBUTTON 301 902 35 25
  VALUE "Bar2"
  ALIGNCENTER
  ONACTIVATE "OnAddScaleBarButton2"
  END
  WID_DROPLIST_Label DROPLIST 450 812 90 23
  CAPTION "Label"
  NUMITEMS = 5
  ITEM ""
  ITEM "Red"
  ITEM "Green"
  ITEM "Blue"
  ITEM "DIC / EM"
  ONSELECT "OnLabelDropList"
  END
  WID_DROPLIST_Y DROPLIST 190 780 170 23
  CAPTION "Y Axis"
  NUMITEMS = 27
  ITEM "Offset"
  ITEM "Amplitude"
  ITEM "X Position"
  ITEM "Y Position"
  ITEM "X Peak Width"
  ITEM "Y Peak Width"
  ITEM "6 N Photons"
  ITEM "ChiSquared"
  ITEM "FitOK"
  ITEM "Frame Number"
  ITEM "Peak Index of Frame"
  ITEM "Peak Global Index"
  ITEM "12 Sigma Offset"
  ITEM "Sigma Amplitude"
  ITEM "Sigma X Pos rtNph"
  ITEM "Sigma Y Pos rtNph"
  ITEM "Sigma X Pos Full"
  ITEM "Sigma Y Pos Full"
  ITEM "18 Grouped Index"
  ITEM "Group X Position"
  ITEM "Group Y Position"
  ITEM "Group Sigma X Pos"
  ITEM "Group Sigma Y Pos"
  ITEM "Group N Photons"
  ITEM "24 Group Size"
  ITEM "Frame Index in Grp"
  ITEM "Label Set"
  END
  WID_DROPLIST_Z DROPLIST 375 780 170 23
  CAPTION "Color"
  NUMITEMS = 27
  ITEM "Offset"
  ITEM "Amplitude"
  ITEM "X Position"
  ITEM "Y Position"
  ITEM "X Peak Width"
  ITEM "Y Peak Width"
  ITEM "6 N Photons"
  ITEM "ChiSquared"
  ITEM "FitOK"
  ITEM "Frame Number"
  ITEM "Peak Index of Frame"
  ITEM "Peak Global Index"
  ITEM "12 Sigma Offset"
  ITEM "Sigma Amplitude"
  ITEM "Sigma X Pos rtNph"
  ITEM "Sigma Y Pos rtNph"
  ITEM "Sigma X Pos Full"
  ITEM "Sigma Y Pos Full"
  ITEM "18 Grouped Index"
  ITEM "Group X Position"
  ITEM "Group Y Position"
  ITEM "Group Sigma X Pos"
  ITEM "Group Sigma Y Pos"
  ITEM "Group N Photons"
  ITEM "24 Group Size"
  ITEM "Frame Index in Grp"
  ITEM "Label Set"
  END
  WID_DROPLIST_RawFileName DROPLIST 11 872 343 26
  CAPTION "Raw Data File"
  ONSELECT "SetRawSliders"
  END
  WID_BUTTON_Back1Step PUSHBUTTON 399 671 140 30
  VALUE "Back 1 Step"
  ALIGNCENTER
  ONACTIVATE "On1StepBack"
  END
  WID_BUTTON_UnZoom_2X PUSHBUTTON 400 634 140 30
  VALUE "UnZoom 2X"
  ALIGNCENTER
  ONACTIVATE "OnUnZoom2X"
  END
  WID_BUTTON_Plot_XgrYgrZgr PUSHBUTTON 20 812 80 25
  VALUE "Xgr,Ygr,Zgr"
  ALIGNCENTER
  ONACTIVATE "OnPlotXgrYgrZgr"
  END
  WID_BUTTON_Plot_XgrZgrYgr PUSHBUTTON 105 812 80 25
  VALUE "Xgr,Zgr,Ygr"
  ALIGNCENTER
  ONACTIVATE "OnPlotXgrZgrXgr"
  END
  WID_BUTTON_Plot_YgrZgrXgr PUSHBUTTON 190 812 80 25
  VALUE "Ygr,Zgr,Xgr"
  ALIGNCENTER
  ONACTIVATE "OnPlotYgrZgrXgr"
  END
  WID_BUTTON_Plot_FrameZX PUSHBUTTON 285 812 80 25
  VALUE "Frame, Z, X"
  ALIGNCENTER
  ONACTIVATE "OnPlotFrameZX"
  END
  WID_BUTTON_Stat3DViewer PUSHBUTTON 310 742 130 30
  VALUE "Static 3D Viewer"
  ALIGNCENTER
  ONACTIVATE "OnStatic3DViewer"
  END
  WID_LABEL_0 TEXT 5 5 540 45
  FRAME = 1
  NUMITEMS = 1
  ITEM " Select Attribute Row --> Histogram"
  WRAP
  WIDTH = 20
  HEIGHT = 3
  END
  WID_TABLE_StartReadSkip TABLE 175 667 211 70
  COLUMNMAJOR
  N_ROWS = 1
  N_COLS = 3
  NUMCOLLABELS = 3
  COLLABEL "Start"
  COLLABEL "Read"
  COLLABEL "Skip"
  EDITABLE
  END
  WID_BUTTON_Reload_Paramlimits PUSHBUTTON 160 742 140 30
  VALUE "Reload Limits Table"
  ALIGNCENTER
  ONACTIVATE "ReloadParamlists"
  END
  WID_BASE_0 BASE 395 705 150 22
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_StartReadSkip PUSHBUTTON -1 -1 148 22
    VALUE "Apply Read/Skip Filter"
    ALIGNLEFT
    END
  END
  WID_BUTTON_Plot_XgrUnwZgrYgr PUSHBUTTON 85 842 100 25
  VALUE "Xgr,UnwZgr,Ygr"
  ALIGNCENTER
  ONACTIVATE "OnPlotXgrUnwZgrXgr"
  END
  WID_BUTTON_Plot_YgrUnwZgrXgr PUSHBUTTON 190 842 100 25
  VALUE "Ygr,UnwZgr,Xgr"
  ALIGNCENTER
  ONACTIVATE "OnPlotYgrUnwZgrXgr"
  END
  WID_SLIDER_FractionHistAnal SLIDER 10 692 140 46
  CAPTION "Fraction (Hist. Analysis)"
  VALUE = 85
  MAXIMUM = 100
  END
  WID_BUTTON_Plot_XgrUnwZgrLbl PUSHBUTTON 300 842 80 25
  VALUE "Xgr,UZgr,Lbl"
  ALIGNCENTER
  ONACTIVATE "OnPlotXgrUnwZgrLbl"
  END
  WID_BUTTON_FilterLabel PUSHBUTTON 274 966 100 25
  VALUE "Add Filter Lbl."
  ALIGNCENTER
  ONACTIVATE "OnAddSigmaFilterButton"
  END
  WID_BUTTON_AddAllLabels PUSHBUTTON 300 934 70 25
  VALUE "Add All"
  ALIGNCENTER
  ONACTIVATE "OnAddAllLabels"
  END
  WID_BUTTON_Plot_XZLbl PUSHBUTTON 375 812 65 25
  VALUE "X,Z,Lbl"
  ALIGNCENTER
  ONACTIVATE "OnPlotXZLbl"
  END
  WID_BUTTON_Overlay_All_Centers PUSHBUTTON 250 997 125 25
  VALUE "Overlay All Centers"
  ALIGNCENTER
  ONACTIVATE "OnPeakOverlayAllCentersButton"
  END
  WID_BUTTON_Overlay_Frame_Centers PUSHBUTTON 120 997 125 25
  VALUE "Overlay Fr. Centers"
  ALIGNCENTER
  ONACTIVATE "OnPeakOverlayFrameCentersButton"
  END
  WID_BUTTON_Overlay_DIC PUSHBUTTON 15 662 130 25
  VALUE "Overlay DIC/EM"
  ALIGNCENTER
  ONACTIVATE "OverlayDIC"
  END
  WID_BUTTON_CustomTIFF PUSHBUTTON 450 742 110 30
  VALUE "Cust. TIFF"
  ALIGNCENTER
  ONACTIVATE "SaveCustomTIFF_menu"
  END
  WID_BUTTON_Plot_FrameUnwrZX PUSHBUTTON 20 842 60 25
  VALUE "Fr, UnwZ"
  ALIGNCENTER
  ONACTIVATE "OnPlotFrameUnwrZX"
  END
  WID_BASE_Redraw BASE 16 42 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Redraw PUSHBUTTON -1 -1 0 0
    VALUE "Redraw"
    ALIGNLEFT
    END
  END
  WID_BUTTON_Plot_spectrum PUSHBUTTON 10 997 100 25
  VALUE "Plot Spectra"
  ALIGNCENTER
  ONACTIVATE "OnPeak_Plot_Spectrum"
  END
  WID_BASE_Redraw_0 BASE 160 42 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Hist_Log_X PUSHBUTTON -1 -1 0 0
    VALUE "Hist. X Log"
    ALIGNLEFT
    ONACTIVATE "Set_Hist_Log_X"
    END
  END
  WID_BASE_Redraw_1 BASE 280 42 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Hist_Log_Y PUSHBUTTON -1 -1 0 0
    VALUE "Hist. Y Log"
    ALIGNLEFT
    ONACTIVATE "Set_Hist_Log_Y"
    END
  END
  WID_BASE_Redraw_2 BASE 430 42 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Allow_Bridge PUSHBUTTON -1 -1 0 0
    VALUE "Allow Bridge"
    ALIGNLEFT
    ONACTIVATE "Set_Allow_Bridge"
    END
  END
  WID_BASE_Tie_RGB BASE 415 1005 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Tie_RGB PUSHBUTTON -1 -1 0 0
    VALUE "Tie RGB"
    ALIGNLEFT
    ONACTIVATE "Set_Tie_RGB"
    END
  END
END
