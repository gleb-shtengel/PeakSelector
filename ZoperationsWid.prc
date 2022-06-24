HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	11/10/2021 15:55.30
VERSION 1
END

WID_BASE_Z_operations BASE 5 5 601 791
REALIZE "Initialize_Z_operations"
TLB
CAPTION "Z-coordinate Operations"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_SLIDER_Z_phase_offset SLIDER 13 108 167 50
  CAPTION "Z offset (nm)"
  VALUE = 0
  MINIMUM = -200
  MAXIMUM = 200
  END
  WID_SLIDER_Z_phase_slope SLIDER 193 108 167 50
  CAPTION "Z slope (nm/1000 frames)"
  VALUE = 0
  MINIMUM = -200
  MAXIMUM = 200
  END
  WID_BUTTON_Test_Wind_3D PUSHBUTTON 15 8 150 35
  VALUE "Test Wind Point 3D"
  ALIGNCENTER
  ONACTIVATE "OnTestWindPoint"
  END
  WID_BUTTON_Write_Wind_3D PUSHBUTTON 195 8 150 35
  VALUE "Write Wind Point 3D"
  ALIGNCENTER
  ONACTIVATE "OnWriteCalibWind"
  END
  WID_BUTTON_ExtractZ PUSHBUTTON 395 99 150 35
  VALUE "Extract Z coordinate"
  ALIGNCENTER
  ONACTIVATE "OnExtractZCoord"
  END
  WID_BUTTON_Add_Offset_Slope PUSHBUTTON 19 172 150 35
  VALUE "Add Offset/Slope"
  ALIGNCENTER
  ONACTIVATE "OnAddOffsetSlope"
  END
  WID_BUTTON_Close PUSHBUTTON 505 727 80 35
  VALUE "Close"
  ALIGNCENTER
  ONACTIVATE "OnButtonClose"
  END
  WID_BUTTON_PickWINDFile PUSHBUTTON 395 28 150 32
  VALUE "Pick WND File"
  ALIGNCENTER
  ONACTIVATE "OnPickWINDFile"
  END
  WID_TEXT_WindFilename TEXT 5 50 380 49
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_Write_Guide_Star PUSHBUTTON 408 362 130 25
  VALUE "Write Guide Star"
  ALIGNCENTER
  ONACTIVATE "OnWriteZDrift"
  END
  WID_BUTTON_Test_Guide_Star PUSHBUTTON 408 335 130 25
  VALUE "Test Guide Star"
  ALIGNCENTER
  ONACTIVATE "OnTestZDrift"
  END
  WID_SLIDER_Z_Sm_Width SLIDER 33 395 300 48
  CAPTION "Guide Star Smoothing Width"
  VALUE = 250
  MINIMUM = 1
  MAXIMUM = 1000
  END
  WID_SLIDER_Z_Fit SLIDER 34 345 300 48
  CAPTION "Guide Star Polyn. Fit Order"
  VALUE = 5
  MINIMUM = 1
  MAXIMUM = 10
  END
  WID_DROPLIST_Z_Fit_Method DROPLIST 48 315 300 30
  CAPTION "GuideStar Drift Correction"
  NUMITEMS = 2
  ITEM "Polynomial"
  ITEM "Weighted Smoothing"
  END
  WID_BUTTON_Safe_Wind_ASCII PUSHBUTTON 395 148 150 35
  VALUE "Safe Wind Curves ASCII"
  ALIGNCENTER
  ONACTIVATE "OnWriteCalibASCII"
  END
  WID_TEXT_WindFilename_ASCII TEXT 376 188 175 50
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_Remove_Tilt PUSHBUTTON 199 172 150 35
  VALUE "Remove XYZ Tilt"
  ALIGNCENTER
  ONACTIVATE "OnRemoveTilt"
  END
  WID_BUTTON_Read_Wind_Period PUSHBUTTON 18 480 150 30
  VALUE "Read Wind Period (nm)"
  ALIGNCENTER
  ONACTIVATE "ReadWindPoint"
  END
  WID_BUTTON_Write_Wind_Period PUSHBUTTON 258 480 150 30
  VALUE "Write Wind Period (nm)"
  ALIGNCENTER
  ONACTIVATE "WriteWindPoint"
  END
  WID_TEXT_WindPeriod TEXT 178 480 70 30
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_Zvalue TEXT 13 718 125 30
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_Write_Zvalue PUSHBUTTON 284 718 200 30
  VALUE "Write Z-value and Z-uncertainty"
  ALIGNCENTER
  ONACTIVATE "On_Write_Zvalue"
  END
  WID_LABEL_Zvalue LABEL 19 703 130 15
  VALUE "Z value (nm) to write"
  ALIGNLEFT
  END
  WID_TEXT_Zuncertainty TEXT 151 718 125 30
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_Zuncertainty LABEL 153 703 170 15
  VALUE "Z uncertainty (nm) to write"
  ALIGNLEFT
  END
  WID_TABLE_EllipticityFitCoeff TABLE 16 520 393 57
  N_ROWS = 1
  N_COLS = 3
  NUMROWLABELS = 1
  ROWLABEL "Ellipticity Fit Coeff."
  EDITABLE
  ONINSERTCHAR "Edit_Ellipticity_Coeff"
  END
  WID_BUTTON_PlotEllipticity PUSHBUTTON 18 585 230 35
  VALUE "Plot Ellipticity vs. Z: Data + Fit"
  ALIGNCENTER
  ONACTIVATE "OnPlotEllipticityDataAndFit"
  END
  WID_BUTTON_Test_EllipticityOnly PUSHBUTTON 258 585 160 35
  VALUE "Test Ellipticity vs Frame"
  ALIGNCENTER
  ONACTIVATE "OnTestEllipOnly"
  END
  WID_BUTTON_Save_EllipticityAndWind PUSHBUTTON 428 585 150 35
  VALUE "Save Ellipticity + Wind"
  ALIGNCENTER
  ONACTIVATE "OnSaveEllipAndWind"
  END
  WID_BUTTON_UnwrapZ PUSHBUTTON 458 535 100 35
  VALUE "Unwrap Z"
  ALIGNCENTER
  ONACTIVATE "UnwrapZCoord"
  END
  WID_BUTTON_Write_Wind_Period_without_scaling PUSHBUTTON 108 445 230 30
  VALUE "Write Wind Period w/o Scaling"
  ALIGNCENTER
  ONACTIVATE "WriteWindPointWithoutScaling"
  END
  WID_BASE_WriteEllipticityGuideStar BASE 370 390 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_WriteEllipticityGuideStar PUSHBUTTON -1 -1 0 0
    VALUE "Correct Ellipticity for GuideStar - Z"
    ALIGNLEFT
    ONACTIVATE "On_Buttonpress_WriteEllipticityGuideStar_Z"
    END
  END
  WID_TABLE_EllipticityCorrectionSlope TABLE 10 627 380 57
  N_ROWS = 1
  N_COLS = 4
  NUMCOLLABELS = 4
  COLLABEL "X cntr. (pix)"
  COLLABEL "X slp. (nm/pix)"
  COLLABEL "Y cntr. (pix)"
  COLLABEL "Y slp. (nm/pix)"
  NUMROWLABELS = 2
  ROWLABEL ""
  ROWLABEL ""
  EDITABLE
  ONINSERTCHAR "Edit_Ellipticity_Correction_Slope"
  END
  WID_BASE_WriteEllipticityGuideStar_0 BASE 396 649 180 22
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_AddEllipticitySlopeCorrection PUSHBUTTON -1 -1 178 22
    VALUE "Unwrap with Ell. Slope Corr."
    ALIGNLEFT
    END
  END
  WID_LABEL_ElliptSlopeCorr LABEL 398 632 170 18
  VALUE "Ellipticity Slope Correction"
  ALIGNLEFT
  END
  WID_BUTTON_UnwrapZ_Lookup PUSHBUTTON 431 480 140 35
  VALUE "Lookup  Unwrap Z"
  ALIGNCENTER
  ONACTIVATE "LookupUnwrapZCoord"
  END
  WID_DROPLIST_LookupUnwrapDisplayType DROPLIST 353 449 225 24
  CAPTION "Lookup Display"
  NUMITEMS = 3
  ITEM "Local - w/o Display"
  ITEM "Local - with Display"
  ITEM "Cluster - No Display"
  END
  WID_TEXT_GuideStarAncFilename TEXT 15 224 344 49
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_PickAncFile PUSHBUTTON 380 245 150 25
  VALUE "Pick ANC File"
  ALIGNCENTER
  ONACTIVATE "OnPickGuideStarAncFile"
  END
  WID_BUTTON_Use_Multiple_Guidestars_GS BASE 380 275 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Use_Multiple_Guidestars_ZGS PUSHBUTTON -1 -1 0 0
    VALUE "Use Multiple GuideStars GS"
    ALIGNLEFT
    ONACTIVATE "OnButton_Press_use_multiple_guidestars_ZGS"
    END
  END
  WID_BUTTON_Write_GuideStarRadius PUSHBUTTON 19 278 250 30
  VALUE "Set Guidestar Area Radius (pix)"
  ALIGNCENTER
  ONACTIVATE "WriteGudeStarRadius"
  END
  WID_TEXT_GuideStar_Radius TEXT 272 278 70 30
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_WL TEXT 501 62 70 30
  NUMITEMS = 1
  ITEM "590.0"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_WL LABEL 400 71 100 25
  VALUE "Wavelength (nm)"
  ALIGNLEFT
  END
  WID_BUTTON_OptimizeSlopeCorrection PUSHBUTTON 398 678 183 30
  VALUE "Optimize Slope Corr."
  ALIGNCENTER
  ONACTIVATE "OptimizeSlopeCorrection"
  END
  WID_BASE_WriteEllipticityGuideStar_E BASE 370 417 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_WriteEllipticityGuideStar_E PUSHBUTTON -1 -1 0 0
    VALUE "Correct Ellipticity for GuideStar - E"
    ALIGNLEFT
    ONACTIVATE "On_Buttonpress_WriteEllipticityGuideStar_E"
    END
  END
  WID_DROPLIST_Optimization_Mode DROPLIST 235 675 150 24
  CAPTION "Mode"
  NUMITEMS = 4
  ITEM "Local Groups"
  ITEM "Local Peaks"
  ITEM "Bridge Groups"
  ITEM "Bridge Peaks"
  END
  WID_BUTTON_Use_Multiple_Guidestars_DH BASE 380 305 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  XPAD = 5
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Use_Multiple_Guidestars_ZDH PUSHBUTTON -1 -1 0 0
    VALUE "Use Multiple GuideStars DH"
    ALIGNLEFT
    ONACTIVATE "OnButton_Press_use_multiple_guidestars_ZDH"
    END
  END
END
