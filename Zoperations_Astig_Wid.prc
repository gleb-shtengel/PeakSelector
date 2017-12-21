HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	12/21/2017 11:41.11
VERSION 1
END

WID_BASE_Z_operations_Astig BASE 5 5 614 651
REALIZE "Initialize_Z_operations_Astig"
TLB
CAPTION "Z-coordinate Operations (Astigmatism Only - no Interference)"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_SLIDER_Z_phase_offset_Astig SLIDER 23 301 167 50
  CAPTION "Z offset (nm)"
  VALUE = 0
  MINIMUM = -200
  MAXIMUM = 200
  END
  WID_BUTTON_ExtractZ_Astig PUSHBUTTON 427 37 150 35
  VALUE "Extract Z coordinate"
  ALIGNCENTER
  ONACTIVATE "OnExtractZCoord_Astig"
  END
  WID_BUTTON_Add_Offset_Astig PUSHBUTTON 204 307 150 35
  VALUE "Add Offset"
  ALIGNCENTER
  ONACTIVATE "OnAddOffset_Astig"
  END
  WID_BUTTON_Close PUSHBUTTON 414 550 145 45
  VALUE "Close"
  ALIGNCENTER
  ONACTIVATE "OnButtonClose_Astig"
  END
  WID_BUTTON_PickCalFile PUSHBUTTON 427 0 150 32
  VALUE "Pick CAL (WND) File"
  ALIGNCENTER
  ONACTIVATE "OnPickCalFile_Astig"
  END
  WID_TEXT_WindFilename_Astig TEXT 5 4 415 49
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_Write_Guide_Star_Astig PUSHBUTTON 406 505 130 30
  VALUE "Write Guide Star"
  ALIGNCENTER
  ONACTIVATE "OnWriteZDrift_Astig"
  END
  WID_BUTTON_Test_Guide_Star_Astig PUSHBUTTON 406 465 130 30
  VALUE "Test Guide Star"
  ALIGNCENTER
  ONACTIVATE "OnTestZDrift_Astig"
  END
  WID_SLIDER_Z_Sm_Width SLIDER 31 540 300 55
  CAPTION "Guide Star Smoothing Width"
  VALUE = 250
  MINIMUM = 1
  MAXIMUM = 1000
  END
  WID_SLIDER_Z_Fit SLIDER 32 475 300 55
  CAPTION "Guide Star Polyn. Fit Order"
  VALUE = 5
  MINIMUM = 1
  MAXIMUM = 10
  END
  WID_DROPLIST_Z_Fit_Method DROPLIST 46 445 300 30
  CAPTION "GuideStar Drift Correction"
  NUMITEMS = 2
  ITEM "Polynomial"
  ITEM "Weighted Smoothing"
  END
  WID_BUTTON_Remove_Tilt_Astig PUSHBUTTON 407 307 150 35
  VALUE "Remove XYZ Tilt"
  ALIGNCENTER
  ONACTIVATE "OnRemoveTilt_Astig"
  END
  WID_BUTTON_Test_EllipticityOnly PUSHBUTTON 20 221 180 35
  VALUE "Test Ellipticity Calibration"
  ALIGNCENTER
  ONACTIVATE "ExtractEllipticityCalib_Astig"
  END
  WID_BUTTON_Save_EllipticityAndWind PUSHBUTTON 220 221 180 35
  VALUE "Save Ellipticity Calibration"
  ALIGNCENTER
  ONACTIVATE "OnSaveEllipticityCal_Astig"
  END
  WID_TEXT_GuideStarAncFilename_Astig TEXT 3 354 365 49
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_PickAncFile PUSHBUTTON 378 355 125 30
  VALUE "Pick ANC File"
  ALIGNCENTER
  ONACTIVATE "OnPickGuideStarAncFile_Astig"
  END
  WID_BASE_WriteEllipticityGuideStar_1 BASE 378 390 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_UseMultipleANCs PUSHBUTTON -1 -1 0 0
    VALUE "Use Multiple GuideStars"
    ALIGNLEFT
    ONACTIVATE "OnButton_Press_use_multiple_GS"
    END
  END
  WID_BUTTON_Write_GuideStarRadius PUSHBUTTON 17 408 250 30
  VALUE "Set Guidestar Area Radius (pix)"
  ALIGNCENTER
  ONACTIVATE "WriteGudeStarRadius_Astig"
  END
  WID_TEXT_GuideStar_Radius_Astig TEXT 270 408 70 30
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_ZCalStep_Astig TEXT 480 130 74 35
  EDITABLE
  WRAP
  ALLEVENTS
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_Cal_Step LABEL 400 135 80 25
  VALUE "Z Step (nm)"
  ALIGNLEFT
  END
  WID_SLIDER_Zastig_Fit SLIDER 33 52 300 55
  CAPTION "Polyn. Fit Order for Astigmatism vs Z"
  VALUE = 5
  MINIMUM = 1
  MAXIMUM = 10
  END
  WID_BUTTON_Convert_Fr_to_Z PUSHBUTTON 420 221 140 35
  VALUE "Convert Fr -> Z"
  ALIGNCENTER
  ONACTIVATE "On_Convert_Frame_to_Z"
  END
  WID_BASE_WriteEllipticity_MS_GuideStar_DPH BASE 379 427 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_UseMultipleANCs_DH PUSHBUTTON -1 -1 0 0
    VALUE "Use Multiple GuideStars (DH)"
    ALIGNLEFT
    ONACTIVATE "OnButton_Press_use_multiple_GS_DH"
    END
  END
  WID_TEXT_Zmin_Astig TEXT 80 130 100 35
  EDITABLE
  WRAP
  ALLEVENTS
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_Cal_Zmin LABEL 10 135 70 25
  VALUE "Z min (nm)"
  ALIGNLEFT
  END
  WID_TEXT_Zmax_Astig TEXT 270 130 100 35
  EDITABLE
  WRAP
  ALLEVENTS
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_Cal_Zmax LABEL 200 135 70 25
  VALUE "Z max (nm)"
  ALIGNLEFT
  END
  WID_LABEL_Cal_num_iter LABEL 10 180 80 25
  VALUE "# of iterations"
  ALIGNLEFT
  END
  WID_TEXT_ZCal_Astig_num_iter TEXT 110 175 75 35
  EDITABLE
  WRAP
  ALLEVENTS
  WIDTH = 20
  HEIGHT = 2
  END
  WID_DROPLIST_LegendColor DROPLIST 235 178 250 31
  CAPTION "Color Order"
  NUMITEMS = 4
  ITEM "Fiducial #"
  ITEM "Fiducial X"
  ITEM "Fiducial Y"
  ITEM "Fiducial Frame#"
  END
  WID_BUTTON_Plot_ZvsFfame_woffset PUSHBUTTON 220 266 180 35
  VALUE "Plot Z vs Frame w offset"
  ALIGNCENTER
  ONACTIVATE "Plot_ZvsFrame_with_offest"
  END
END
