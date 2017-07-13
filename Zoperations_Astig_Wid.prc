HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	07/12/2017 11:23.44
VERSION 1
END

WID_BASE_Z_operations_Astig BASE 5 5 601 513
REALIZE "Initialize_Z_operations_Astig"
TLB
CAPTION "Z-coordinate Operations (Astigmatism Only - no Interference)"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_SLIDER_Z_phase_offset_Astig SLIDER 25 171 167 50
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
  WID_BUTTON_Add_Offset_Astig PUSHBUTTON 206 177 150 35
  VALUE "Add Offset"
  ALIGNCENTER
  ONACTIVATE "OnAddOffset_Astig"
  END
  WID_BUTTON_Close PUSHBUTTON 416 420 145 45
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
  WID_BUTTON_Write_Guide_Star_Astig PUSHBUTTON 408 375 130 30
  VALUE "Write Guide Star"
  ALIGNCENTER
  ONACTIVATE "OnWriteZDrift_Astig"
  END
  WID_BUTTON_Test_Guide_Star_Astig PUSHBUTTON 408 335 130 30
  VALUE "Test Guide Star"
  ALIGNCENTER
  ONACTIVATE "OnTestZDrift_Astig"
  END
  WID_SLIDER_Z_Sm_Width SLIDER 33 410 300 55
  CAPTION "Guide Star Smoothing Width"
  VALUE = 250
  MINIMUM = 1
  MAXIMUM = 1000
  END
  WID_SLIDER_Z_Fit SLIDER 34 345 300 55
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
  WID_BUTTON_Remove_Tilt_Astig PUSHBUTTON 409 177 150 35
  VALUE "Remove XYZ Tilt"
  ALIGNCENTER
  ONACTIVATE "OnRemoveTilt_Astig"
  END
  WID_BUTTON_Test_EllipticityOnly PUSHBUTTON 20 130 180 35
  VALUE "Test Ellipticity Calibration"
  ALIGNCENTER
  ONACTIVATE "ExtractEllipticityCalib_Astig"
  END
  WID_BUTTON_Save_EllipticityAndWind PUSHBUTTON 220 130 180 35
  VALUE "Save Ellipticity Calibration"
  ALIGNCENTER
  ONACTIVATE "OnSaveEllipticityCal_Astig"
  END
  WID_TEXT_GuideStarAncFilename_Astig TEXT 5 224 365 49
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_PickAncFile PUSHBUTTON 380 225 125 30
  VALUE "Pick ANC File"
  ALIGNCENTER
  ONACTIVATE "OnPickGuideStarAncFile_Astig"
  END
  WID_BASE_WriteEllipticityGuideStar_1 BASE 380 260 0 0
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
  WID_BUTTON_Write_GuideStarRadius PUSHBUTTON 19 278 250 30
  VALUE "Set Guidestar Area Radius (pix)"
  ALIGNCENTER
  ONACTIVATE "WriteGudeStarRadius_Astig"
  END
  WID_TEXT_GuideStar_Radius_Astig TEXT 272 278 70 30
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_ZCalStep_Astig TEXT 496 82 74 35
  EDITABLE
  WRAP
  ALLEVENTS
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_Cal_Step LABEL 388 88 112 27
  VALUE "Cal. Z Step (nm)"
  ALIGNLEFT
  END
  WID_SLIDER_Zastig_Fit SLIDER 33 52 300 55
  CAPTION "Polyn. Fit Order for Astigmatism vs Z"
  VALUE = 5
  MINIMUM = 1
  MAXIMUM = 10
  END
  WID_BUTTON_Convert_Fr_to_Z PUSHBUTTON 420 130 140 35
  VALUE "Convert Fr -> Z"
  ALIGNCENTER
  ONACTIVATE "On_Convert_Frame_to_Z"
  END
  WID_BASE_WriteEllipticity_MS_GuideStar_DPH BASE 380 295 0 0
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
END
