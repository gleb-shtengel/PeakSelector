HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	04/19/2011 16:21.00
VERSION 1
END

WID_BASE_GuideStar BASE 5 5 332 492
REALIZE "Initialize_XY_GuideStar"
TLB
CAPTION "Test/Write XY Guide Star"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_Write_Guide_Star PUSHBUTTON 168 362 130 32
  VALUE "Write Guide Star"
  ALIGNCENTER
  ONACTIVATE "OnWriteGuideStarXY"
  END
  WID_SLIDER_XY_Fit SLIDER 13 50 300 48
  CAPTION "Guide Star Polyn. Fit Order"
  VALUE = 5
  MINIMUM = 1
  MAXIMUM = 10
  END
  WID_BUTTON_Test_Guide_Star PUSHBUTTON 20 362 130 32
  VALUE "Test Guide Star"
  ALIGNCENTER
  ONACTIVATE "OnTestGuideStarXY"
  END
  WID_SLIDER_XY_Sm_Width SLIDER 14 110 300 48
  CAPTION "Guide Star Smoothing Width"
  VALUE = 100
  MINIMUM = 1
  MAXIMUM = 1000
  END
  WID_DROPLIST_XY_Fit_Method DROPLIST 15 16 300 30
  CAPTION "GuideStar Drift Correction"
  NUMITEMS = 2
  ITEM "Polynomial"
  ITEM "Weighted Smoothing"
  END
  WID_BUTTON_Close PUSHBUTTON 90 414 145 35
  VALUE "Close"
  ALIGNCENTER
  ONACTIVATE "OnButtonClose"
  END
  WID_TEXT_XY_GuideStarAncFilename TEXT 8 231 296 49
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_PickAncFile_MultipleGS_XY PUSHBUTTON 160 194 150 32
  VALUE "Pick ANC File"
  ALIGNCENTER
  ONACTIVATE "OnPick_XYGuideStarAncFile"
  END
  WID_BASE_XY_Multiple_Guidestars BASE 14 198 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_UseMultipleGuideStars_XY PUSHBUTTON -1 -1 0 0
    VALUE "Use Multiple GuideStars"
    ALIGNLEFT
    END
  END
  WID_BUTTON_Write_GuideStarRadius PUSHBUTTON 9 296 218 30
  VALUE "Set Guidestar Area Radius (pix)"
  ALIGNCENTER
  ONACTIVATE "Write_XY_GudeStarRadius"
  END
  WID_TEXT_XY_GuideStar_Radius TEXT 237 297 70 30
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
END
