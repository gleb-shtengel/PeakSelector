HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	03/10/2014 09:48.50
VERSION 1
END

WID_BASE_ExtractMultiLabel BASE 5 5 674 251
REALIZE "Initialize_ExtractMultiLabel"
TLB
CAPTION "Extract Peaks Multilablel"
MODAL
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_StartMLExtract PUSHBUTTON 350 155 150 55
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "StartMLExtract"
  END
  WID_DROPLIST_FitDisplayType DROPLIST 5 156 225 30
  CAPTION "Fit-Display Level"
  NUMITEMS = 5
  ITEM "No Display"
  ITEM "Some Frames/Peaks"
  ITEM "All Frames/Peaks "
  ITEM "Cluster - No Display"
  ITEM "IDL Bridge - No Disp"
  END
  WID_BUTTON_CancelReExtract PUSHBUTTON 510 155 150 55
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancelReExtract"
  END
  WID_DROPLIST_SetSigmaFitSym_ML DROPLIST 5 190 175 30
  CAPTION "SetSigmaFitSymmetry"
  NUMITEMS = 2
  ITEM "R"
  ITEM "X Y"
  ONSELECT "SetSigmaSym_ML"
  END
  WID_BUTTON_PickFile1 PUSHBUTTON 509 10 150 30
  VALUE "Pick Cam1 File"
  ALIGNCENTER
  ONACTIVATE "OnPickCam1DatFile"
  END
  WID_TEXT_Cam1Filename TEXT 4 9 500 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_Cam2Filename TEXT 4 59 500 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_PickFile2 PUSHBUTTON 509 60 150 30
  VALUE "Pick Cam2 File"
  ALIGNCENTER
  ONACTIVATE "OnPickCam2DatFile"
  END
  WID_TEXT_Cam3Filename TEXT 4 109 500 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_PickFile3 PUSHBUTTON 509 110 150 30
  VALUE "Pick Cam3 File"
  ALIGNCENTER
  ONACTIVATE "OnPickCam3DatFile"
  END
  WID_BASE_0 BASE 215 190 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_Use_InfoFile_Flip PUSHBUTTON -1 -1 0 0
    VALUE "Use Info File to Flip"
    ALIGNLEFT
    END
  END
END
