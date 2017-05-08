HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	11/30/2010 12:39.10
VERSION 1
END

WID_BASE_ReExtractMultiLabel BASE 5 5 674 251
REALIZE "Initialize_ReExtractMultiLabel"
TLB
CAPTION "Re-Extract Peaks Multilablel"
MODAL
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_StartReExtract PUSHBUTTON 320 155 150 55
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "StartReExtract"
  END
  WID_DROPLIST_FitDisplayType DROPLIST 14 156 225 30
  CAPTION "Fit-Display Level"
  NUMITEMS = 4
  ITEM "No Display"
  ITEM "Some Frames/Peaks"
  ITEM "All Frames/Peaks "
  ITEM "Cluster - No Display"
  END
  WID_BUTTON_CancelReExtract PUSHBUTTON 500 155 150 55
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancelReExtract"
  END
  WID_DROPLIST_SetSigmaFitSym_Reextract DROPLIST 14 191 225 30
  CAPTION "SetSigmaFitSymmetry"
  NUMITEMS = 2
  ITEM "R"
  ITEM "X Y"
  ONSELECT "SetSigmaSym_Reextract"
  END
  WID_BUTTON_PickFile1 PUSHBUTTON 509 10 150 30
  VALUE "Pick Cam1 File"
  ALIGNCENTER
  ONACTIVATE "OnPickCam1File"
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
  ONACTIVATE "OnPickCam2File"
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
  ONACTIVATE "OnPickCam3File"
  END
END
