HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	11/30/2010 14:04.07
VERSION 1
END

WID_BASE_TransformedFilenames BASE 5 5 618 440
REALIZE "Initialize_TransformedFilenames"
TLB
CAPTION "Select names for Transformed Data Files"
MODAL
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_StartTransform PUSHBUTTON 6 215 150 55
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "StartSaveTransformed"
  END
  WID_TEXT_Label1Filename TEXT 4 20 600 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_Label2Filename TEXT 4 70 600 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_Label3Filename TEXT 4 120 600 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_SumFilename TEXT 3 170 600 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_SumFile LABEL 5 155 0 0
  VALUE "Sum File:"
  ALIGNLEFT
  END
  WID_LABEL_SumFile_0 LABEL 5 5 0 0
  VALUE "Label 1 File:"
  ALIGNLEFT
  END
  WID_LABEL_SumFile_1 LABEL 5 55 0 0
  VALUE "Label 2 File:"
  ALIGNLEFT
  END
  WID_LABEL_SumFile_2 LABEL 5 105 0 0
  VALUE "Label 3 File:"
  ALIGNLEFT
  END
  WID_BUTTON_CancelReExtract PUSHBUTTON 449 216 150 55
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancelSave"
  END
  WID_BUTTON_TransformAndReExtract PUSHBUTTON 183 215 195 55
  VALUE "Confirm, Transform, ReExtract"
  ALIGNCENTER
  ONACTIVATE "StartTransformAndReExtract"
  END
  WID_DROPLIST_SetSigmaFitSym_TRS DROPLIST 340 290 225 30
  CAPTION "SetSigmaFitSymmetry"
  NUMITEMS = 2
  ITEM "R"
  ITEM "X Y"
  END
  WID_DROPLIST_TransformEngine DROPLIST 20 290 225 30
  CAPTION "Transformation Engine"
  NUMITEMS = 3
  ITEM "Local"
  ITEM "Cluster"
  ITEM "IDL Bridge"
  END
  WID_SLIDER_FramesPerNode_tr SLIDER 380 345 150 50
  CAPTION "Frames per Node (Cluster)"
  VALUE = 2500
  MINIMUM = 0
  MAXIMUM = 10000
  END
  WID_SLIDER_Group_Gap_tr SLIDER 20 345 150 50
  CAPTION "Group Gap"
  VALUE = 3
  MAXIMUM = 32
  END
  WID_SLIDER_Grouping_Radius_tr SLIDER 200 345 150 50
  CAPTION "Grouping Radius*100"
  VALUE = 25
  MAXIMUM = 200
  END
END
