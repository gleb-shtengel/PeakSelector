HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	11/09/2020 10:41.30
VERSION 1
END

WID_BASE_iPALM_MACRO BASE 5 5 749 870
REALIZE "Initialize_Transform_Extract_ReExtract_Filter_GetZ_iPALM"
TLB
CAPTION "iPALM Macro: Transform, Extract, Reextract, Fiter, Group, Extract Z"
MODAL
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_StartTrsnformation_Macro PUSHBUTTON 530 720 180 40
  VALUE "Perform Transformation Only"
  ALIGNCENTER
  ONACTIVATE "Start_Transformation_Macro"
  END
  WID_BUTTON_CancelReExtract PUSHBUTTON 570 780 150 40
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCanceliPALM_Macro"
  END
  WID_DROPLIST_SetSigmaFitSym_iPALM DROPLIST 280 190 175 30
  CAPTION "SetSigmaFitSymmetry"
  NUMITEMS = 2
  ITEM "R"
  ITEM "X Y"
  ONSELECT "Set_SigmaFitSym_iPALM"
  END
  WID_BTTN_PickFile1 PUSHBUTTON 509 10 150 30
  VALUE "Pick Cam1 File"
  ALIGNCENTER
  ONACTIVATE "OnPickCam1TxtFile"
  END
  WID_TXT_Cam1Filename TEXT 4 10 500 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TXT_Cam2Filename TEXT 4 50 500 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BTTN_PickFile2 PUSHBUTTON 509 50 150 30
  VALUE "Pick Cam2 File"
  ALIGNCENTER
  ONACTIVATE "OnPickCam2TxtFile"
  END
  WID_TXT_Cam3Filename TEXT 3 90 500 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BTTN_PickFile3 PUSHBUTTON 508 90 150 30
  VALUE "Pick Cam3 File"
  ALIGNCENTER
  ONACTIVATE "OnPickCam3TxtFile"
  END
  WID_TXT_ANCFilename TEXT 4 145 500 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BTTN_PickANCFile PUSHBUTTON 509 144 150 30
  VALUE "Pick *.anc File"
  ALIGNCENTER
  ONACTIVATE "OnPickANCFile_iPALM"
  END
  WID_BTTN_PickWINDFile PUSHBUTTON 565 530 150 32
  VALUE "Pick WND File"
  ALIGNCENTER
  ONACTIVATE "OnPickWINDFile_iPALM"
  END
  WID_TXT_WindFilename TEXT 294 566 430 50
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_SLIDER_Grouping_Radius_iPALM SLIDER 435 640 140 50
  CAPTION "Grouping Radius*100"
  VALUE = 25
  MAXIMUM = 200
  END
  WID_SLIDER_Group_Gap_iPALM SLIDER 280 640 140 50
  CAPTION "Group Gap"
  VALUE = 3
  MAXIMUM = 32
  END
  WID_SLIDER_FramesPerNode_iPALM SLIDER 590 640 140 50
  CAPTION "Frames per Node (Cluster)"
  VALUE = 2500
  MINIMUM = 0
  MAXIMUM = 10000
  END
  WID_DROPLIST_TransformEngine_iPALM DROPLIST 10 190 225 30
  CAPTION "Transformation Engine"
  NUMITEMS = 3
  ITEM "Local"
  ITEM "Cluster"
  ITEM "IDL Bridge"
  ONSELECT "Set_TransformEngine_iPALM"
  END
  WID_Filter_Parameters_iPALM_Macro TABLE 380 270 320 234
  N_ROWS = 10
  N_COLS = 1
  NUMCOLLABELS = 1
  COLLABEL "Value"
  NUMROWLABELS = 10
  ROWLABEL "Nph. Min."
  ROWLABEL "Full Sigma X Max. (pix.)"
  ROWLABEL "Full Sigma Y Max. (pix.)"
  ROWLABEL "Sigma Z Max. (nm)"
  ROWLABEL "Coherence Min."
  ROWLABEL "Coherence Max."
  ROWLABEL "Max. A for (Wx-A)*(Wx-A)<B"
  ROWLABEL "B for (Wx-A)*(Wx-A)<B"
  ROWLABEL "Min. A  for (Wx-A)*(Wx-A)>B"
  ROWLABEL "B for (Wx-A)*(Wx-A)>B"
  EDITABLE
  END
  WID_BASE_UseInfoFileToFlip_iPALM BASE 515 196 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_Use_InfoFile_Flip PUSHBUTTON -1 -1 0 0
    VALUE "Use Info File to Flip"
    ALIGNLEFT
    END
  END
  WID_BASE_UseInfoFileToFlip_iPALM_0 BASE 515 235 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_Use_SkipTransformation PUSHBUTTON -1 -1 0 0
    VALUE "Skip Data Transformation"
    ALIGNLEFT
    END
  END
  WID_BUTTON_StartMLExtract_Fast PUSHBUTTON 330 720 150 40
  VALUE "Confirm and Start Fast"
  ALIGNCENTER
  ONACTIVATE "Start_iPALM_Macro_Fast"
  END
  WID_LABEL_ConfirmStartFast LABEL 312 765 249 18
  VALUE "Large Transformed Files are not creared"
  ALIGNLEFT
  END
  WID_LABEL_ConfirmStartFast_1 LABEL 311 782 249 18
  VALUE "Works only in Cluseter or IDL Bridge modes"
  ALIGNLEFT
  END
  WID_TABLE_InfoFile_iPALM_macro TABLE 10 265 260 540
  FRAME = 1
  N_ROWS = 26
  N_COLS = 1
  NUMCOLLABELS = 1
  COLLABEL "Values"
  NUMROWLABELS = 26
  ROWLABEL "Zero Dark Cnt"
  ROWLABEL "X pixels Data"
  ROWLABEL "Y pixels Data"
  ROWLABEL "Max # Frames"
  ROWLABEL "Initial Frame"
  ROWLABEL "Final Frame"
  ROWLABEL "Peak Threshold Criteria"
  ROWLABEL "File type (0 - .dat, 1 - .tif)"
  ROWLABEL "Min Peak Ampl."
  ROWLABEL "Max Peak Ampl."
  ROWLABEL "Min Peak Width"
  ROWLABEL "Max Peak Width"
  ROWLABEL "Limit ChiSq"
  ROWLABEL "Counts Per Electron"
  ROWLABEL "Max # Peaks Iter1"
  ROWLABEL "Max # Peaks Iter2"
  ROWLABEL "Flip Horizontal"
  ROWLABEL "Flip Vertical"
  ROWLABEL "(Half) Gauss Size (d)"
  ROWLABEL "Appr. Gauss Sigma"
  ROWLABEL "Max Block Size"
  ROWLABEL "Sparse OverSampling"
  ROWLABEL "Sparse Lambda"
  ROWLABEL "Sparse Delta"
  ROWLABEL "Sp. Max Error"
  ROWLABEL "Sp. Max # of Iter."
  EDITABLE
  ONINSERTCHAR "DoInsertInfo_iPALM_Macro"
  END
  WID_DROPLIST_ZExctractEngine_iPALM DROPLIST 31 222 225 30
  CAPTION "Z-Extraction Engine"
  NUMITEMS = 2
  ITEM "Local"
  ITEM "IDL Bridge"
  END
END
