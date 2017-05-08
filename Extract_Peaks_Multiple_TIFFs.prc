HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	05/08/2017 14:53.54
VERSION 1
END

WID_BASE_Extract_Peaks_Multiple_TIFFs BASE 5 5 749 870
REALIZE "Initialize_Extract_Peaks_Multiple_TIFFs"
TLB
CAPTION "Extract Peaks from multiple TIFF files in a single directory"
MODAL
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_StartMLExtract PUSHBUTTON 570 720 150 40
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "Start_iPALM_Macro"
  END
  WID_BUTTON_CancelReExtract PUSHBUTTON 570 780 150 40
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCanceliPALM_Macro"
  END
  WID_DROPLIST_SetSigmaFitSym_mTIFFS DROPLIST 280 190 175 30
  CAPTION "SetSigmaFitSymmetry"
  NUMITEMS = 2
  ITEM "R"
  ITEM "X Y"
  ONSELECT "Set_SigmaFitSym_mTIFFS"
  END
  WID_BTTN_SelectDirectory PUSHBUTTON 509 10 200 30
  VALUE "Select Directory"
  ALIGNCENTER
  ONACTIVATE "On_Select_Directory"
  END
  WID_TXT_mTIFFS_Directory TEXT 4 10 500 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
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
  WID_DROPLIST_TransformEngine_mTIFFS DROPLIST 10 190 225 30
  CAPTION "Transformation Engine"
  NUMITEMS = 3
  ITEM "Local"
  ITEM "Cluster"
  ITEM "IDL Bridge"
  ONSELECT "Set_TransformEngine_mTIFFS"
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
  WID_TABLE_InfoFile__mTIFFS TABLE 10 230 260 540
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
  ONINSERTCHAR "DoInsertInfo_mTIFFS"
  END
END
