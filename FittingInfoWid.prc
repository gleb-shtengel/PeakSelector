HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	09/15/2017 14:11.30
VERSION 1
END

WID_BASE_Info BASE 5 5 330 900
REALIZE "DoRealizeInfo"
TLB
CAPTION "Fitting Info"
MODAL
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_TABLE_InfoFile TABLE 29 11 260 510
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
  ONINSERTCHAR "DoInsertInfo"
  END
  WID_BUTTON_Info_OK PUSHBUTTON 167 793 140 55
  VALUE "Confirm and Start Fit"
  ALIGNCENTER
  ONACTIVATE "OnInfoOK"
  END
  WID_DROPLIST_FitDisplayType DROPLIST 5 530 300 30
  REALIZE "DoRealizeDropListDispType"
  CAPTION "Fit-Display Level"
  NUMITEMS = 5
  ITEM "No Display"
  ITEM "Some Frames/Peaks"
  ITEM "All Frames/Peaks "
  ITEM "Cluster - No Display"
  ITEM "IDL Bridge - No Disp"
  END
  WID_DROPLIST_SetSigmaFitSym DROPLIST 5 610 300 30
  CAPTION "Gaussian Fit"
  NUMITEMS = 3
  ITEM "R"
  ITEM "X Y unconstrained"
  ITEM "X Y constr: SigX(Z), SigY(Z)"
  ONSELECT "SetSigmaFitSym"
  END
  WID_BUTTON_CancelFit PUSHBUTTON 7 793 140 55
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancelFit"
  END
  WID_DROPLIST_Localization_Method DROPLIST 5 570 300 30
  CAPTION "Localization method"
  NUMITEMS = 2
  ITEM "Gaussian Fit"
  ITEM "Sparse Sampling"
  ONSELECT "SetLocalizationMethod"
  END
  WID_BUTTON_PickCalFile_FittingInfo PUSHBUTTON 5 645 150 32
  VALUE "Pick CAL (WND) File"
  ALIGNCENTER
  ONACTIVATE "OnPickCalFile_Astig_FittingInfo"
  END
  WID_TEXT_WindFilename_Astig__FittingInfo TEXT 5 680 300 80
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
END
