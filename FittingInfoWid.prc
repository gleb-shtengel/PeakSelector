HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	02/19/2014 15:17.36
VERSION 1
END

WID_BASE_Info BASE 5 5 330 736
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
  WID_BUTTON_Info_OK PUSHBUTTON 171 642 140 55
  VALUE "Confirm and Start Fit"
  ALIGNCENTER
  ONACTIVATE "OnInfoOK"
  END
  WID_DROPLIST_FitDisplayType DROPLIST 38 530 225 30
  REALIZE "DoRealizeDropListDispType"
  CAPTION "Fit-Display Level"
  NUMITEMS = 5
  ITEM "No Display"
  ITEM "Some Frames/Peaks"
  ITEM "All Frames/Peaks "
  ITEM "Cluster - No Display"
  ITEM "IDL Bridge - No Disp"
  END
  WID_DROPLIST_SetSigmaFitSym DROPLIST 40 565 225 30
  CAPTION "SetSigmaFitSymmetry"
  NUMITEMS = 2
  ITEM "R"
  ITEM "X Y"
  ONSELECT "SetSigmaFitSym"
  END
  WID_BUTTON_CancelFit PUSHBUTTON 11 642 140 55
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancelFit"
  END
  WID_DROPLIST_Localization_Method DROPLIST 43 605 225 30
  CAPTION "Localization method"
  NUMITEMS = 2
  ITEM "Gaussian Fit"
  ITEM "Sparse Sampling"
  ONSELECT "SetLocalizationMethod"
  END
END
