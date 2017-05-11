HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	02/21/2012 20:01.00
VERSION 1
END

WID_BASE_Transform_2D_and_Spectrum BASE 5 5 998 743
REALIZE "Initialize_Transform_2D_and_spectrum"
TLB
CAPTION "Transform Data and process 2D and Spectrum analysis"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_TEXT_XY_Filename TEXT 5 20 500 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_TrnSpFilename TEXT 5 70 500 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_XY_File LABEL 5 5 0 0
  VALUE "X-Y Localization File:"
  ALIGNLEFT
  END
  WID_LABEL_TrnSpFile LABEL 5 55 0 0
  VALUE "Transformed Spectrum File:"
  ALIGNLEFT
  END
  WID_BUTTON_stop PUSHBUTTON 660 133 150 55
  VALUE "Stop"
  ALIGNCENTER
  ONACTIVATE "OnStopSpectralProcessing"
  END
  WID_BUTTON_Process PUSHBUTTON 820 12 160 55
  VALUE "Process Spectra"
  ALIGNCENTER
  ONACTIVATE "Start_Process_2DSpectrum"
  END
  WID_DROPLIST_FitDisplayType_Spectrum DROPLIST 637 81 225 30
  CAPTION "Fit-Display Level"
  NUMITEMS = 4
  ITEM "No Display"
  ITEM "Some Frames/Peaks"
  ITEM "All Frames/Peaks "
  ITEM "Cluster - No Display"
  END
  WID_BUTTON_CreateSingleCal PUSHBUTTON 245 210 200 30
  VALUE "Create Single Calibration"
  ALIGNCENTER
  ONACTIVATE "Create_Single_Sp_Cal"
  END
  WID_TEXT_SpCalFilename TEXT 5 150 600 32
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_TrnSpFile_0 LABEL 7 130 0 0
  VALUE "Calibration File:"
  ALIGNLEFT
  END
  WID_BUTTON_PickCalFile PUSHBUTTON 460 115 180 32
  VALUE "Pick Calibration File"
  ALIGNCENTER
  ONACTIVATE "OnPickSpCalFile"
  END
  WID_SP_Cal_Table TABLE 10 280 432 430
  N_ROWS = 200
  N_COLS = 5
  NUMCOLLABELS = 5
  COLLABEL "Sp #1"
  COLLABEL "Sp #2"
  COLLABEL "Sp #3"
  COLLABEL "Sp #4"
  COLLABEL "Sp #5"
  EDITABLE
  ONINSERTCHAR "Do_Edit_Sp_Cal"
  END
  WID_DRAW_Spectra DRAW 460 280 440 430
  END
  WID_BUTTON_SaveCalFile PUSHBUTTON 460 185 180 32
  VALUE "Save Calibration File"
  ALIGNCENTER
  ONACTIVATE "OnSaveSpCalFile"
  END
  WID_BUTTON_RemoveSingleCal_0 PUSHBUTTON 5 210 200 30
  VALUE "Remove Single Calibration"
  ALIGNCENTER
  ONACTIVATE "Remove_Single_Sp_Cal"
  END
  WID_TEXT_CalSpNum TEXT 210 210 30 30
  NUMITEMS = 1
  ITEM "0"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TABLE_WlCal_FrStart_FrStop TABLE 660 199 154 77
  N_ROWS = 2
  N_COLS = 1
  NUMROWLABELS = 2
  ROWLABEL "Start Frame"
  ROWLABEL "Stop Frame"
  EDITABLE
  ONINSERTCHAR "Do_Edit_Cal_Frames"
  END
  WID_BUTTON_WlShiftSingleCal_1 PUSHBUTTON 125 245 200 30
  VALUE "Shift Single Calibration (pix)"
  ALIGNCENTER
  ONACTIVATE "WlShift_Single_Sp_Cal"
  END
  WID_TEXT_WlShift TEXT 330 245 30 30
  NUMITEMS = 1
  ITEM "0"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BASE_0 BASE 454 236 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_TransformFirstForCal PUSHBUTTON -1 -1 0 0
    VALUE "Transform First (For Single Cal)"
    ALIGNLEFT
    END
  END
  WID_BUTTON_Transform PUSHBUTTON 650 12 160 55
  VALUE "Confirm + Transform"
  ALIGNCENTER
  ONACTIVATE "StartTransform"
  END
  WID_BUTTON_CancelReExtract_0 PUSHBUTTON 825 132 150 55
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancelSave"
  END
  WID_BTTN_Pick_XY_File PUSHBUTTON 520 20 100 30
  VALUE "Pick XY File"
  ALIGNCENTER
  ONACTIVATE "OnPickXYTxtFile"
  END
  WID_BTTN_Pick_SP_File PUSHBUTTON 520 70 100 30
  VALUE "Pick SP File"
  ALIGNCENTER
  ONACTIVATE "OnPickSPTxtFile"
  END
  WID_BUTTON_PlotDistributions PUSHBUTTON 825 207 150 55
  VALUE "Plot Distributions"
  ALIGNCENTER
  ONACTIVATE "Plot_Spectral_Weigths_Distributions"
  END
END
