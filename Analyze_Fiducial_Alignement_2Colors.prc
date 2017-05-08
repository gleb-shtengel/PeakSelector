HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	01/22/2013 11:21.51
VERSION 1
END

WID_BASE_AnalyzeMultipleFiducials_2Colors BASE 5 5 383 373
REALIZE "Initialize_AnalizeMultiple_Fiducials_2Colors"
TLB
CAPTION "Analyze Fiducial Co-Localization ofr  2 Colors"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_Analyze_Fiducial_colocalization_2color_OK PUSHBUTTON 250 305 120 30
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "On_AnalyzeMultiple_Fiducial_Colocalization_Start"
  END
  WID_LABEL_Xrange LABEL 40 109 85 15
  VALUE "X Range (pixels)"
  ALIGNLEFT
  END
  WID_TEXT_Xrange1 TEXT 149 104 100 30
  NUMITEMS = 1
  ITEM "1.5"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_Yrange1 TEXT 149 139 100 30
  NUMITEMS = 1
  ITEM "1.5"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_YRange LABEL 40 144 85 15
  VALUE "Y Range (pixels)"
  ALIGNLEFT
  END
  WID_LABEL_Status LABEL 20 275 35 15
  VALUE "Status"
  ALIGNLEFT
  END
  WID_TEXT_Status_Fid2Color_Analysis TEXT 75 264 280 35
  NUMITEMS = 1
  ITEM "Press "
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_FidFilename TEXT 26 49 331 48
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_PickFidFile PUSHBUTTON 205 12 140 30
  VALUE "Pick *.anc File"
  ALIGNCENTER
  ONACTIVATE "OnPickFidFile"
  END
  WID_LABEL_anc_explanation LABEL 10 14 177 23
  VALUE "Use *.anc file to list fiducial peaks"
  ALIGNLEFT
  END
  WID_LABEL_ZRange LABEL 40 178 85 15
  VALUE "Z Range (nm)"
  ALIGNLEFT
  END
  WID_TEXT_ZRange1 TEXT 149 174 100 30
  NUMITEMS = 1
  ITEM "200"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BASE_0 BASE 2 312 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_use_green PUSHBUTTON -1 -1 0 0
    VALUE "Use Green Fiducials (Red if not selected)"
    ALIGNLEFT
    END
  END
  WID_LABEL_MinNumber LABEL 38 226 85 15
  VALUE "Min # of peaks"
  ALIGNLEFT
  END
  WID_TEXT_MiNumberofPeaks TEXT 149 215 100 30
  NUMITEMS = 1
  ITEM "50"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
END
