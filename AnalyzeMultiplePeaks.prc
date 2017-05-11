HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	07/01/2011 09:10.30
VERSION 1
END

WID_BASE_AnalyzeMultiplePeaks BASE 5 5 566 555
REALIZE "Initialize_AnalizeMultiplePeaks"
TLB
CAPTION "Analyze Multiple Peaks"
MODAL
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_AnalyzeMultiple_OK PUSHBUTTON 79 460 100 30
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "On_AnalyzeMultiple_Start"
  END
  WID_LABEL_Xrange LABEL 48 110 85 15
  VALUE "X Range (pixels)"
  ALIGNLEFT
  END
  WID_TEXT_Xrange TEXT 149 100 100 30
  NUMITEMS = 1
  ITEM "1.5"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_Yrange TEXT 149 135 100 30
  NUMITEMS = 1
  ITEM "1.5"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_YRange LABEL 49 145 85 15
  VALUE "Y Range (pixels)"
  ALIGNLEFT
  END
  WID_LABEL_Status LABEL 189 470 35 15
  VALUE "Status"
  ALIGNLEFT
  END
  WID_TEXT_Status TEXT 224 460 190 30
  NUMITEMS = 1
  ITEM "Press "
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_PeaksFilename TEXT 26 49 331 48
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BUTTON_PickPeaksFile PUSHBUTTON 200 12 150 30
  VALUE "Pick *.anc File"
  ALIGNCENTER
  ONACTIVATE "OnPickPeaksFile"
  END
  WID_LABEL_anc_explanation LABEL 10 14 177 23
  VALUE "Use *.anc file to list peaks"
  ALIGNLEFT
  END
  WID_LABEL_MinNpeaks LABEL 48 180 85 15
  VALUE "Min # of Peaks"
  ALIGNLEFT
  END
  WID_TEXT_MinNPeaks TEXT 149 170 100 30
  NUMITEMS = 1
  ITEM "200"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_DROPLIST_Analyze_Multiple_Filter DROPLIST 120 220 170 30
  CAPTION "Filter"
  NUMITEMS = 2
  ITEM "Frame Peaks"
  ITEM "Grouped Peaks"
  ONSELECT "On_Select_Peaks_Filter"
  END
  WID_LABEL_SAVE_ASCII_ParamList_Explanation LABEL 72 349 515 17
  VALUE "Enter (space separated) the indecis of the parameters that you want to have saved"
  ALIGNLEFT
  END
  WID_DROPLIST_Save_Peak_ASCII_Parameters DROPLIST 163 313 315 30
  CAPTION "Save Parameters"
  NUMITEMS = 2
  ITEM "All"
  ITEM "From the list below"
  ONSELECT "On_Select_Peak_SaveASCII_ParamChoice"
  END
  WID_DROPLIST_Save_Peak_ASCII_XY DROPLIST 371 268 170 30
  CAPTION "X-Y  Coord. Units"
  NUMITEMS = 2
  ITEM "Pixels"
  ITEM "nm"
  ONSELECT "On_Select_Peak_SaveASCII_units"
  END
  WID_TEXT_ASCII_Peak_Save_Parameter_List TEXT 16 368 513 57
  SCROLL
  EDITABLE
  ALLEVENTS
  ONINSERTCHAR "On_ASCII_Peak_ParamList_change"
  ONINSERTSTRING "On_ASCII_Peak_ParamList_change"
  WIDTH = 20
  HEIGHT = 1
  END
  WID_BASE_Save_Each_Peak_Distribution BASE 20 267 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Save_Each_Peak_Distribution PUSHBUTTON -1 -1 0 0
    VALUE "Save Each Peak Distribution into Sepearte ASCII file"
    ALIGNLEFT
    END
  END
END
