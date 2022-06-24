HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	10/16/2018 08:26.15
VERSION 1
END

WID_BASE_Save_Data_ASCII BASE 5 5 535 362
REALIZE "Initialize_Save_Data_ASCII"
TLB
CAPTION "Save Data into Tab Delimited ASCII file"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_TEXT_ASCII_Filename TEXT 6 40 511 44
  EDITABLE
  WRAP
  ALLEVENTS
  ONINSERTCHAR "On_ASCII_Filename_change"
  ONINSERTSTRING "On_ASCII_Filename_change"
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_ASCII_Save_Parameter_List TEXT 9 195 513 57
  SCROLL
  EDITABLE
  ALLEVENTS
  ONINSERTCHAR "On_Save_ASCII_ParamList_change"
  ONINSERTSTRING "On_Save_ASCII_ParamList_change"
  WIDTH = 20
  HEIGHT = 1
  END
  WID_DROPLIST_Save_ASCII_Filter DROPLIST 21 95 170 30
  CAPTION "Filter"
  NUMITEMS = 2
  ITEM "Frame Peaks"
  ITEM "Grouped Peaks"
  ONSELECT "On_Select_SaveASCII_Filter"
  END
  WID_DROPLIST_Save_ASCII_XY DROPLIST 306 95 170 30
  CAPTION "X-Y  Coord. Units"
  NUMITEMS = 2
  ITEM "Pixels"
  ITEM "nm"
  ONSELECT "On_Select_SaveASCII_units"
  END
  WID_DROPLIST_Save_ASCII_Parameters DROPLIST 98 140 315 30
  CAPTION "Save Parameters"
  NUMITEMS = 3
  ITEM "All"
  ITEM "From the list below"
  ITEM "From the list below into binary .sav"
  ONSELECT "On_Select_SaveASCII_ParamChoice"
  END
  WID_BUTTON_Save_ASCII PUSHBUTTON 70 269 150 35
  VALUE "Save"
  ALIGNCENTER
  ONACTIVATE "Save_ASCII"
  END
  WID_BUTTON_Cancel_Save_ASCII PUSHBUTTON 276 269 150 35
  VALUE "Close"
  ALIGNCENTER
  ONACTIVATE "OnCancel_SAVE_ASCII"
  END
  WID_LABEL_Save_ASCII_FileName_Text LABEL 5 23 519 18
  VALUE "Edit / Enter in the field below the filename (including file ext.) where you want to save the data"
  ALIGNLEFT
  END
  WID_LABEL_SAVE_ASCII_ParamList_Explanation LABEL 7 176 515 17
  VALUE "Enter (space separated) the indecis of the parameters that you want to have saved"
  ALIGNLEFT
  END
END
