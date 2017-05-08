HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	10/21/2011 09:35.53
VERSION 1
END

WID_BASE_Import_Data_ASCII BASE 5 5 535 412
REALIZE "Initialize_Import_Data_ASCII"
TLB
CAPTION "Import Data from Tab Delimited ASCII file"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_TEXT_Import_ASCII_Filename TEXT 9 40 366 44
  EDITABLE
  WRAP
  ALLEVENTS
  ONINSERTCHAR "On_ASCII_Filename_change"
  ONINSERTSTRING "On_ASCII_Filename_change"
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_ASCII_Import_Parameter_List TEXT 9 195 513 57
  SCROLL
  EDITABLE
  ALLEVENTS
  ONINSERTCHAR "On_Import_ASCII_ParamList_change"
  ONINSERTSTRING "On_Import_ASCII_ParamList_change"
  WIDTH = 20
  HEIGHT = 1
  END
  WID_DROPLIST_Import_ASCII_XY DROPLIST 24 103 170 30
  CAPTION "X-Y  Coord. Units"
  NUMITEMS = 2
  ITEM "Pixels"
  ITEM "nm"
  ONSELECT "On_Select_ImportASCII_units"
  END
  WID_BUTTON_Import_ASCII PUSHBUTTON 68 320 150 35
  VALUE "Import"
  ALIGNCENTER
  ONACTIVATE "Import_ASCII"
  END
  WID_BUTTON_Cancel_Import_ASCII PUSHBUTTON 274 320 150 35
  VALUE "Close"
  ALIGNCENTER
  ONACTIVATE "OnCancel_Import_ASCII"
  END
  WID_LABEL_Import_ASCII_FileName_Text LABEL 8 13 519 18
  VALUE "Pick / Edit / Enter in the field below the filename (including file ext.) of the data file"
  ALIGNLEFT
  END
  WID_LABEL_Import_ASCII_ParamList_Explanation1 LABEL 5 140 515 20
  VALUE "Enter the PeakSelector parameter indecis corresponding to the columns in your data file."
  ALIGNLEFT
  END
  WID_BUTTON_PickASCIIFile PUSHBUTTON 395 44 120 30
  VALUE "Pick ASCII File"
  ALIGNCENTER
  ONACTIVATE "OnPickASCIIFile"
  END
  WID_LABEL_Import_ASCII_ParamList_Explanation1_0 LABEL 5 162 515 20
  VALUE "If the column in your data file does not have a corresponding PeakSelector parameter, enter -1."
  ALIGNLEFT
  END
  WID_TEXT_Import_ASCII_nm_per_pixel TEXT 424 98 70 30
  NUMITEMS = 1
  ITEM "133.33"
  EDITABLE
  WRAP
  ALLEVENTS
  ONINSERTCHAR "Change_Import_ASCII_nm_per_pixel"
  ONINSERTSTRING "Change_Import_ASCII_nm_per_pixel"
  WIDTH = 20
  HEIGHT = 2
  END
  WID_BASE_SkipFirstLine_ImportASCII BASE 55 270 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_SkipFirstLine_ImportASCII PUSHBUTTON -1 -1 0 0
    VALUE "Skip First Line (check if titles line is present)"
    ALIGNLEFT
    END
  END
END
