HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	07/23/2009 06:46.15
VERSION 1
END

WID_BASE_Recalc_PeakW_Product BASE 5 5 399 203
REALIZE "Initialize_Recalculate_Menu"
TLB
CAPTION "Recalculate the Peakwidth Product"
MODAL
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_Recalculate PUSHBUTTON 10 113 200 36
  VALUE "Recalculate CGroupParams[12,*]"
  ALIGNCENTER
  ONACTIVATE "OnRecalculate"
  END
  WID_BUTTON_Cancel PUSHBUTTON 250 113 112 36
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancel"
  END
  WID_TEXT_offset_value TEXT 180 57 70 32
  NUMITEMS = 1
  ITEM "0.0"
  EDITABLE
  WIDTH = 20
  HEIGHT = 1
  END
  WID_LABEL_Offset_V LABEL 130 65 48 21
  VALUE "Offset"
  ALIGNLEFT
  END
  WID_LABEL_0 LABEL 1 22 389 23
  VALUE "CGroupParams[12,*] = (CGroupParams[4,*] - Offset) * (CGroupParams[5,*] - Offset)"
  ALIGNLEFT
  END
END
