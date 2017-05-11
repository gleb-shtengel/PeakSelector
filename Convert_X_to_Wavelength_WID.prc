HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	04/17/2015 10:35.37
VERSION 1
END

WID_BASE_Convert_X_to_Wavelength BASE 5 5 399 203
REALIZE "Initialize_Convert_X_to_Wavelength"
TLB
CAPTION "Calculate Wavelength from X-coordinate"
MODAL
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_Convert_X_to_Wavelength PUSHBUTTON 10 113 200 36
  VALUE "Recalculate CGroupParams[12,*]"
  ALIGNCENTER
  ONACTIVATE "OnRecalculate_Wavelength"
  END
  WID_BUTTON_Cancel PUSHBUTTON 250 113 112 36
  VALUE "Cancel"
  ALIGNCENTER
  ONACTIVATE "OnCancel"
  END
  WID_TEXT_offset_value TEXT 300 55 70 32
  NUMITEMS = 1
  ITEM "0.0"
  EDITABLE
  ALLEVENTS
  ONINSERTCHAR "Change_offset_text"
  WIDTH = 20
  HEIGHT = 1
  END
  WID_LABEL_Offset LABEL 240 63 48 20
  VALUE "Offset"
  ALIGNLEFT
  END
  WID_LABEL_Wavelength_Conversion LABEL 10 18 360 23
  VALUE "CGroupParams[12,*] = CGroupParams[2,*] * Dispersion + Offset"
  ALIGNLEFT
  END
  WID_LABEL_Dispersion LABEL 10 60 120 20
  VALUE "Dispersion (nm/pix)"
  ALIGNLEFT
  END
  WID_TEXT_dispersion_value TEXT 130 55 70 32
  NUMITEMS = 1
  ITEM "0.0"
  EDITABLE
  ALLEVENTS
  ONINSERTCHAR "Change_dispersion_text"
  WIDTH = 20
  HEIGHT = 1
  END
END
