HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	10/03/2011 13:29.06
VERSION 1
END

WID_BASE_SaveZslicesTIFF BASE 5 5 357 283
REALIZE "Initialize_ZslicesTIFF"
TLB
CAPTION "Save Z-slices into Individual TIFF files"
MODAL
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_DROPLIST_Normalization DROPLIST 20 12 250 22
  CAPTION "Image Normalization"
  NUMITEMS = 2
  ITEM "Compound Image (fast)"
  ITEM "Run Z-slices twice (slow)"
  END
  WID_BUTTON_ZslicesTIFF_OK PUSHBUTTON 82 220 193 30
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "On_Zslices_TIFF_Start"
  END
  WID_LABEL_Zstart LABEL 60 50 75 15
  VALUE "Z start (nm)"
  ALIGNLEFT
  END
  WID_TEXT_Zstart TEXT 150 45 100 30
  NUMITEMS = 1
  ITEM "0.00"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_TEXT_Zstop TEXT 150 85 100 30
  NUMITEMS = 1
  ITEM "250.00"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_Zstop LABEL 60 90 75 15
  VALUE "Z stop (nm)"
  ALIGNLEFT
  END
  WID_LABEL_Zstep LABEL 60 130 75 15
  VALUE "Z step (nm)"
  ALIGNLEFT
  END
  WID_TEXT_Zstep TEXT 150 125 100 30
  NUMITEMS = 1
  ITEM "5.00"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_Status LABEL 10 177 35 15
  VALUE "Status"
  ALIGNLEFT
  END
  WID_TEXT_Status TEXT 59 170 286 35
  NUMITEMS = 1
  ITEM "Press "
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
END
