HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	11/11/2021 08:51.40
VERSION 1
END

WID_BASE_iPALM_Split_ZSlabs_into_Separate_Files BASE 5 5 552 386
REALIZE "Initialize_SplitZslabs_Wid"
TLB
CAPTION "Split Data into Separate Z-Slab Files"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_SplitZSlabsOK PUSHBUTTON 364 249 129 46
  VALUE "Confirm and Start"
  ALIGNCENTER
  ONACTIVATE "OnSplitZSlabsOK"
  END
  WID_SLIDER_FramesPerZSlab SLIDER 50 250 250 50
  CAPTION "Frames per Z-Slab"
  VALUE = 250
  MINIMUM = 1
  MAXIMUM = 1000
  END
  WID_SLIDER_Number_of_ZSlabs SLIDER 50 150 250 50
  CAPTION "Number of Z-Slabs (Sample Piezo Z-levels)"
  VALUE = 5
  MINIMUM = 1
  MAXIMUM = 25
  END
  WID_BUTTON_PickFilePrefix PUSHBUTTON 400 40 100 30
  VALUE "Pick File Prefix"
  ALIGNCENTER
  ONACTIVATE "OnPickFilePrefix"
  END
  WID_TEXT_FilePrefix TEXT 20 40 350 40
  NUMITEMS = 1
  ITEM "Select *.anc File"
  EDITABLE
  WRAP
  WIDTH = 20
  HEIGHT = 2
  END
END
