HEADER
; IDL Visual Widget Builder Resource file. Version 1
; Generated on:	07/10/2017 14:14.16
VERSION 1
END

WID_BASE_SaveCustomTIFF BASE 5 5 1446 1071
REALIZE "Initialize_Custom_TIFF"
TLB
CAPTION "Save Custom TIFF files"
XPAD = 3
YPAD = 3
SPACE = 3
BEGIN
  WID_BUTTON_Save_Separate_TIFFs PUSHBUTTON 51 820 280 30
  VALUE "Save Volume as Separate TIFF files"
  ALIGNCENTER
  ONACTIVATE "Save_Volume_TIFF_separate_files"
  END
  WID_DRAW_Custom_TIFF DRAW 380 0 1024 1024
  FRAME = 1
  REALIZE "CustomTIFF_Draw_Realize"
  SCROLLWIDTH = 1024
  SCROLLHEIGHT = 1024
  END
  WID_IMAGE_SCALING_Parameters TABLE 21 340 335 110
  RESIZECOLUMNS
  N_ROWS = 3
  N_COLS = 1
  NUMCOLLABELS = 1
  COLLABEL "Value"
  NUMROWLABELS = 3
  ROWLABEL "NM per Image Pixel"
  ROWLABEL "Total Image Pixels X"
  ROWLABEL "Total Image Pixels Y"
  EDITABLE
  ONINSERTCHAR "DoInsert_Cust_TIFF_Scale_Param"
  END
  WID_BUTTON_Render_cust_TIFF PUSHBUTTON 20 105 140 30
  VALUE "Render"
  ALIGNCENTER
  ONACTIVATE "Render_cust_TIFF"
  END
  WID_DROPLIST_Accumulate_cust_TIFF DROPLIST 10 75 170 25
  CAPTION "Accumulation"
  NUMITEMS = 2
  ITEM "Envelope"
  ITEM "Sum"
  ONSELECT "Cust_TIFF_Select_Accumulation"
  END
  WID_DROPLIST_Filter_cust_TIFF DROPLIST 10 40 170 25
  CAPTION "Filter"
  NUMITEMS = 2
  ITEM "Frame Peaks"
  ITEM "Grouped Peaks"
  ONSELECT "Cust_TIFF_Select_Filter"
  END
  WID_DROPLIST_Function_cust_TIFF DROPLIST 10 5 170 25
  CAPTION "Function"
  NUMITEMS = 3
  ITEM "Center Locations"
  ITEM "Gaussian Normalized"
  ITEM "Gaussian Amplitude"
  ONSELECT "Cust_TIFF_Select_Function"
  END
  WID_SLIDER_Bot_cust_TIFF SLIDER 204 156 160 46
  CAPTION "Stretch Bottom"
  MAXIMUM = 1000
  ONVALUECHANGED "OnStretchBottom_cust_TIFF"
  END
  WID_SLIDER_Gamma_cust_TIFF SLIDER 204 101 160 46
  CAPTION "Gamma"
  VALUE = 500
  MAXIMUM = 1000
  ONVALUECHANGED "OnGamma_cust_TIFF"
  END
  WID_SLIDER_Top_cust_TIFF SLIDER 205 46 160 46
  CAPTION "Stretch Top"
  VALUE = 500
  MAXIMUM = 1000
  ONVALUECHANGED "OnStretchTop_cust_TIFF"
  END
  WID_DROPLIST_Label_cust_TIFF DROPLIST 250 5 100 25
  CAPTION "Label"
  NUMITEMS = 5
  ITEM ""
  ITEM "Red"
  ITEM "Green"
  ITEM "Blue"
  ITEM "DIC / EM"
  ONSELECT "OnLabelDropList_cust_TIFF"
  END
  WID_BUTTON_Save_cust_TIFF_0 PUSHBUTTON 30 260 140 30
  VALUE "Save TIFF"
  ALIGNCENTER
  ONACTIVATE "Save_cust_TIFF"
  END
  WID_BUTTON_ScaleBar_cust_TIFF PUSHBUTTON 222 251 130 30
  VALUE "Add Scale Bar1"
  ALIGNCENTER
  ONACTIVATE "OnAddScaleBarButton_cust_TIFF"
  END
  WID_BUTTON_Generate3D PUSHBUTTON 81 640 193 30
  VALUE "Generate 3D Volume"
  ALIGNCENTER
  ONACTIVATE "On_Generate3D"
  END
  WID_SLIDER_Z_slice SLIDER 16 690 350 55
  CAPTION "Z slice #"
  VALUE = 50
  MAXIMUM = 100
  ONVALUECHANGED "Display_Zslice"
  END
  WID_BUTTON_Save_Multiframe_TIFF PUSHBUTTON 51 770 280 30
  VALUE "Save Volume as Multi-frame TIFF file"
  ALIGNCENTER
  ONACTIVATE "Save_Volume_TIFF"
  END
  WID_IMAGE_Zcoord_Parameters TABLE 21 460 335 125
  N_ROWS = 4
  N_COLS = 1
  NUMCOLLABELS = 1
  COLLABEL "Value"
  NUMROWLABELS = 4
  ROWLABEL "Z start (nm)"
  ROWLABEL "Z stop (nm)"
  ROWLABEL "Z step (nm)"
  ROWLABEL "Z - X scaling"
  EDITABLE
  ONINSERTCHAR "DoInsert_Cust_TIFF_ZScale_Param"
  END
  WID_TEXT_Zsubvolume TEXT 281 590 70 30
  NUMITEMS = 2
  ITEM "100.0"
  ITEM ""
  EDITABLE
  WRAP
  ALLEVENTS
  ONINSERTCHAR "Change_Subvolume"
  ONINSERTSTRING "Change_Subvolume"
  WIDTH = 20
  HEIGHT = 2
  END
  WID_LABEL_subvolume_txt LABEL 21 600 250 15
  VALUE "Gaussian Cloud Radius (subvolume) (nm)"
  ALIGNLEFT
  END
  WID_BUTTON_Save_Separate_PNGs PUSHBUTTON 51 870 280 30
  VALUE "Save Volume as Separate PNG files"
  ALIGNCENTER
  ONACTIVATE "Save_Volume_PNG_separate_files"
  END
  WID_BUTTON_Overlay_DIC_EM_cust_TIFF PUSHBUTTON 20 180 140 30
  VALUE "Overlay DIC/EM"
  ALIGNCENTER
  ONACTIVATE "Overlay_DIC_cust_TIFF"
  END
  WID_BUTTON_Save_Multiframe_Monochrome_TIFF PUSHBUTTON 20 920 340 30
  VALUE "Save Volume as Monochrome Multi-frame TIFF stack"
  ALIGNCENTER
  ONACTIVATE "Save_Volume_TIFF_Monochrome"
  END
  WID_BUTTON_TotalRawData_cust PUSHBUTTON 20 140 140 30
  VALUE "Total Raw Data"
  ALIGNCENTER
  ONACTIVATE "OnTotalRawDataButton_cust"
  END
  WID_BUTTON_Render_cust_DIC PUSHBUTTON 20 220 140 30
  VALUE "Render Only DIC/EM"
  ALIGNCENTER
  ONACTIVATE "Draw_DIC_only_cust_TIFF"
  END
  WID_BASE_Tie_RGB_CustTIFF BASE 236 211 0 0
  COLUMNS = 1
  NONEXCLUSIVE
  CAPTION "IDL"
  BEGIN
    WID_BUTTON_Tie_RGB_CustTIFF PUSHBUTTON -1 -1 0 0
    VALUE "Tie RGB"
    ALIGNLEFT
    ONACTIVATE "Set_Tie_RGB_CustTIFF"
    END
  END
  WID_BUTTON_Save_cust_TIFF_float PUSHBUTTON 30 300 140 30
  VALUE "Save TIFF float"
  ALIGNCENTER
  ONACTIVATE "Save_cust_TIFF_float"
  END
END
