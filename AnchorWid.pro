; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	04/15/2013 12:03.58
; 
pro WID_BASE_AnchorPts_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_AnchorPts'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_Anchors_XY_Table'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        DoInsertAnchor, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_0'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Do_RGB_Transforms, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_AddRed'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_AddRedFiducial, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_AddGreen'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_AddGreenFiducial, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_AddBlue'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_AddBlueFiducial, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_2'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        ClearFiducials, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_RTG'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        RGB_check_buttons, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_RTB'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        RGB_check_buttons, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_GTB'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        RGB_check_buttons, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_GTR'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        RGB_check_buttons, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_BTR'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        RGB_check_buttons, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_BTG'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        RGB_check_buttons, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PickANCFile'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPickANCFile, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Load_ANC'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Load_ANC_File, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_ANC'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Save_ANC_File, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_TRANSFORM_METHOD'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        Set_Transf_Method, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_Anchors_Z_Table'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        DoInsertZAnchor, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_DisplayFiducials'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Display_RGB_fiducials, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_AutoDisplay_Selected_Fiducials'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_AutoDisplay_Selected_Fiducials, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Set_Fid_outline_size'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        SetFiducialOutlineSize, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_RemoveFiducial'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_RemoveFiducial, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_AutoDetect'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_AutodetectRedFiducials, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_AutoDetect_Parameters'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        DoInsert_Autodetect_Param, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_AutoDetect_0'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_AutodetectMatchingFiducials, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_AutoDetect_Match_Parameters'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        DoInsert_Autodetect_Matching_Param, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Test_Transformation'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        TestFiducialTransformation, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Display_Fiducial_IDs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_Display_Fiducial_IDs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Remove_Unmatched'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_Remove_Unmatched, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_POLYWARP_Degree'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_SLIDER' )then $
        Set_PW_deg, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Adj_Scl'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_AdjustScale, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Remove_Bad_Fiducials'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_Remove_Bad_Fiducials, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Swap_RedGreen'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_SwapRedGreen, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Swap_RedBlue'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_SwapRedBlue, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_XYlimits'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_LimitXY, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_XYlimits_table'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        Set_XY_limits, Event
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_STR' )then $
        Set_XY_limits, Event
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_DEL' )then $
        Set_XY_limits, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_LeaveOrigTotalRaw'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Set_LeaveOrigTotRaw, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_DisplayFiducials_with_overalys'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Display_RGB_fiducials_with_overlays, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Refind'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_RefindFiducials, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Copy_Red_to_Green'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_Copy_Red_to_Green, Event
    end
    else:
  endcase

end

pro WID_BASE_AnchorPts, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'AnchorWid_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_AnchorPts = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_AnchorPts' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=1068 ,SCR_YSIZE=906  $
      ,NOTIFY_REALIZE='InitializeFidAnchWid' ,TITLE='Fiducial Anchor'+ $
      ' Points' ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_Anchors_XY_Table = Widget_Table(WID_BASE_AnchorPts,  $
      UNAME='WID_Anchors_XY_Table' ,YOFFSET=365 ,SCR_XSIZE=476  $
      ,SCR_YSIZE=510 ,/EDITABLE ,COLUMN_LABELS=[ 'Red X', 'Red Y',  $
      'Green X', 'Green Y', 'Blue X', 'Blue Y' ] ,XSIZE=6 ,YSIZE=500)

  
  WID_BUTTON_0 = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_0' ,XOFFSET=490 ,YOFFSET=100 ,SCR_XSIZE=90  $
      ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Transform')

  
  WID_BUTTON_AddRed = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_AddRed' ,XOFFSET=130 ,YOFFSET=58  $
      ,SCR_XSIZE=95 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Add Red'+ $
      ' Fid.')

  
  WID_BUTTON_AddGreen = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_AddGreen' ,XOFFSET=240 ,YOFFSET=58  $
      ,SCR_XSIZE=95 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Add Green'+ $
      ' Fid.')

  
  WID_BUTTON_AddBlue = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_AddBlue' ,XOFFSET=350 ,YOFFSET=58  $
      ,SCR_XSIZE=95 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Add Blue'+ $
      ' Fid.')

  
  WID_BUTTON_2 = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_2' ,XOFFSET=15 ,YOFFSET=45 ,SCR_XSIZE=90  $
      ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Clear All')

  
  WID_LABEL_0 = Widget_Label(WID_BASE_AnchorPts, UNAME='WID_LABEL_0'  $
      ,XOFFSET=115 ,YOFFSET=93 ,SCR_XSIZE=368 ,SCR_YSIZE=19  $
      ,/ALIGN_LEFT ,VALUE='To add new fiducial point, first select'+ $
      ' the area of the interest on the image.')

  
  WID_BASE_0 = Widget_Base(WID_BASE_AnchorPts, UNAME='WID_BASE_0'  $
      ,XOFFSET=130 ,YOFFSET=130 ,TITLE='IDL' ,COLUMN=1  $
      ,/NONEXCLUSIVE)

  
  WID_BUTTON_RTG = Widget_Button(WID_BASE_0, UNAME='WID_BUTTON_RTG'  $
      ,/ALIGN_LEFT ,VALUE='Red to Green')

  
  WID_BASE_1 = Widget_Base(WID_BASE_AnchorPts, UNAME='WID_BASE_1'  $
      ,XOFFSET=130 ,YOFFSET=160 ,TITLE='IDL' ,COLUMN=1  $
      ,/NONEXCLUSIVE)

  
  WID_BUTTON_RTB = Widget_Button(WID_BASE_1, UNAME='WID_BUTTON_RTB'  $
      ,/ALIGN_LEFT ,VALUE='Red to Blue')

  
  WID_BASE_2 = Widget_Base(WID_BASE_AnchorPts, UNAME='WID_BASE_2'  $
      ,XOFFSET=240 ,YOFFSET=160 ,TITLE='IDL' ,COLUMN=1  $
      ,/NONEXCLUSIVE)

  
  WID_BUTTON_GTB = Widget_Button(WID_BASE_2, UNAME='WID_BUTTON_GTB'  $
      ,/ALIGN_LEFT ,VALUE='Green to Blue')

  
  WID_BASE_3 = Widget_Base(WID_BASE_AnchorPts, UNAME='WID_BASE_3'  $
      ,XOFFSET=240 ,YOFFSET=130 ,TITLE='IDL' ,COLUMN=1  $
      ,/NONEXCLUSIVE)

  
  WID_BUTTON_GTR = Widget_Button(WID_BASE_3, UNAME='WID_BUTTON_GTR'  $
      ,/ALIGN_LEFT ,VALUE='Green to Red')

  
  WID_BASE_4 = Widget_Base(WID_BASE_AnchorPts, UNAME='WID_BASE_4'  $
      ,XOFFSET=350 ,YOFFSET=130 ,TITLE='IDL' ,COLUMN=1  $
      ,/NONEXCLUSIVE)

  
  WID_BUTTON_BTR = Widget_Button(WID_BASE_4, UNAME='WID_BUTTON_BTR'  $
      ,/ALIGN_LEFT ,VALUE='Blue to Red')

  
  WID_BASE_5 = Widget_Base(WID_BASE_AnchorPts, UNAME='WID_BASE_5'  $
      ,XOFFSET=350 ,YOFFSET=160 ,TITLE='IDL' ,COLUMN=1  $
      ,/NONEXCLUSIVE)

  
  WID_BUTTON_BTG = Widget_Button(WID_BASE_5, UNAME='WID_BUTTON_BTG'  $
      ,/ALIGN_LEFT ,VALUE='Blue to Green')

  
  WID_BUTTON_PickANCFile = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_PickANCFile' ,XOFFSET=330 ,YOFFSET=205  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Pick *.anc'+ $
      ' File')

  
  WID_TEXT_ANCFilename = Widget_Text(WID_BASE_AnchorPts,  $
      UNAME='WID_TEXT_ANCFilename' ,XOFFSET=12 ,YOFFSET=247  $
      ,SCR_XSIZE=420 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,VALUE=[ 'Select'+ $
      ' *.anc File' ] ,XSIZE=20 ,YSIZE=2)

  
  WID_BUTTON_Load_ANC = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_Load_ANC' ,XOFFSET=10 ,YOFFSET=205  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Load'+ $
      ' Fiducials (*.anc)')

  
  WID_BUTTON_Save_ANC = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_Save_ANC' ,XOFFSET=170 ,YOFFSET=205  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Save'+ $
      ' Fiducials (*.anc)')

  
  WID_LABEL_5 = Widget_Label(WID_BASE_AnchorPts, UNAME='WID_LABEL_5'  $
      ,XOFFSET=13 ,YOFFSET=279 ,SCR_XSIZE=420 ,SCR_YSIZE=15  $
      ,/ALIGN_LEFT ,VALUE='Leave this field empty if you do not want'+ $
      ' to save fiducials into file')

  
  WID_DROPLIST_TRANSFORM_METHOD = Widget_Droplist(WID_BASE_AnchorPts,  $
      UNAME='WID_DROPLIST_TRANSFORM_METHOD' ,XOFFSET=208 ,YOFFSET=8  $
      ,SCR_XSIZE=226 ,SCR_YSIZE=26 ,VALUE=[ 'Linear Regression',  $
      'POLYWARP', 'Pivot and Average (3 pts only)' ])

  
  WID_LABEL_2 = Widget_Label(WID_BASE_AnchorPts, UNAME='WID_LABEL_2'  $
      ,XOFFSET=19 ,YOFFSET=4 ,SCR_XSIZE=193 ,SCR_YSIZE=18  $
      ,/ALIGN_LEFT ,VALUE='Method for image transformation')

  
  WID_LABEL_1 = Widget_Label(WID_BASE_AnchorPts, UNAME='WID_LABEL_1'  $
      ,XOFFSET=115 ,YOFFSET=110 ,SCR_XSIZE=193 ,SCR_YSIZE=18  $
      ,/ALIGN_LEFT ,VALUE='Then press one of the buttons above.')

  
  WID_LABEL_3 = Widget_Label(WID_BASE_AnchorPts, UNAME='WID_LABEL_3'  $
      ,XOFFSET=18 ,YOFFSET=22 ,SCR_XSIZE=193 ,SCR_YSIZE=19  $
      ,/ALIGN_LEFT ,VALUE='(when there are at least 3 fiducials)')

  
  WID_BASE_6 = Widget_Base(WID_BASE_AnchorPts, UNAME='WID_BASE_6'  $
      ,XOFFSET=563 ,YOFFSET=335 ,TITLE='IDL' ,COLUMN=1  $
      ,/NONEXCLUSIVE)

  
  WID_BUTTON_Align_Z = Widget_Button(WID_BASE_6,  $
      UNAME='WID_BUTTON_Align_Z' ,/ALIGN_LEFT ,VALUE='Align Z')

  
  WID_Anchors_Z_Table = Widget_Table(WID_BASE_AnchorPts,  $
      UNAME='WID_Anchors_Z_Table' ,XOFFSET=481 ,YOFFSET=365  $
      ,SCR_XSIZE=276 ,SCR_YSIZE=510 ,/EDITABLE ,COLUMN_LABELS=[ 'Red'+ $
      ' Z', 'Green Z', 'Blue Z' ] ,XSIZE=3 ,YSIZE=500)

  
  WID_BUTTON_DisplayFiducials = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_DisplayFiducials' ,XOFFSET=23 ,YOFFSET=300  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Display'+ $
      ' Fiducials')

  
  WID_BASE_AutoDisplayCompleteFiducialSet =  $
      Widget_Base(WID_BASE_AnchorPts,  $
      UNAME='WID_BASE_AutoDisplayCompleteFiducialSet' ,XOFFSET=29  $
      ,YOFFSET=340 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_AutoDisplay_Selected_Fiducials =  $
      Widget_Button(WID_BASE_AutoDisplayCompleteFiducialSet,  $
      UNAME='WID_BUTTON_AutoDisplay_Selected_Fiducials' ,/ALIGN_LEFT  $
      ,VALUE='Display fiducials when selected')

  
  WID_BUTTON_Set_Fid_outline_size = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_Set_Fid_outline_size' ,XOFFSET=208  $
      ,YOFFSET=300 ,SCR_XSIZE=250 ,SCR_YSIZE=35 ,/ALIGN_CENTER  $
      ,VALUE='Set Fiducial Outline Circle Rad. (pix)')

  
  WID_TEXT_FidOutlineSize = Widget_Text(WID_BASE_AnchorPts,  $
      UNAME='WID_TEXT_FidOutlineSize' ,XOFFSET=478 ,YOFFSET=304  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=35 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BUTTON_RemoveFiducial = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_RemoveFiducial' ,XOFFSET=680 ,YOFFSET=228  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=31 ,/ALIGN_CENTER ,VALUE='Remove'+ $
      ' Fid.')

  
  WID_TEXT_FidRemoveNumber = Widget_Text(WID_BASE_AnchorPts,  $
      UNAME='WID_TEXT_FidRemoveNumber' ,XOFFSET=695 ,YOFFSET=265  $
      ,SCR_XSIZE=50 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '0' ]  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_BUTTON_AutoDetect = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_AutoDetect' ,XOFFSET=800 ,YOFFSET=5  $
      ,SCR_XSIZE=200 ,SCR_YSIZE=31 ,/ALIGN_CENTER ,VALUE='Autodetect'+ $
      ' Red Fiducials')

  
  WID_AutoDetect_Parameters = Widget_Table(WID_BASE_AnchorPts,  $
      UNAME='WID_AutoDetect_Parameters' ,XOFFSET=630 ,YOFFSET=3  $
      ,/EDITABLE ,COLUMN_LABELS=[ 'Value', '' ] ,ROW_LABELS=[ 'Thr.'+ $
      ' Min.', 'Thr. Max.', 'Rad. (pix.)' ] ,XSIZE=1 ,YSIZE=3)

  
  WID_BUTTON_AutoDetect_0 = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_AutoDetect_0' ,XOFFSET=800 ,YOFFSET=75  $
      ,SCR_XSIZE=200 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Autodetect'+ $
      ' Matching Fiducials')

  
  WID_AutoDetect_Match_Parameters = Widget_Table(WID_BASE_AnchorPts,  $
      UNAME='WID_AutoDetect_Match_Parameters' ,XOFFSET=630  $
      ,YOFFSET=104 ,/EDITABLE ,COLUMN_LABELS=[ 'Value' ]  $
      ,ROW_LABELS=[ 'Amp. Min.', 'Amp. Max.', 'Rad. (pix.)' ]  $
      ,XSIZE=1 ,YSIZE=3)

  
  WID_Anchors_Transf_Dist_Test = Widget_Table(WID_BASE_AnchorPts,  $
      UNAME='WID_Anchors_Transf_Dist_Test' ,XOFFSET=770 ,YOFFSET=365  $
      ,SCR_XSIZE=276 ,SCR_YSIZE=510 ,/EDITABLE ,COLUMN_LABELS=[  $
      'Dist. R-G', 'Dist. B-G', 'Dist. R-B' ] ,XSIZE=3 ,YSIZE=500)

  
  WID_BUTTON_Test_Transformation = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_Test_Transformation' ,XOFFSET=851  $
      ,YOFFSET=210 ,SCR_XSIZE=200 ,SCR_YSIZE=31 ,/ALIGN_CENTER  $
      ,VALUE='Test Transformation')

  
  WID_BASE_AutoDisplayFiducialIDs = Widget_Base(WID_BASE_AnchorPts,  $
      UNAME='WID_BASE_AutoDisplayFiducialIDs' ,XOFFSET=289  $
      ,YOFFSET=340 ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Display_Fiducial_IDs =  $
      Widget_Button(WID_BASE_AutoDisplayFiducialIDs,  $
      UNAME='WID_BUTTON_Display_Fiducial_IDs' ,/ALIGN_LEFT  $
      ,VALUE='Display Fiducials IDs')

  
  WID_DROPLIST_Autodetect_Filter =  $
      Widget_Droplist(WID_BASE_AnchorPts,  $
      UNAME='WID_DROPLIST_Autodetect_Filter' ,XOFFSET=800 ,YOFFSET=42  $
      ,SCR_XSIZE=150 ,SCR_YSIZE=25 ,TITLE='Use' ,VALUE=[ 'Frame'+ $
      ' Peaks', 'Grouped Peaks' ])

  
  WID_BUTTON_Remove_Unmatched = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_Remove_Unmatched' ,XOFFSET=800 ,YOFFSET=110  $
      ,SCR_XSIZE=200 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Remove'+ $
      ' Unmatched Fiducials')

  
  WID_Anchors_Transf_AverageErrors = Widget_Table(WID_BASE_AnchorPts,  $
      UNAME='WID_Anchors_Transf_AverageErrors' ,XOFFSET=764  $
      ,YOFFSET=305 ,SCR_XSIZE=296 ,SCR_YSIZE=55 ,/EDITABLE  $
      ,COLUMN_LABELS=[ 'Dist. R-G', 'Dist. B-G', 'Dist. R-B' ]  $
      ,ROW_LABELS=[ 'Average' ] ,XSIZE=3 ,YSIZE=1)

  
  WID_Anchors_Transf_WorstErrors = Widget_Table(WID_BASE_AnchorPts,  $
      UNAME='WID_Anchors_Transf_WorstErrors' ,XOFFSET=764  $
      ,YOFFSET=245 ,SCR_XSIZE=296 ,SCR_YSIZE=55 ,/EDITABLE  $
      ,COLUMN_LABELS=[ 'Dist. R-G', 'Dist. B-G', 'Dist. R-B' ]  $
      ,ROW_LABELS=[ 'Worst' ] ,XSIZE=3 ,YSIZE=1)

  
  WID_SLIDER_POLYWARP_Degree = Widget_Slider(WID_BASE_AnchorPts,  $
      UNAME='WID_SLIDER_POLYWARP_Degree' ,XOFFSET=460 ,YOFFSET=1  $
      ,SCR_XSIZE=140 ,SCR_YSIZE=48 ,TITLE='Polywarp Degree'  $
      ,MINIMUM=1 ,MAXIMUM=5 ,VALUE=1)

  
  WID_BASE_RescaleSigma = Widget_Base(WID_BASE_AnchorPts,  $
      UNAME='WID_BASE_RescaleSigma' ,XOFFSET=475 ,YOFFSET=137  $
      ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_Adj_Scl = Widget_Button(WID_BASE_RescaleSigma,  $
      UNAME='WID_BUTTON_Adj_Scl' ,/ALIGN_LEFT ,VALUE='Adj.'+ $
      ' Scale/Sigmas')

  
  WID_BUTTON_Remove_Bad_Fiducials = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_Remove_Bad_Fiducials' ,XOFFSET=800  $
      ,YOFFSET=153 ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Remove Bad Fiducials')

  
  WID_TEXT_FidRemove_Thr = Widget_Text(WID_BASE_AnchorPts,  $
      UNAME='WID_TEXT_FidRemove_Thr' ,XOFFSET=995 ,YOFFSET=153  $
      ,SCR_XSIZE=56 ,SCR_YSIZE=30 ,/EDITABLE ,/WRAP ,VALUE=[ '20' ]  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_DROPLIST_Fiducial_Source = Widget_Droplist(WID_BASE_AnchorPts,  $
      UNAME='WID_DROPLIST_Fiducial_Source' ,XOFFSET=460 ,YOFFSET=62  $
      ,SCR_XSIZE=125 ,SCR_YSIZE=25 ,TITLE='Use' ,VALUE=[ 'Frame'+ $
      ' Peaks', 'Window Cntr' ])

  
  WID_LABEL_andSAve_1 = Widget_Label(WID_BASE_AnchorPts,  $
      UNAME='WID_LABEL_andSAve_1' ,XOFFSET=795 ,YOFFSET=190  $
      ,SCR_XSIZE=260 ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Repeat'+ $
      ' Test+Remove till Worst is better then above')

  
  WID_BUTTON_Swap_RedGreen = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_Swap_RedGreen' ,XOFFSET=15 ,YOFFSET=90  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Swap R / G')

  
  WID_BUTTON_Swap_RedBlue = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_Swap_RedBlue' ,XOFFSET=15 ,YOFFSET=160  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Swap R / B')

  
  WID_BASE_XYlimits = Widget_Base(WID_BASE_AnchorPts,  $
      UNAME='WID_BASE_XYlimits' ,XOFFSET=475 ,YOFFSET=187  $
      ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_XYlimits = Widget_Button(WID_BASE_XYlimits,  $
      UNAME='WID_BUTTON_XYlimits' ,/ALIGN_LEFT ,VALUE='Limit X-Y'+ $
      ' coords')

  
  WID_XYlimits_table = Widget_Table(WID_BASE_AnchorPts,  $
      UNAME='WID_XYlimits_table' ,XOFFSET=440 ,YOFFSET=218  $
      ,SCR_XSIZE=230 ,SCR_YSIZE=75 ,/EDITABLE ,COLUMN_LABELS=[ 'Min',  $
      'Max' ] ,ROW_LABELS=[ 'X', 'Y' ] ,XSIZE=2 ,YSIZE=2)

  
  WID_BASE_LeaveOrigTotalRaw = Widget_Base(WID_BASE_AnchorPts,  $
      UNAME='WID_BASE_LeaveOrigTotalRaw' ,XOFFSET=475 ,YOFFSET=162  $
      ,TITLE='IDL' ,COLUMN=1 ,/NONEXCLUSIVE)

  
  WID_BUTTON_LeaveOrigTotalRaw =  $
      Widget_Button(WID_BASE_LeaveOrigTotalRaw,  $
      UNAME='WID_BUTTON_LeaveOrigTotalRaw' ,/ALIGN_LEFT ,VALUE='Leave'+ $
      ' Orig. TotalRaw')

  
  WID_BUTTON_DisplayFiducials_with_overalys =  $
      Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_DisplayFiducials_with_overalys' ,XOFFSET=646  $
      ,YOFFSET=307 ,SCR_XSIZE=100 ,SCR_YSIZE=35 ,/ALIGN_CENTER  $
      ,VALUE='Disp Fid+Over')

  
  WID_BUTTON_Refind = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_Refind' ,XOFFSET=980 ,YOFFSET=40  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Re-Find')

  
  WID_BUTTON_Copy_Red_to_Green = Widget_Button(WID_BASE_AnchorPts,  $
      UNAME='WID_BUTTON_Copy_Red_to_Green' ,XOFFSET=16 ,YOFFSET=122  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=25 ,/ALIGN_CENTER ,VALUE='Copy R->G')

  Widget_Control, /REALIZE, WID_BASE_AnchorPts

  XManager, 'WID_BASE_AnchorPts', WID_BASE_AnchorPts, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro AnchorWid, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_AnchorPts, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
