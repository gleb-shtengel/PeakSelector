; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	04/30/2008 07:48.18
; 
pro WID_BASE_XYZ_Fid_Pts_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_XYZ_Fid_Pts'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TABLE_0'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        DoInsert_XYZ_Anchor, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_0'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Do_XYZ_Tilt_Transforms, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Add_XYZ_Fiducial'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnButton_XYZ_AddFiducial, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Clear_XYZ'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Clear_XYZ_Fiducials, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Pick_XYZ_File'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnPick_XYZ_File, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Load_XYZ'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Load_XYZ_File, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Save_XYZ'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Save_XYZ_File, Event
    end
    else:
  endcase

end

pro WID_BASE_XYZ_Fid_Pts, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'Remove_XYZ_Tilt_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_XYZ_Fid_Pts = Widget_Base( GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_XYZ_Fid_Pts' ,XOFFSET=5 ,YOFFSET=5  $
      ,SCR_XSIZE=453 ,SCR_YSIZE=460  $
      ,NOTIFY_REALIZE='Initialize_XYZ_Fid_Wid' ,TITLE='XYZ Tilt'+ $
      ' Fiducial Points' ,SPACE=3 ,XPAD=3 ,YPAD=3)

  
  WID_TABLE_0 = Widget_Table(WID_BASE_XYZ_Fid_Pts,  $
      UNAME='WID_TABLE_0' ,XOFFSET=50 ,YOFFSET=40 ,SCR_XSIZE=279  $
      ,SCR_YSIZE=115 ,/EDITABLE ,COLUMN_LABELS=[ 'X', 'Y', 'Z' ]  $
      ,XSIZE=3 ,YSIZE=3)

  
  WID_BUTTON_0 = Widget_Button(WID_BASE_XYZ_Fid_Pts,  $
      UNAME='WID_BUTTON_0' ,XOFFSET=10 ,YOFFSET=250 ,SCR_XSIZE=90  $
      ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Transform')

  
  WID_BUTTON_Add_XYZ_Fiducial = Widget_Button(WID_BASE_XYZ_Fid_Pts,  $
      UNAME='WID_BUTTON_Add_XYZ_Fiducial' ,XOFFSET=120 ,YOFFSET=184  $
      ,SCR_XSIZE=115 ,SCR_YSIZE=31 ,/ALIGN_CENTER ,VALUE='Add XYZ'+ $
      ' Fiducial')

  
  WID_BUTTON_Clear_XYZ = Widget_Button(WID_BASE_XYZ_Fid_Pts,  $
      UNAME='WID_BUTTON_Clear_XYZ' ,XOFFSET=10 ,YOFFSET=185  $
      ,SCR_XSIZE=90 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Clear All')

  
  WID_LABEL_0 = Widget_Label(WID_BASE_XYZ_Fid_Pts,  $
      UNAME='WID_LABEL_0' ,XOFFSET=122 ,YOFFSET=215 ,SCR_XSIZE=292  $
      ,SCR_YSIZE=18 ,/ALIGN_LEFT ,VALUE='To add new fiducial point,'+ $
      ' first select the area of the interest')

  
  WID_LABEL_1 = Widget_Label(WID_BASE_XYZ_Fid_Pts,  $
      UNAME='WID_LABEL_1' ,XOFFSET=124 ,YOFFSET=230 ,SCR_XSIZE=280  $
      ,SCR_YSIZE=18 ,/ALIGN_LEFT ,VALUE='on the image. Then press one'+ $
      ' of the buttons above.')

  
  WID_BUTTON_Pick_XYZ_File = Widget_Button(WID_BASE_XYZ_Fid_Pts,  $
      UNAME='WID_BUTTON_Pick_XYZ_File' ,XOFFSET=215 ,YOFFSET=268  $
      ,SCR_XSIZE=180 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Pick *.xyz'+ $
      ' File')

  
  WID_TEXT_XYZ_Filename = Widget_Text(WID_BASE_XYZ_Fid_Pts,  $
      UNAME='WID_TEXT_XYZ_Filename' ,XOFFSET=4 ,YOFFSET=377  $
      ,SCR_XSIZE=432 ,SCR_YSIZE=32 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_BUTTON_Load_XYZ = Widget_Button(WID_BASE_XYZ_Fid_Pts,  $
      UNAME='WID_BUTTON_Load_XYZ' ,XOFFSET=20 ,YOFFSET=330  $
      ,SCR_XSIZE=180 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Load'+ $
      ' Fiducials (*.xyz)')

  
  WID_BUTTON_Save_XYZ = Widget_Button(WID_BASE_XYZ_Fid_Pts,  $
      UNAME='WID_BUTTON_Save_XYZ' ,XOFFSET=215 ,YOFFSET=330  $
      ,SCR_XSIZE=180 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Save'+ $
      ' Fiducials (*.xyz)')

  
  WID_LABEL_andSAve = Widget_Label(WID_BASE_XYZ_Fid_Pts,  $
      UNAME='WID_LABEL_andSAve' ,XOFFSET=5 ,YOFFSET=290  $
      ,SCR_XSIZE=120 ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='(and save'+ $
      ' Fiducials')

  
  WID_LABEL_andSAve_0 = Widget_Label(WID_BASE_XYZ_Fid_Pts,  $
      UNAME='WID_LABEL_andSAve_0' ,XOFFSET=5 ,YOFFSET=305  $
      ,SCR_XSIZE=120 ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='into ANC'+ $
      ' file)')

  
  WID_LABEL_5 = Widget_Label(WID_BASE_XYZ_Fid_Pts,  $
      UNAME='WID_LABEL_5' ,XOFFSET=5 ,YOFFSET=410 ,SCR_XSIZE=500  $
      ,SCR_YSIZE=15 ,/ALIGN_LEFT ,VALUE='Leave this field empty if'+ $
      ' you do not want to save fiducials into file')

  Widget_Control, /REALIZE, WID_BASE_XYZ_Fid_Pts

  XManager, 'WID_BASE_XYZ_Fid_Pts', WID_BASE_XYZ_Fid_Pts, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Remove_XYZ_Tilt, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_XYZ_Fid_Pts, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
