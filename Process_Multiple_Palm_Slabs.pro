; 
; IDL Widget Interface Procedures. This Code is automatically 
;     generated and should not be modified.

; 
; Generated on:	01/10/2019 08:04.20
; 
pro WID_BASE_Process_Multiple_PALM_Slabs_event, Event

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
      widget_info(Event.id, /tree_root) : event.id)


  wWidget =  Event.top

  case wTarget of

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BASE_Process_Multiple_PALM_Slabs'): begin
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Cancel_ZvsV_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        OnCancel_ZvsV_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_Select_RunDat_File_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Select_RunDat_File_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_Parameters_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_CH' )then $
        Do_Change_Params_mSlabs, Event
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TABLE_STR' )then $
        Do_Change_Params_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Shift_ZvsV_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Assign_zStates_and_Shift_ZvsV_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BTTN_Load_RunDat_File'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        On_Load_RunDat_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_TEXT_ZvsV_Slope_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_TEXT_CH' )then $
        On_Change_ZvsV_Slope_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Assign_zStates_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Assign_zStates_mSlabs, Event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_Assign_Transition_Frames_mSlabs'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then $
        Assign_Transition_Frames_mSlabs, Event
    end
    else:
  endcase

end

pro WID_BASE_Process_Multiple_PALM_Slabs, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_

  Resolve_Routine, 'Process_Multiple_Palm_Slabs_eventcb',/COMPILE_FULL_FILE  ; Load event callback routines
  
  WID_BASE_Process_Multiple_PALM_Slabs = Widget_Base(  $
      GROUP_LEADER=wGroup,  $
      UNAME='WID_BASE_Process_Multiple_PALM_Slabs' ,XOFFSET=5  $
      ,YOFFSET=5 ,SCR_XSIZE=562 ,SCR_YSIZE=686  $
      ,NOTIFY_REALIZE='Initialize_Process_Multiple_PALM_Slabs'  $
      ,TITLE='Process Multiple PALM Sbals (Z Voltage States)'  $
      ,SPACE=3 ,XPAD=3 ,YPAD=3 ,TLB_FRAME_ATTR=1)

  
  WID_BUTTON_Cancel_ZvsV_mSlabs =  $
      Widget_Button(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BUTTON_Cancel_ZvsV_mSlabs' ,XOFFSET=217 ,YOFFSET=576  $
      ,SCR_XSIZE=130 ,SCR_YSIZE=40 ,/ALIGN_CENTER ,VALUE='Cancel')

  
  WID_BTTN_Select_RunDat_File_mSlabs =  $
      Widget_Button(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BTTN_Select_RunDat_File_mSlabs' ,XOFFSET=100  $
      ,YOFFSET=7 ,SCR_XSIZE=300 ,SCR_YSIZE=30 ,/ALIGN_CENTER  $
      ,VALUE='Select Run Setup (.dat) File')

  
  WID_TXT_RunDat_Filename_mSlabs =  $
      Widget_Text(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_TXT_RunDat_Filename_mSlabs' ,XOFFSET=15 ,YOFFSET=42  $
      ,SCR_XSIZE=500 ,SCR_YSIZE=69 ,/EDITABLE ,/WRAP ,XSIZE=20  $
      ,YSIZE=2)

  
  WID_Parameters_mSlabs =  $
      Widget_Table(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_Parameters_mSlabs' ,XOFFSET=10 ,YOFFSET=163  $
      ,SCR_XSIZE=529 ,SCR_YSIZE=157 ,/EDITABLE ,/RESIZEABLE_COLUMNS  $
      ,COLUMN_LABELS=[ 'Z Voltage (V)', '# of Frames', '# of Trans.'+ $
      ' Frames', 'Z offset (nm)' ] ,ROW_LABELS=[ 'State 0', 'State 1'  $
      ] ,XSIZE=4 ,YSIZE=10)

  
  WID_BUTTON_Shift_ZvsV_mSlabs =  $
      Widget_Button(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BUTTON_Shift_ZvsV_mSlabs' ,XOFFSET=90 ,YOFFSET=460  $
      ,SCR_XSIZE=350 ,SCR_YSIZE=35 ,/ALIGN_CENTER ,VALUE='Assign'+ $
      ' States + Shift Z according to State Table')

  
  WID_BTTN_Load_RunDat_File =  $
      Widget_Button(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BTTN_Load_RunDat_File' ,XOFFSET=100 ,YOFFSET=118  $
      ,SCR_XSIZE=300 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Load Run'+ $
      ' Setup File')

  
  WID_LABEL_ZvsV_Slope_mSlabs =  $
      Widget_Label(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_LABEL_ZvsV_Slope_mSlabs' ,XOFFSET=30 ,YOFFSET=355  $
      ,SCR_XSIZE=180 ,SCR_YSIZE=25 ,/ALIGN_LEFT ,VALUE='Z(nm) vs'+ $
      ' Voltage(V) Slope ')

  
  WID_TEXT_ZvsV_Slope_mSlabs =  $
      Widget_Text(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_TEXT_ZvsV_Slope_mSlabs' ,XOFFSET=220 ,YOFFSET=350  $
      ,SCR_XSIZE=70 ,SCR_YSIZE=35 ,/EDITABLE ,/ALL_EVENTS ,/WRAP  $
      ,XSIZE=20 ,YSIZE=2)

  
  WID_BUTTON_Assign_zStates_mSlabs =  $
      Widget_Button(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BUTTON_Assign_zStates_mSlabs' ,XOFFSET=90  $
      ,YOFFSET=415 ,SCR_XSIZE=350 ,SCR_YSIZE=35 ,/ALIGN_CENTER  $
      ,VALUE='Assign  Z States according to State Table')

  
  WID_BUTTON_Assign_Transition_Frames_mSlabs =  $
      Widget_Button(WID_BASE_Process_Multiple_PALM_Slabs,  $
      UNAME='WID_BUTTON_Assign_Transition_Frames_mSlabs' ,XOFFSET=89  $
      ,YOFFSET=507 ,SCR_XSIZE=350 ,SCR_YSIZE=35 ,/ALIGN_CENTER  $
      ,VALUE='Assign  Transition Frames to Z State -1')

  Widget_Control, /REALIZE, WID_BASE_Process_Multiple_PALM_Slabs

  XManager, 'WID_BASE_Process_Multiple_PALM_Slabs', WID_BASE_Process_Multiple_PALM_Slabs, /NO_BLOCK  

end
; 
; Empty stub procedure used for autoloading.
; 
pro Process_Multiple_Palm_Slabs, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
  WID_BASE_Process_Multiple_PALM_Slabs, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_
end
