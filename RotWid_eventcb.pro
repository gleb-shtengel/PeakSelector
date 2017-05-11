;
; IDL Event Callback Procedures
; RotWid_eventcb
;
; Generated on:	09/17/2006 10:59.14
;
;-----------------------------------------------------------------
; Slider Value Changed Callback Procedure.
; Argument:
;   Event structure:
;
;   {WIDGET_SLIDER, ID:0L, TOP:0L, HANDLER:0L, VALUE:0L, DRAG:0}
;
;   ID is the widget ID of the component generating the event. TOP is
;       the widget ID of the top level widget containing ID. HANDLER
;       contains the widget ID of the widget associated with the
;       handler routine.

;   VALUE returns the new value of the slider. DRAG returns integer 1
;       if the slider event was generated as part of a drag
;       operation, or zero if the event was generated when the user
;       had finished positioning the slider.

;   Retrieve the IDs of other widgets in the widget hierarchy using
;       id=widget_info(Event.top, FIND_BY_UNAME=name)
;-----------------------------------------------------------------
;
; Empty stub procedure used for autoloading.
;
pro RotWid_eventcb
end
;
;-----------------------------------------------------------------
;
pro OnRealizeRotz, wWidget
WIDGET_CONTROL, wWidget, SET_UVALUE=0, /NO_COPY
end
;
;-----------------------------------------------------------------
;
pro OnRotate, Event			; calls Twist,theta

WidRotAngleID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_Rot')
widget_control,WidRotAngleID,get_value=rot_angle
theta=!pi/180.*rot_angle
Twist,theta
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro OnCancel, Event			; cancels and closes the menu widget
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro Twist, theta					;Rotates data by theta degrees in xy space
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]

xc=ParamLimits[2,2]
yc=ParamLimits[3,2]
Par_Size = (CGrpSize < (size(paramlimits))[1])
newx=(cos(theta)*(CGroupParams[2,*]-ParamLimits[2,2])-sin(theta)*(CGroupParams[3,*]-ParamLimits[3,2]))+ParamLimits[2,2]
newy=(sin(theta)*(CGroupParams[2,*]-ParamLimits[2,2])+cos(theta)*(CGroupParams[3,*]-ParamLimits[3,2]))+ParamLimits[3,2]
CGroupParams[2,*]=newx>0
CGroupParams[3,*]=newy>0
newx=(cos(theta)*(CGroupParams[19,*]-ParamLimits[19,2])-sin(theta)*(CGroupParams[20,*]-ParamLimits[20,2]))+ParamLimits[19,2]
newy=(sin(theta)*(CGroupParams[19,*]-ParamLimits[19,2])+cos(theta)*(CGroupParams[20,*]-ParamLimits[20,2]))+ParamLimits[20,2]
CGroupParams[19,*]=newx>0
CGroupParams[20,*]=newy>0
x0=[0,0,xydsz[0],xydsz[0]]
y0=[0,xydsz[1],0,xydsz[1]]
x1=(cos(theta)*(x0-xc)-sin(theta)*(y0-yc))+xc
y1=(sin(theta)*(x0-xc)+cos(theta)*(y0-yc))+yc
ParamLimits[2,0] = 0; min(x1)
ParamLimits[3,0] = 0; min(y1)
ParamLimits[2,1] = max(x1)
ParamLimits[3,1] = max(y1)
ParamLimits[19,0:1]=ParamLimits[2,0:1]
ParamLimits[20,0:1]=ParamLimits[3,0:1]
xsz = max(x1);-min(x1)
ysz = max(y1);-min(x1)
xydsz = [xsz,ysz]
;totalrawdata=rot(totalrawdata,-theta/!dtor,/interp)
P=[[xc -(xc*cos(theta)+yc*sin(theta)), sin(theta)],[ cos(theta),0]]
Q=[[yc -(yc*cos(theta)-xc*sin(theta)), cos(theta)],	[ -sin(theta),0]]
TotalRawData=poly_2D(temporary(TotalRawData),P,Q,1,xsz,ysz,MISSING=0)
if (n_elements(DIC) gt 1) then DIC=poly_2D(temporary(DIC),P,Q,1,xsz,ysz,MISSING=0)
wtable = Widget_Info(TopID, find_by_uname='WID_TABLE_0')
widget_control,wtable,set_value=transpose(ParamLimits[0:(Par_Size-1),0:3]), use_table_select=[0,0,3,(Par_Size-1)]
widget_control, wtable, /editable,/sensitive

return
end