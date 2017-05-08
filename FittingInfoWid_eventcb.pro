;
; IDL Event Callback Procedures
; FittingInfoWid_eventcb
;
; Generated on:	04/05/2007 14:23.27
;
;-----------------------------------------------------------------
; Notify Realize Callback Procedure.
; Argument:
;   wWidget - ID number of specific widget.
;
;
;   Retrieve the IDs of other widgets in the widget hierarchy using
;       id=widget_info(Event.top, FIND_BY_UNAME=name)

;-----------------------------------------------------------------
pro DoRealizeInfo, wWidget			;On create InfoFit fill with values from .txt file if availible
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
wtable = Widget_Info(wWidget, find_by_uname='WID_TABLE_InfoFile')
values=[thisfitcond.zerodark,  thisfitcond.xsz,  thisfitcond.ysz,  thisfitcond.Nframesmax,$
		thisfitcond.Frm0,  thisfitcond.FrmN,  thisfitcond.Thresholdcriteria,$
		thisfitcond.filetype,  thisfitcond.LimBotA1,  thisfitcond.LimTopA1,$
		thisfitcond.LimBotSig,  thisfitcond.LimTopSig,  thisfitcond.LimChiSq,$
		thisfitcond.Cntpere, thisfitcond.maxcnt1, thisfitcond.maxcnt2,$
		thisfitcond.fliphor,thisfitcond.flipvert, thisfitcond.MaskSize,$
		thisfitcond.GaussSig,thisfitcond.MaxBlck,thisfitcond.SparseOversampling,$
		thisfitcond.SparseLambda,thisfitcond.SparseDelta,thisfitcond.SpError,thisfitcond.SpMaxIter]
widget_control,wtable,set_value=transpose(values), use_table_select=[0,0,0,25]
if !VERSION.OS_family eq 'unix' then widget_control,wtable,COLUMN_WIDTH=[150,100],use_table_select = [ -1, 0, 0, 25 ]
WidDListSigmaSym = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_SetSigmaFitSym')
widget_control,WidDListSigmaSym,SET_DROPLIST_SELECT=thisfitcond.SigmaSym		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
WidID_DROPLIST_Localization_Method = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Localization_Method')
widget_control,WidID_DROPLIST_Localization_Method,SET_DROPLIST_SELECT=thisfitcond.LocalizationMethod		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
end
;
;-----------------------------------------------------------------
;
pro DoRealizeDropListDispType, wWidget
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
widget_control,wWidget,SET_DROPLIST_SELECT = TransformEngine ? 3 : 1			;Sets the default value to "Some Frames/Peaks" for Windows and to "Cluster" for UNIX
end
;
;-----------------------------------------------------------------
;
pro DoInsertInfo, Event				;edits the fit info table
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
widget_control,event.id,get_value=thevalue
value=float(reform(thevalue))
CASE event.y OF
	0:thisfitcond.zerodark=value[event.y]
	1:thisfitcond.xsz=value[event.y]
	2:thisfitcond.ysz=value[event.y]
	3:thisfitcond.Nframesmax=value[event.y]
	4:thisfitcond.Frm0=value[event.y]
	5:thisfitcond.FrmN=value[event.y]
	6:thisfitcond.Thresholdcriteria=value[event.y]
	7:thisfitcond.filetype=value[event.y]
	8:thisfitcond.LimBotA1=value[event.y]
	9:thisfitcond.LimTopA1=value[event.y]
	10:thisfitcond.LimBotSig=value[event.y]
	11:thisfitcond.LimTopSig=value[event.y]
	12:thisfitcond.LimChiSq=value[event.y]
	13:thisfitcond.Cntpere=value[event.y]
	14:thisfitcond.maxcnt1=value[event.y]
	15:thisfitcond.maxcnt2=value[event.y]
	16:thisfitcond.fliphor=value[event.y]
	17:thisfitcond.flipvert=value[event.y]
	18:thisfitcond.MaskSize=value[event.y]
	19:thisfitcond.GaussSig=value[event.y]
	20:thisfitcond.MaxBlck=value[event.y]
	21:thisfitcond.SparseOversampling=value[event.y]
	22:thisfitcond.SparseLambda=value[event.y]
	23:thisfitcond.SparseDelta=value[event.y]
	24:thisfitcond.SpError=value[event.y]
	25:thisfitcond.SpMaxIter=value[event.y]
ENDCASE
widget_control,event.id,set_value=transpose(value),use_table_select=[0,0,0,25]
end
;
;-----------------------------------------------------------------
;
pro SetSigmaFitSym, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
	WidDListSigmaSym = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_SetSigmaFitSym')
	SigmaSym=widget_info(WidDListSigmaSym,/DropList_Select)		;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit else x & y indep
	thisfitcond.SigmaSym = SigmaSym
   ;print,'SetSigmaSym',thisfitcond
end
;
;-----------------------------------------------------------------
;
pro OnCancelFit, Event				;Cancels and returns peak fitting options widget
	widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro OnInfoOK, Event					;Starts fitting of data according to settings
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
widget_control,/hourglass

WidDListDispLevel = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_FitDisplayType')
DispType=widget_info(WidDListDispLevel,/DropList_Select)	;set to 0 - min display while fitting 1 - some display 2 - full display  3 - Cluster  4- GPU
TransformEngine = (DispType eq 3) ? 1 : 0
SetSigmaFitSym, Event
WriteInfoFile								;writes .txt file of basic fitting parameters
ReadRawLoop6,DispType		;goes through data and fits peaks
widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro FittingInfoWid_eventcb
end
;
;-----------------------------------------------------------------
;
pro SetLocalizationMethod, Event
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
	thisfitcond.LocalizationMethod=Event.index
end
