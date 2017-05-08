;
; IDL Event Callback Procedures
; ParameterInterpolation_eventcb
;
; Generated on:	12/03/2010 14:09.59
;
;-----------------------------------------------------------------
; Activate Button Callback Procedure.
; Argument:
;   Event structure:
;
;   {WIDGET_BUTTON, ID:0L, TOP:0L, HANDLER:0L, SELECT:0}
;
;   ID is the widget ID of the component generating the event. TOP is
;       the widget ID of the top level widget containing ID. HANDLER
;       contains the widget ID of the widget associated with the
;       handler routine.

;   SELECT is set to 1 if the button was set, and 0 if released.
;       Normal buttons do not generate events when released, so
;       SELECT will always be 1. However, toggle buttons (created by
;       parenting a button to an exclusive or non-exclusive base)
;       return separate events for the set and release actions.

;   Retrieve the IDs of other widgets in the widget hierarchy using
;       id=widget_info(Event.top, FIND_BY_UNAME=name)

;-----------------------------------------------------------------
;
; Empty stub procedure used for autoloading.
;
pro ParameterInterpolation_eventcb
end
;
;-----------------------------------------------------------------
;
pro Initialize_XY_Interp_Menu, wWidget
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate

WidD_XVariable_ID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_X_Interp')
widget_control,WidD_XVariable_ID, SET_VALUE=RowNames[0:(CGrpSize-1)], Set_Droplist_Select=20
WidD_YVariable_ID = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Y_Interp')
widget_control,WidD_YVariable_ID, SET_VALUE=RowNames[0:(CGrpSize-1)], Set_Droplist_Select=19

end
;
;-----------------------------------------------------------------
;
pro OnWrite_XY_Interp, Event		;Drift corrects x and y pix to constant guide star coordinates
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

BKGRND = 'Black'			; set to 'White' to get white background
ExtractSubset_XY_Interp, Event, deviation, BKGRND


WidD_XVariable_ID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_X_Interp')
X_dr_index = widget_info(WidD_XVariable_ID,/DropList_Select)
widget_control, WidD_XVariable_ID, GET_VALUE=X_Labels
X_label=X_Labels[X_dr_index]
X_ind=min(where(RowNames eq X_label))

WidD_YVariable_ID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Y_Interp')
Y_dr_index = widget_info(WidD_YVariable_ID,/DropList_Select)
widget_control, WidD_YVariable_ID, GET_VALUE=Y_Labels
Y_label=Y_Labels[Y_dr_index]
Y_ind=min(where(RowNames eq Y_label))

CGroupParams[Y_ind,*] = CGroupParams[Y_ind,*] - deviation

end
;
;-----------------------------------------------------------------
;
pro OnTest_XY_Interpolation, Event	;Shows fit to data:	XY coordinates
BKGRND = 'Black'			; set to 'White' to get white background
ExtractSubset_XY_Interp, Event, deviation, BKGRND
end
;
;-----------------------------------------------------------------
pro OnButtonClose_XY_Interp, Event
	widget_control,event.top,/destroy
end
;
;-----------------------------------------------------------------
;
pro ExtractSubset_XY_Interp, Event, deviation, BKGRND	;Pulls out subset of data from param limits and fits x,y vs frames
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common InfoFit, pth, filen, ini_filename, thisfitcond, saved_pks_filename, TransformEngine, grouping_gap, grouping_radius100, idl_pwd, temp_dir; TransformEngine : 0=Local, 1=Cluster
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

WidDListDispFitMethodID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_XY_Interp_Method')
FitMethod = widget_info(WidDListDispFitMethodID,/DropList_Select)

WidD_XVariable_ID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_X_Interp')
X_dr_index = widget_info(WidD_XVariable_ID,/DropList_Select)
widget_control, WidD_XVariable_ID, GET_VALUE = X_Labels
X_label=X_Labels[X_dr_index]
X_ind=min(where(RowNames eq X_label))

WidD_YVariable_ID = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Y_Interp')
Y_dr_index = widget_info(WidD_YVariable_ID,/DropList_Select)
widget_control, WidD_YVariable_ID, GET_VALUE=Y_Labels
Y_label=Y_Labels[Y_dr_index]
Y_ind=min(where(RowNames eq Y_label))

PeakLbls = [	'Offset' ,$
				'Amplitude' ,$
				'X Position' ,$
				'Y Position' ,$
				'X Peak Width' ,$
				'Y Peak Width' ,$
				'6 N Photons' ,$
				'ChiSquared' ,$
				'FitOK' ,$
				'Frame Number' ,$
				'Peak Index of Frame' ,$
				'Peak Global Index' ,$
				'12 X PkW * Y PkW' ,$
				'Sigma Amplitude' ,$
				'Sigma X Pos rtNph' ,$
				'Sigma Y Pos rtNph' ,$
				'Sigma X Pos Full' ,$
				'Sigma Y Pos Full' ,$
				'Amplitude L1' ,$
				'Amplitude L2' ,$
				'Amplitude L3' ,$
				'30 FitOK Labels' ,$
				'Sigma Amp L1' ,$
				'Sigma Amp L2' ,$
				'Sigma Amp L3' ,$
				'Z Position' ,$
				'Sigma Z' ,$
				'36 Coherence' ,$
				'XY Ellipticity' ,$
				'Unwrapped Z' ,$
				'Unwrapped Z Error' ]

PeakParamId=where(PeakLbls eq X_label)

if PeakParamId ge 0 then FilterIt else GroupFilterIt

subsetindex=where(filter eq 1,cnt)
;print, 'subset has ',cnt,' points'
if cnt le 0 then return

subset=CGroupParams[*,subsetindex]

if FitMethod eq 0 then begin
	WidSldFitOrderID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_XY_Poly_Interp_Order')
	widget_control,WidSldFitOrderID,get_value=fitorder
	x_center=mean(subset[X_ind,*])
	y_center=mean(subset[Y_ind,*])
	Ycoef=poly_fit((subset[X_ind,*]-x_center),(subset[Y_ind,*]),fitorder,YFIT=YFit)
	deviation = poly((CGroupParams[X_ind,*]-x_center),ycoef)-Yfit[0]
	;Yfit=Yfit+y_center
endif else begin
	WidSmWidthID = Widget_Info(Event.Top, find_by_uname='WID_SLIDER_XY_Interp_Sm_Width')
	widget_control,WidSmWidthID,get_value=SmWidth
	indecis=sort(subset[X_ind,*])
	n_ele=n_elements(indecis)
	x_data = subset[X_ind,indecis]
	y_data = subset[Y_ind,indecis]
	Ysmooth=smooth(subset[6,indecis]*subset[Y_ind,indecis],SmWidth,/EDGE_TRUNCATE)/smooth(subset[6,indecis],SmWidth,/EDGE_TRUNCATE)
	Yres=interpol(Ysmooth,subset[X_ind,indecis],CGroupParams[X_ind,*])
	Yfit=Yres[subsetindex]
	deviation=Yres-Yfit[0]
endelse

if BKGRND eq 'White' then begin
	!p.background=255
	!P.Font=1
	DEVICE, SET_FONT='Helvetica Bold', /TT_FONT
	SET_CHARACTER_SIZE=[50,55]
	col=0
	plot,subset[X_ind,*],subset[Y_ind,*],xtitle=X_label,ytitle=Y_label,xrange=Paramlimits[X_ind,0:1],xstyle=1,yrange=(Paramlimits[Y_ind,0:1]),$
		ystyle=1, col=col,thick=2,charthick=2,charsize=2,xthick=2,ythick=2,psym=3,symsize=0.15
	oplot,subset[X_ind,*],YFit,col=100,psym=6,symsize=0.25

	;if FitMethod eq 0 then for i =0,fitorder do xyouts,0.8,0.45-0.02*i,ycoef[i],/normal,col=col,charthick=2,charsize=2
endif
if BKGRND ne 'White' then begin
	plot,subset[X_ind,*],subset[Y_ind,*],xtitle=X_label,ytitle=Y_label,xrange=Paramlimits[X_ind,0:1],xstyle=1,yrange=Paramlimits[Y_ind,0:1],ystyle=1,$
		psym=3,symsize=0.15
	oplot,subset[X_ind,*],YFit,col=100,psym=6,symsize=0.25
	if FitMethod eq 0 then for i =0,fitorder do xyouts,0.8,0.45-0.02*i,ycoef[i],/normal
endif
!p.background=0
return
end

