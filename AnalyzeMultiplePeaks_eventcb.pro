;
; IDL Event Callback Procedures
; AnalyzeMultiplePeaks_eventcb
;
; Generated on:	07/08/2009 07:13.14
;
;
;
;
;-----------------------------------------------------------------
; Empty stub procedure used for autoloading.
pro AnalyzeMultiplePeaks_eventcb
end
;
;-----------------------------------------------------------------
;
pro Initialize_AnalizeMultiplePeaks, wWidget
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WID_Base0_Peakselector, ID:TopID, TOP:TopID, HANDLER:TopID }

;Xrange_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Xrange')
;widget_control,Xrange_ID,SET_VALUE = '4.0'

;Yrange_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Yrange')
;widget_control,Yrange_ID,SET_VALUE = '4.0'

	WidID_DROPLIST_Analyze_Multiple_Filter = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Analyze_Multiple_Filter')
	widget_control,WidID_DROPLIST_Analyze_Multiple_Filter,SET_DROPLIST_SELECT = SaveASCII_Filter
	WidID_DROPLIST_Save_Peak_ASCII_XY = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Save_Peak_ASCII_XY')
	widget_control,WidID_DROPLIST_Save_Peak_ASCII_XY,SET_DROPLIST_SELECT = SaveASCII_units
		WidID_DROPLIST_Save_Peak_ASCII_Parameters = Widget_Info(wWidget, find_by_uname='WID_DROPLIST_Save_Peak_ASCII_Parameters')
	widget_control,WidID_DROPLIST_Save_Peak_ASCII_Parameters,SET_DROPLIST_SELECT = SaveASCII_ParamChoice

	WidID_TEXT_ASCII_Peak_Save_Parameter_List = Widget_Info(wWidget, find_by_uname='WID_TEXT_ASCII_Peak_Save_Parameter_List')
	widget_control,WidID_TEXT_ASCII_Peak_Save_Parameter_List,SET_VALUE = string(SaveASCII_ParamList),/NO_NEWLINE,/EDITABLE

end
;
;-----------------------------------------------------------------
;
pro OnPickPeaksFile, Event
PeaksFile = Dialog_Pickfile(/read,get_path=fpath,filter=['*.anc'],title='Select *.anc file to open')
PeaksFileInfo=file_info(PeaksFile)
if PeaksFile ne '' then cd,fpath
PeaksFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_PeaksFilename')
widget_control,PeaksFileWidID,SET_VALUE = PeaksFile
end
;
;-----------------------------------------------------------------
;
pro On_Select_Peaks_Filter, Event
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
	WidID_DROPLIST_Analyze_Multiple_Filter = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Analyze_Multiple_Filter')
	SaveASCII_Filter = widget_info(WidID_DROPLIST_Analyze_Multiple_Filter,/DropList_Select)
end
;
;-----------------------------------------------------------------
;
pro On_Select_Peak_SaveASCII_ParamChoice, Event
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
	WidID_DROPLIST_Save_Peak_ASCII_Parameters = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Save_Peak_ASCII_Parameters')
	SaveASCII_ParamChoice = widget_info(WidID_DROPLIST_Save_Peak_ASCII_Parameters,/DropList_Select)
end
;
;-----------------------------------------------------------------
;
pro On_Select_Peak_SaveASCII_units, Event
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
	WidID_DROPLIST_Save_Peak_ASCII_XY = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Save_Peak_ASCII_XY')
	SaveASCII_units = widget_info(WidID_DROPLIST_Save_Peak_ASCII_XY,/DropList_Select)
end
;
;-----------------------------------------------------------------
;
pro On_ASCII_Peak_ParamList_change, Event
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
	WidID_TEXT_ASCII_Peak_Save_Parameter_List = Widget_Info(Event.Top, find_by_uname='WID_TEXT_ASCII_Peak_Save_Parameter_List')
	widget_control,WidID_TEXT_ASCII_Peak_Save_Parameter_List,GET_VALUE = SaveASCII_ParamString
	len=strlen(SaveASCII_ParamString)
	i=0 & j=0
	while i lt len-1 do begin
		chr=STRMID(SaveASCII_ParamString,i,1)
		;print,i,'  chr=',chr,' byte=' , byte(chr)
		if (byte(chr) ne 32) and  (byte(chr) ne 9) then begin
			ParamStr=chr
			while ((i lt len) and (byte(chr) ne 32) and  (byte(chr) ne 9)) do begin
				i++
				chr=STRMID(SaveASCII_ParamString,i,1)
				ParamStr=ParamStr+chr
			endwhile
			if j eq 0 then SaveASCII_ParamList = fix(strcompress(ParamStr)) else SaveASCII_ParamList = [SaveASCII_ParamList,fix(strcompress(ParamStr))]
			j++
		endif
		i++
	endwhile
end
;
;-----------------------------------------------------------------
;
pro On_AnalyzeMultiple_Start, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common SaveASCII, SaveASCII_Filename, SaveASCII_Filter, SaveASCII_units, SaveASCII_ParamChoice, SaveASCII_ParamList
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel,  z_media_multiplier
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WID_Base0_Peakselector, ID:TopID, TOP:TopID, HANDLER:TopID }
Filter_ID = Widget_Info(Event.top, find_by_uname='WID_DROPLIST_Analyze_Multiple_Filter')
filter_select=Widget_Info(Filter_ID,/DROPLIST_SELECT)		; 0 - Frame Peaks, 1 - Group Peaks

wstatus = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Status')

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

WidSldFractionHistAnalID = Widget_Info(TopID, find_by_uname='WID_SLIDER_FractionHistAnal')
widget_control,WidSldFractionHistAnalID,get_value=fr_search
fr_srch_str = strtrim(fr_search,2)

Xrange_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_Xrange')
widget_control,Xrange_ID,GET_VALUE = Xrange_txt
Xrange=float(Xrange_txt[0])

Yrange_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_Yrange')
widget_control,Yrange_ID,GET_VALUE = Yrange_txt
Yrange=float(Yrange_txt[0])

MinNPeaks_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_MinNPeaks')
widget_control,MinNPeaks_ID,GET_VALUE = MinNPeaks_txt
MinNPeaks=float(MinNPeaks_txt[0])

PeaksFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_PeaksFilename')
widget_control,PeaksFileWidID,GET_VALUE = PeaksFile
PeaksFileInfo=file_info(PeaksFile)
if ~PeaksFileInfo.exists then begin
	z=dialog_message('Peaks File does not exist')
	return
endif

WidID_BUTTON_Save_Each_Peak_Distribution = widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_Save_Each_Peak_Distribution')
Save_individual_ASCII = widget_info(WidID_BUTTON_Save_Each_Peak_Distribution,/button_set)

AnchorPnts=fltarr(6,100)
openr,1,PeaksFile
readf,1,AnchorPnts
close,1

indices=where(AnchorPnts[0,*] ne 0,PeakNum)

PeakX=AnchorPnts[0,indices]
PeakY=AnchorPnts[1,indices]

wstatus = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Status')

for ip=0,(PeakNum-1) do begin
	widget_control,wstatus,set_value='Analyzing Peak '+strmid(ip+1,2)+' of '+strmid(PeakNum,2)
	ParamLimits[2,0]=PeakX[ip]-Xrange/2.0
	ParamLimits[2,1]=PeakX[ip]+Xrange/2.0
	ParamLimits[2,2]=PeakX[ip]
	ParamLimits[2,3]=Xrange
	ParamLimits[19,0]=PeakX[ip]-Xrange/2.0
	ParamLimits[19,1]=PeakX[ip]+Xrange/2.0
	ParamLimits[19,2]=PeakX[ip]
	ParamLimits[19,3]=Xrange
	ParamLimits[3,0]=PeakY[ip]-Yrange/2.0
	ParamLimits[3,1]=PeakY[ip]+Yrange/2.0
	ParamLimits[3,2]=PeakY[ip]
	ParamLimits[3,3]=Yrange
	ParamLimits[20,0]=PeakY[ip]-Yrange/2.0
	ParamLimits[20,1]=PeakY[ip]+Yrange/2.0
	ParamLimits[20,2]=PeakY[ip]
	ParamLimits[20,3]=Yrange
	if filter_select then begin
		OnGroupCentersButton, Event1
		if total(filter) ge MinNPeaks then begin
			OnAnalyze2, Dist_Results, 1, Event1
			Results = (ip eq 0) ? transpose(Dist_Results) : [Results, transpose(Dist_Results)]
		endif
	endif else begin
		OnPeakCentersButton, Event1
		if total(filter) ge MinNPeaks then begin
			OnAnalyze2, Dist_Results, 0, Event1
			Results = (ip eq 0) ? transpose(Dist_Results) : [Results, transpose(Dist_Results)]
		endif
	endelse

	if Save_individual_ASCII then begin
		FilteredPeakIndex=where(filter eq 1,cnt1)
		if SaveASCII_ParamChoice then indecis = SaveASCII_ParamList[where(SaveASCII_ParamList lt CGrpSize)] else indecis = indgen(CGrpSize)
		if (cnt1 gt 0) and (n_elements(indecis) gt 0) then begin
			FGroupParams=CGroupParams[*,FilteredPeakIndex]
			if SaveASCII_units then begin
				FGroupParams[2:5,*]*=nm_per_pixel
				FGroupParams[14:17,*]*=nm_per_pixel
				FGroupParams[19:22,*]*=nm_per_pixel
			endif
			curr_ext='_cluster_'+strtrim(ip,2)+'.txt'
			SaveASCII_Filename = addextension(PeaksFile,curr_ext)
			Title_String=RowNames[indecis[0]]
			for i=1,(n_elements(indecis)-1) do Title_String=Title_String+'	'+RowNames[indecis[i]]
			close,2
			openw,2,SaveASCII_Filename,width=1024
			printf,2,Title_String
			printf,2,FGroupParams[indecis,*],FORMAT='('+strtrim((n_elements(indecis)-1),2)+'(E13.5,"'+string(9B)+'"),E13.5)'
			close,2
		endif
	endif
endfor

widget_control,event.top,/destroy

;Dist_Results=[Nph_mean,Nph_std,cnt,XPos,YPos,ZPos,UnwZPos,FWHM[0],FWHM[1],FWHM[2],FWHM[3],Xstd,Ystd,Zstd,UnwZstd,X90,Y90,Z90,UnwZ90]

if n_elements(results) lt 1 then return

device,decompose=0
loadct,12
!p.multi=[0,1,2,0,0]

istart=[7,15]
istop=istart+3
xttl= ['Distribution FWHM (nm)','Distribution of Width @'+fr_srch_str+'% (nm)']
xw=['X FWHM = ','X width @'+fr_srch_str+'% =']
yw=['Y FWHM = ','Y width @'+fr_srch_str+'% =']
if CGrpSize eq 49 then begin
	zw=['Z FWHM = ','Z width @'+fr_srch_str+'% =']
	unwrzw=['Unwr Z FWHM = ','Unwr Z width @'+fr_srch_str+'% =']


for i=0,1 do begin
	nbns=25
	mn = min(Results[*,istart[i]:istop[i]])
	mx = max(Results[*,istart[i]:istop[i]])
	binsize=(mx-mn)/(nbns-1.0)
	xx=fltarr(2*nbns)
	histhist=fltarr(2*nbns)
	evens=2*indgen(nbns)
	odds=evens+1
	x=findgen(nbns)/nbns*(mx-mn)+mn
	dx=0.5 * (mx-mn) / nbns
	xx[evens]=x-dx
	xx[odds]=x+dx
	hist_count=intarr(4)
	lbl_color = [200,40,105,150]
	for ix=istart[i],istop[i] do begin
		hist=histogram(Results[*,ix],min=mn,max=mx,nbins=nbns)
		histhist[evens]=hist
		histhist[odds]=hist
		if ix eq istart[i] then histhist_multilable=transpose(histhist) else histhist_multilable = [histhist_multilable, transpose(histhist)]
		hist_count[ix-istart[i]]=max(histhist)
		xcoord=xx
	endfor

	yrange_hist=[0, max(hist_count)*1.1]
	xrange_hist = [mn,mx]

	tk=1.5
	plot,xx,histhist_multilable[0,*],xstyle=1, xtitle=xttl[i], ytitle='# of Clusters', $
		thick=1.0, xthick=1.0, ythick=1.0, charsize=tk, charthick=tk, xrange = xrange_hist, yrange = yrange_hist
	oplot,xx,histhist_multilable[0,*], color=lbl_color[0]
	oplot,xx,histhist_multilable[1,*], color=lbl_color[1]
	oplot,xx,histhist_multilable[2,*], color=lbl_color[2]
	oplot,xx,histhist_multilable[3,*], color=lbl_color[3]
	xmean=mean(Results[*,istart[i]])
	ymean=mean(Results[*,istart[i]+1])
	zmean=mean(Results[*,istart[i]+2])
	unwrzmean=mean(Results[*,istart[i]+3])
	xstdev=stdev(Results[*,istart[i]])
	ystdev=stdev(Results[*,istart[i]+1])
	zstdev=stdev(Results[*,istart[i]+2])
	unwrzstdev=stdev(Results[*,istart[i]+3])

	xyouts,0.42,0.95-i*0.5,xw[i]+strtrim(xmean,2)+'  +/-  '+strtrim(xstdev,2)+'    nm ', color=lbl_color[0],charsize=1.5,/NORMAL
	xyouts,0.42,0.93-i*0.5,yw[i]+strtrim(ymean,2)+'  +/-  '+strtrim(ystdev,2)+'    nm ', color=lbl_color[1],charsize=1.5,/NORMAL
	xyouts,0.42,0.91-i*0.5,zw[i]+strtrim(zmean,2)+'  +/-  '+strtrim(zstdev,2)+'    nm ', color=lbl_color[2],charsize=1.5,/NORMAL
	xyouts,0.42,0.89-i*0.5,unwrzw[i]+strtrim(unwrzmean,2)+'  +/-  '+strtrim(unwrzstdev,2)+'    nm ', color=lbl_color[3],charsize=1.5,/NORMAL
endfor

device,decompose=0
loadct,3
!p.multi=[0,0,0,0,0]
!p.background=0
!P.NOERASE=0

WidLabel0 = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
widget_control,WidLabel0,get_value=dfilename
bmp_filename=AddExtension(dfilename,'_ClusterWidth_Distributions.bmp')
presentimage=tvrd(true=1)
write_bmp,bmp_filename,presentimage,/rgb
endif

end
