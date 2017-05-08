;
; Empty stub procedure used for autoloading.
;
pro Analyze_Fiducial_Alignement_2Colors_eventcb
end
;
;-----------------------------------------------------------------
;
pro Initialize_AnalizeMultiple_Fiducials_2Colors, wWidget

Xrange_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Xrange1')
widget_control,Xrange_ID,SET_VALUE = '1.5'

Yrange_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_Yrange1')
widget_control,Yrange_ID,SET_VALUE = '1.5'

Zrange_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_ZRange1')
widget_control,Zrange_ID,SET_VALUE = '100.0'

MinNumberofpeaks_ID = Widget_Info(wWidget, find_by_uname='WID_TEXT_MiNumberofPeaks')
widget_control,MinNumberofpeaks_ID,SET_VALUE = '50'


end
;
;-----------------------------------------------------------------
;
pro On_AnalyzeMultiple_Fiducial_Colocalization_Start, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common hist, xcoord, histhist, xtitle, mult_colors_hist, histhist_multilable, hist_log_x, hist_log_y, hist_nbins, RowNames
common materials, lambda_vac, nd_water, nd_oil, nm_per_pixel
common  AnchorParams,  AnchorPnts,  AnchorFile, ZPnts, Fid_Outl_Sz, AutoDisp_Sel_Fids, Disp_Fid_IDs, AnchPnts_MaxNum, AutoDet_Params, AutoMatch_Params, Adj_Scl, transf_scl, Transf_Meth, PW_deg, XYlimits, Use_XYlimits, LeaveOrigTotalRaw
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WID_Base0_Peakselector, ID:TopID, TOP:TopID, HANDLER:TopID }


X_ind=min(where(RowNames eq 'X Position'))
Y_ind=min(where(RowNames eq 'Y Position'))
GrX_ind=min(where(RowNames eq 'Group X Position'))
GrY_ind=min(where(RowNames eq 'Group Y Position'))
Z_ind=min(where(RowNames eq 'Z Position'))
GrZ_ind=min(where(RowNames eq 'Group Z Position'))
UnwZ_ind=min(where(RowNames eq 'Unwrapped Z'))
UnwGrZ_ind=min(where(RowNames eq 'Unwrapped Group Z'))

wstatus_2color_coloc_analysis = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Status_Fid2Color_Analysis')

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif
ParamLimits0=ParamLimits

Xrange_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_Xrange1')
widget_control,Xrange_ID,GET_VALUE = Xrange_txt
Xrange=float(Xrange_txt[0])

Yrange_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_Yrange1')
widget_control,Yrange_ID,GET_VALUE = Yrange_txt
Yrange=float(Yrange_txt[0])

Zrange_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_ZRange1')
widget_control,Zrange_ID,GET_VALUE = Zrange_txt
Zrange=float(Zrange_txt[0])

MinNumberofpeaks_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_MiNumberofPeaks')
widget_control,MinNumberofpeaks_ID,GET_VALUE = MinNumber_txt
Min_Number_of_Peaks=fix(MinNumber_txt[0])

FidFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_FidFilename')
widget_control,FidFileWidID,GET_VALUE = FidFile
FidFileInfo=file_info(FidFile)
if ~FidFileInfo.exists then begin
	z=dialog_message('Peaks File does not exist')
	return
endif

AnchorPnts=dblarr(6,AnchPnts_MaxNum)
AnchorPnts_line=dblarr(6)
ZPnts=dblarr(3,AnchPnts_MaxNum)
ZPnts_line=dblarr(3)
close,5
openr,5,FidFile
ip=0
while not EOF(5) do begin
	readf,5,AnchorPnts_line
	AnchorPnts[*,ip] = AnchorPnts_line
	ip+=1
endwhile
close,5

Anc_Z_FileInfo=file_info(FidFile+'z')
if Anc_Z_FileInfo.exists then begin
	ip=0
	openr,5,(FidFile+'z')
	while not EOF(5) do begin
		readf,5,ZPnts_line
		ZPnts[*,ip] = ZPnts_line
		ip+=1
	endwhile
	close,5
endif

use_green_button_id=widget_info(event.top,FIND_BY_UNAME='WID_BUTTON_use_green')		;
fid_id=widget_info(use_green_button_id,/button_set)


indices=where(AnchorPnts[2*fid_id,*] ne 0,PeakNum)

PeakX=AnchorPnts[2*fid_id,indices]
PeakY=AnchorPnts[2*fid_id+1,indices]
if Anc_Z_FileInfo.exists then PeakZ=ZPnts[fid_id,indices]

Fid_analysis_results=fltarr(13,PeakNum)

for ip=0,(PeakNum-1) do begin

	ParamLimits[X_ind,0]=PeakX[ip]-Xrange/2.0
	ParamLimits[X_ind,1]=PeakX[ip]+Xrange/2.0
	ParamLimits[X_ind,2]=PeakX[ip]
	ParamLimits[X_ind,3]=Xrange
	ParamLimits[Y_ind,0]=PeakY[ip]-Yrange/2.0
	ParamLimits[Y_ind,1]=PeakY[ip]+Yrange/2.0
	ParamLimits[Y_ind,2]=PeakY[ip]
	ParamLimits[Y_ind,3]=Yrange
	if Anc_Z_FileInfo.exists then begin
		ParamLimits[Z_ind,0]=PeakZ[ip]-Zrange/2.0
		ParamLimits[Z_ind,1]=PeakZ[ip]+Zrange/2.0
		ParamLimits[Z_ind,2]=PeakZ[ip]
		ParamLimits[Z_ind,3]=Zrange
	endif

	OnPeakCentersButton, Event1

	str_out='Peak '+strmid(ip+1,2)+' of '+strmid(PeakNum,2)
   	xyouts,0.27,0.91,str_out,/NORMAL,color=200,charsize=1.5
   	widget_control,wstatus_2color_coloc_analysis,set_value='Analyzing '+str_out
	if total(filter) ge Min_Number_of_Peaks then begin
		red_filter = filter and (CGroupParams[26,*] eq 1)
		red_indecis = where(red_filter eq 1, red_count)
		green_filter = filter and (CGroupParams[26,*] eq 2)
		green_indecis = where(green_filter eq 1, green_count)
		if ((size(red_indecis))[0] gt 0) and ((size(green_indecis))[0] gt 0) then begin
			red_Nph = total(CGroupParams[6,red_indecis])
			red_Nph_mean = mean(CGroupParams[6,red_indecis])
			red_xposition=total(CGroupParams[X_ind,red_indecis]*CGroupParams[6,red_indecis])/red_Nph
			red_yposition=total(CGroupParams[Y_ind,red_indecis]*CGroupParams[6,red_indecis])/red_Nph
			red_zposition=(CGrpSize ge 33) ? total(CGroupParams[Z_ind,red_indecis]*CGroupParams[6,red_indecis])/red_Nph : 0
			green_Nph = total(CGroupParams[6,green_indecis])
			green_Nph_mean = mean(CGroupParams[6,green_indecis])
			green_xposition=total(CGroupParams[X_ind,green_indecis]*CGroupParams[6,green_indecis])/green_Nph
			green_yposition=total(CGroupParams[Y_ind,green_indecis]*CGroupParams[6,green_indecis])/green_Nph
			green_zposition=(CGrpSize ge 33) ? total(CGroupParams[Z_ind,green_indecis]*CGroupParams[6,green_indecis])/green_Nph : 0
			Fid_analysis_results[0,ip] = red_count
			Fid_analysis_results[1,ip] = red_Nph_mean
			Fid_analysis_results[2,ip] = red_xposition
			Fid_analysis_results[3,ip] = red_yposition
			Fid_analysis_results[4,ip] = red_zposition
			Fid_analysis_results[5,ip] = green_count
			Fid_analysis_results[6,ip] = green_Nph_mean
			Fid_analysis_results[7,ip] = green_xposition
			Fid_analysis_results[8,ip] = green_yposition
			Fid_analysis_results[9,ip] = green_zposition
			Fid_analysis_results[10,ip] = (red_xposition - green_xposition)*nm_per_pixel
			Fid_analysis_results[11,ip] = (red_yposition - green_yposition)*nm_per_pixel
			Fid_analysis_results[12,ip] = red_zposition - green_zposition
		endif
	endif
endfor

ParamLimits=ParamLimits0
indecis=where(Fid_analysis_results[0,*] gt 0,cnt)

if cnt gt 0 then begin
	WidLabel0 = Widget_Info(TopID, find_by_uname='WID_LABEL_0')
	widget_control,WidLabel0,get_value=dfilename
	sfilename=AddExtension(dfilename,'_fiducial_coloc.txt')
	finfo=file_info(sfilename)
	close,1
	openw,1,sfilename,width=1024
	printf,1,'Red Npks	Red Nph (mean)	Red X (pixels)	Red Y (pixels)	Red Z (nm)	Green Npks	Green Nph (mean)	Green X (pixels)	Green Y (pixels)	Green Z (nm)	Delta X (nm)	Delta Y (nm)	Delta Z (nm)'
	printf,1,Fid_analysis_results[*,indecis],FORMAT='(12(E12.4,"'+string(9B)+'"),E12.4)'
	close,1
endif

widget_control,event.top,/destroy

nbns=25
mn = min(Fid_analysis_results[10:12,*])
mx = max(Fid_analysis_results[10:12,*])
binsize=(mx-mn)/(nbns-1.0)
xx=fltarr(2*nbns)
histhist=fltarr(2*nbns)
evens=2*indgen(nbns)
odds=evens+1
x=findgen(nbns)/nbns*(mx-mn)+mn
dx=0.5 * (mx-mn) / nbns
xx[evens]=x-dx
xx[odds]=x+dx
hist_count=intarr(3)
lbl_color = [200,40,105]
for ix=10,12 do begin
	hist=histogram(Fid_analysis_results[ix,*],min=mn,max=mx,nbins=nbns)
	histhist[evens]=hist
	histhist[odds]=hist
	if ix eq 10 then histhist_multilable=transpose(histhist) else histhist_multilable = [histhist_multilable, transpose(histhist)]
	hist_count[ix-10]=max(histhist)
	xcoord=xx
endfor

yrange_hist=[0, max(hist_count)*1.1]
xrange_hist = [mn,mx]

device,decompose=0
loadct,12
tk=1.5
plot,xx,histhist_multilable[0,*],xstyle=1, xtitle='2Color Fiducial misalignement (nm)', ytitle='Molecule Count', $
		thick=1.0, xthick=1.0, ythick=1.0, charsize=tk, charthick=tk, xrange = xrange_hist, yrange = yrange_hist
oplot,xx,histhist_multilable[0,*], color=lbl_color[0]
oplot,xx,histhist_multilable[1,*], color=lbl_color[1]
oplot,xx,histhist_multilable[2,*], color=lbl_color[2]
xstdev=stdev(Fid_analysis_results[10,*])
ystdev=stdev(Fid_analysis_results[11,*])
zstdev=stdev(Fid_analysis_results[12,*])
xyouts,0.12,0.95,'STDs of difference between fiducial coordinates in 2 colors. Total #:  '+strtrim(PeakNum,2),charsize=1.5,/NORMAL
xyouts,0.12,0.92,'X coordinate :'+strtrim(xstdev,2)+'    nm ', color=lbl_color[0],charsize=1.5,/NORMAL
xyouts,0.12,0.89,'Y coordinate :'+strtrim(ystdev,2)+'    nm ', color=lbl_color[1],charsize=1.5,/NORMAL
xyouts,0.12,0.86,'Z coordinate :'+strtrim(zstdev,2)+'    nm ', color=lbl_color[2],charsize=1.5,/NORMAL
device,decompose=0
loadct,3

bmp_filename=AddExtension(dfilename,'_fiducial_coloc.bmp')
presentimage=tvrd(true=1)
write_bmp,bmp_filename,presentimage,/rgb

end
;
;-----------------------------------------------------------------
;
pro OnPickFidFile, Event
FidFile = Dialog_Pickfile(/read,get_path=fpath,filter=['*.anc'],title='Select *.anc file to open')
FidFileInfo=file_info(FidFile)
if FidFile ne '' then cd,fpath
FidFileWidID = Widget_Info(Event.Top, find_by_uname='WID_TEXT_FidFilename')
widget_control,FidFileWidID,SET_VALUE = FidFile
end
;
;-----------------------------------------------------------------
;
