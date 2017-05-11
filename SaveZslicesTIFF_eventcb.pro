
; Empty stub procedure used for autoloading.
;
pro SaveZslicesTIFF_eventcb
end
;
;-----------------------------------------------------------------
;
pro Initialize_ZslicesTIFF, wWidget

end
;
;-----------------------------------------------------------------
;
pro On_Zslices_TIFF_Start, Event
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
common display_info, labelcontrast, hue_scale, Max_Prob_2DPALM, def_w
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }

wlabel = Widget_Info(Event1.Top, find_by_uname='WID_LABEL_0')
widget_control,wlabel,get_value=filename0

wstatus = Widget_Info(Event.Top, find_by_uname='WID_TEXT_Status')

if n_elements(CGroupParams) lt 1 then begin
	z=dialog_message('Please load a data file')
	return      ; if data not loaded return
endif

Zstart_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_Zstart')
widget_control,Zstart_ID,GET_VALUE = Zstart_txt
Zstart=float(Zstart_txt[0])

Zstop_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_Zstop')
widget_control,Zstop_ID,GET_VALUE = Zstop_txt
Zstop=float(Zstop_txt[0])

Zstep_ID = Widget_Info(Event.top, find_by_uname='WID_TEXT_Zstep')
widget_control,Zstep_ID,GET_VALUE = Zstep_txt
Zstep=float(Zstep_txt[0])

nz=fix((Zstop-Zstart)/Zstep-0.001)+1
Param40=ParamLimits[40,*]
filter0=filter

FunctionId=widget_info(Event1.top, FIND_BY_UNAME='WID_DROPLIST_Function')
FunctionItem=widget_info(FunctionId,/DropList_Select)
FilterId=widget_info(Event1.top, FIND_BY_UNAME='WID_DROPLIST_Filter')
FilterItem=widget_info(FilterId,/DropList_Select)
AccumId=widget_info(Event1.top, FIND_BY_UNAME='WID_DROPLIST_Accumulate')
AccumItem=widget_info(AccumId,/DropList_Select)
wxsz=1024 & wysz=1024
mgw=GetScaleFactor(ParamLimits, xydsz, wxsz, wysz)
dxsz=xydsz[0] & dysz=xydsz[1]
dxmn = fix(dxsz < paramlimits[2,0])>0 & dymn = fix(dysz < paramlimits[3,0])>0
dxmx = fix(0 > paramlimits[2,1])<dysz & dymx = fix(0 > paramlimits[3,1])<dysz

WidSldTopID = Widget_Info(Event1.Top, find_by_uname='WID_SLIDER_Top')
widget_control,WidSldTopID,get_value=topV
WidSldBotID = Widget_Info(Event1.Top, find_by_uname='WID_SLIDER_Bot')
widget_control,WidSldBotID,get_value=botV
WidSldGammaID = Widget_Info(Event1.Top, find_by_uname='WID_SLIDER_Gamma')
widget_control,WidSldGammaID,get_value=gamma
if topV le botV then begin
	topV = botV+1
	widget_control,WidSldTopID,set_value=TopV
endif

WidImageNormalization = Widget_Info(Event.Top, find_by_uname='WID_DROPLIST_Normalization')
ImgNormalization = widget_info(WidImageNormalization,/DropList_Select)
nlbls=max(CGroupParams[26,*])

if ImgNormalization eq 0 then begin
	widget_control,wstatus,set_value='rendering compound image'
	RenderWithoutAutoscale, Event1
	sz_im=size(image)
	if sz_im[0] eq 2 then begin
		gamma=labelcontrast[1,1]
		image=image^(gamma/100.0)
		img_mn=min(image)
		img_mx=max(image)
	endif else begin
			if sz_im[0] eq 3 then begin
				img_mn=dblarr(nlbls)	& img_mx=dblarr(nlbls)
				for jj=0,nlbls-1 do begin
					gamma=labelcontrast[1,jj+1]
					image[*,*,jj]=image[*,*,jj]^(gamma/100.0)
					img_mn[jj]=min(image[*,*,jj])
					img_mx[jj]=max(image[*,*,jj])
				endfor
			endif
	endelse
	img_scl=(!D.TABLE_SIZE-1)/(img_mx-img_mn)
	if nlbls eq 1 then scaled_image=(image-img_mn)*img_scl else begin
		scaled_image=image
		for jj=0,nlbls-1 do scaled_image[*,*,jj]=(image[*,*,jj]-img_mn[jj])*img_scl[jj]
	endelse
	tv,scaled_image,true=3
endif else begin

	for iz=0,nz-1 do begin
		widget_control,wstatus,set_value='Run0: rendering image '+strmid(iz+1,2)+' of '+strmid(nz,2)
		filter=filter0
		Zmin=Zstart+iz*Zstep
		Zmax=(Zmin+Zstep)<Zstop
		Zrange=Zmax-Zmin
		ParamLimits[40,0]=Zmin
		ParamLimits[40,1]=Zmax
		ParamLimits[40,2]=(Zmin+Zmax)/2
		ParamLimits[40,4]=Zrange
		RenderWithoutAutoscale, Event1
		sz_im=size(image)
		if iz eq 0 then begin
			if sz_im[0] eq 2 then begin
				gamma=labelcontrast[1,1]
				image=image^(gamma/100.0)
				img_mn=min(image)
				img_mx=max(image)
			endif else begin
				if sz_im[0] eq 3 then begin
					img_mn=dblarr(nlbls)	& img_mx=dblarr(nlbls)
					for jj=0,nlbls-1 do begin
						gamma=labelcontrast[1,jj+1]
						image[*,*,jj]=image[*,*,jj]^(gamma/100.0)
						img_mn[jj]=min(image[*,*,jj])
						img_mx[jj]=max(image[*,*,jj])
					endfor
				endif
			endelse
		endif else begin
			if sz_im[0] eq 2 then begin
				gamma=labelcontrast[1,1]
				image=image^(gamma/100.0)
				img_mn=min(image)<img_mn
				img_mx=max(image)>img_mx
			endif else begin
				if sz_im[0] eq 3 then begin
					for jj=0,nlbls-1 do begin
						gamma=labelcontrast[1,jj+1]
						image[*,*,jj]=image[*,*,jj]^(gamma/100.0)
						img_mn[jj]=min(image[*,*,jj])<img_mn[jj]
						img_mx[jj]=max(image[*,*,jj])>img_mx[jj]
					endfor
				endif
			endelse
		endelse
	endfor
endelse
img_scl=(!D.TABLE_SIZE-1)/(img_mx-img_mn)

for iz=0,nz-1 do begin
	filter=filter0
	Zmin=Zstart+iz*Zstep
	Zmax=(Zmin+Zstep)<Zstop
	Zrange=Zmax-Zmin
	ParamLimits[40,0]=Zmin
	ParamLimits[40,1]=Zmax
	ParamLimits[40,2]=(Zmin+Zmax)/2
	ParamLimits[40,4]=Zrange
	widget_control,wstatus,set_value='Rendering and saving image '+strtrim(strmid(iz+1,2))+' of '+strtrim(strmid(nz,2))
	RenderWithoutAutoscale, Event1
	if sz_im[0] eq 2 then begin
		gamma=labelcontrast[1,1]
		image=image^(gamma/100.0)
		scaled_image=(image-img_mn)*img_scl
		tv,scaled_image
		presentimage=reverse(tvrd(true=1),3)
		ext_new='_Z='+strtrim(string(Zmin,FORMAT='(F10.1)'),2)+'nm-'+strtrim(string(Zmax,FORMAT='(F10.1)'),2)+'nm.tiff'
		filename=AddExtension(filename0,ext_new)
		write_tiff,filename,presentimage,orientation=1
	endif else begin
		if sz_im[0] eq 3 then begin
			for jj=0,nlbls-1 do begin
				scaled_image=image*0.
				gamma=labelcontrast[1,jj+1]
				scaled_image[*,*,jj]=(image[*,*,jj]^(gamma/100.0)-img_mn[jj])*img_scl[jj]
				tv,scaled_image,true=3
				presentimage=reverse(tvrd(true=1),3)
				ext_new='_lbl='+strtrim(string(jj,FORMAT='(I1)'),2)+'_Z='+strtrim(string(Zmin,FORMAT='(F10.1)'),2)+'nm-'+strtrim(string(Zmax,FORMAT='(F10.1)'),2)+'nm.tiff'
				filename=AddExtension(filename0,ext_new)
				write_tiff,filename,presentimage,orientation=1
				wait,0.25
			endfor
		endif
	endelse
endfor
ParamLimits[40,*]=Param40
filter=filter0
widget_control,event.top,/destroy

end

