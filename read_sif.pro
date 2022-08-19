; - Replaced ReadTillNewSpaceSpace() with ReadTillNewSpace() in the version read of TinstaImage
; - Replaced franes_in_chunk with frames_in_chunk in TImage
; - Added number_of_chunks = number_of_chunks < TImageParams.no_images	to make sure that the number of chunks does not
; exceed the number of images
; - replaced i with number_of_chunks in two places in remainder reading inside TImage
; -  modified the TImage procedure so that the remainder frames are always read but only returned as ThisData if requested.
; initially they were not read if the remainedr was not requested, which caused errors reading the rest of the data
; - replaced "data" with "thisdata" in the initial line thisdata=reform(data,xpix,ypix,frames_in_chunk)inside TImage
; - added creation and saving the data into .dat file and .txt file
; -------------------------------------
Function ReadTillNewLine
c=0b
cc=32b
while (c ne 10b) do begin
	readu,1,c
	cc=[[cc],c]
endwhile
return,strtrim(cc,2)
end
; -------------------------------------
Function ReadTillNewSpace
c=0b
cc=32b
while (c ne 32b) do begin
	readu,1,c
	cc=[[cc],c]
endwhile
return,strtrim(cc,2)
end
; -------------------------------------
Function ReadTillNewSpaceSpace
c=0b
cc=32b
while (c ne 32b) do begin
	readu,1,c
	cc=[[cc],c]
endwhile
readu,1,c
if c ne 32b then print,'error on read till new space space'
return,strtrim(cc,2)
end
; -------------------------------------
Function ReadLengthString
len					=ReadTillNewLine()
if fix(len) gt 0 then begin
	text=bytarr(len)
	readu,1,text
	text=strtrim(text,2)
endif else text = 'null'
return,text
end
; -------------------------------------
Function TUserText
Text=''
version				=ReadTillNewSpace()
;if version eq '0' then return, Text
if version ne '65538' then print,version+' TUserText version not expected'
Text				=ReadLengthString()
null				=ReadTillNewLine()
return, Text
end
; -------------------------------------
Function TShutter,TinstaImageVersion
if (TinstaImageVersion) ne '65567'then version=	ReadTillNewSpaceSpace() else version = ReadTillNewSpace()

;if version eq '0' then return, TShutterParams
if version ne '65538' then print,version+' TShutter version not expected'
	TShutterParams={	$
		type:				'',$
		mode:				'',$
		custom_bg_mode:		'',$
		custom_mode:		'',$
		closing_time:		'',$
		opening_time:		''}
	TShutterParams={	$
		type:				ReadTillNewSpace(),$
		mode:				ReadTillNewSpace(),$
		custom_bg_mode:		ReadTillNewSpace(),$
		custom_mode:		ReadTillNewSpace(),$
		closing_time:   ReadTillNewSpace(),$
		opening_time:		ReadTillNewLine()}
return, TShutterParams
end
; -------------------------------------
Function TShamrockSave
version=		ReadTillNewSpace()
;print,'TShamrock vesrion:  ', version
;if version ne '65536' then print,version+' TShamrockSave version not expected'
	TShamrockSaveParams={	$
		isActive:			ReadTillNewSpace(),$
		waveDrivePresent:	ReadTillNewSpace(),$
		wavelength:			ReadTillNewSpace(),$
		gratingTurretPresent:	ReadTillNewSpace(),$
		grating:			ReadTillNewSpace(),$
		gratingLines:		ReadTillNewSpace(),$
		gratingBlaze:		ReadTillNewSpace(),$
		slitPresent:		ReadTillNewSpace(),$
		slitWidth:			ReadTillNewSpace(),$
		flipperMirrorPresent:	ReadTillNewSpace(),$
		flipperPort:		ReadTillNewSpace(),$
		filterPresent:		ReadTillNewSpace(),$
		filterIndex:		ReadTillNewSpace(),$
	;	len:				ReadTillNewSpace(),$		?????
		filterLabel:		ReadTillNewSpace(),$
		accessoryAttached:		ReadTillNewSpace(),$
		port1State:			ReadTillNewSpace(),$
		port2State:			ReadTillNewSpace(),$
		port3State:			ReadTillNewSpace(),$
		inputPortState:		ReadTillNewSpace(),$
		outputSlitPresent:	ReadTillNewSpace(),$
		outputSlitWidth:	ReadTillNewSpace()	}
		null=		ReadTillNewLine()
if version ge 65540 then begin
	;print,'TShamrock:reading extra lines
	while(strmatch(null,'65539*') ne 1) do begin
	   null=ReadTillNewLine()
	endwhile
	null = ReadTillNewLine()
	while(strmatch(null,'65538*') ne 1) do begin
	   null=ReadTillNewLine()
  endwhile

;	for i=0,9 do	null=		ReadTillNewLine()
endif
return, TShamrockSaveParams
end
; -------------------------------------
Function TInstaImage
version	=		ReadTillNewSpace()
print,"TInstaImage: ", version
;if (version ne '65552') and (version ne '65555') then print,version+' TInstaImage version not expected'
if version eq '65552' then begin
InstaParams={		$
	type:				ReadTillNewSpace(),$
	active:				ReadTillNewSpace(),$
	structure:			ReadTillNewSpace(),$
	timedate:			ReadTillNewSpace(),$
	temperature:		ReadTillNewSpace(),$
	head:				byte(ReadTillNewSpace()),$
	store_type:			byte(ReadTillNewSpace()),$
	data_type:			byte(ReadTillNewSpace()),$
	mode:				byte(ReadTillNewSpace()),$
	trigger_source:		byte(ReadTillNewSpace()),$
	trigger_level:		ReadTillNewSpace(),$
	exposure_time:		ReadTillNewSpace(),$
	delay:				ReadTillNewSpace(),$
	integration_cycle_time:	ReadTillNewSpace(),$
	no_integrations:	ReadTillNewSpace(),$
	sync:				byte(ReadTillNewSpace()),$
	kinetic_cycle_time:	ReadTillNewSpace(),$
	pixel_readout_time:	ReadTillNewSpace(),$
	no_points:			ReadTillNewSpace(),$
	fast_track_height:	ReadTillNewSpace(),$
	gain:				ReadTillNewSpace(),$
	gate_delay:			ReadTillNewSpace(),$
	gate_width:			ReadTillNewSpace(),$
	gate_step:			ReadTillNewSpace(),$
	track_height:		ReadTillNewSpace(),$
	series_length:		ReadTillNewSpace(),$
	read_pattern:		byte(ReadTillNewSpace()),$
	shutter_delay:		byte(ReadTillNewSpace()),$
	st_center_row:		ReadTillNewSpace(),$
	mt_offset:			ReadTillNewSpace(),$
	operation_mode:		ReadTillNewSpace(),$
	FlipX:				ReadTillNewSpace(),$
	FlipY:				ReadTillNewSpace(),$
	Clock:				ReadTillNewSpace(),$
	AClock:				ReadTillNewSpace(),$
	MCP:				ReadTillNewSpace(),$
	Prop:				ReadTillNewSpace(),$
	IOC:				ReadTillNewSpace(),$
	Freq:				ReadTillNewSpace(),$
	VertClockAmp:		ReadTillNewSpace(),$
	data_v_shift_speed:	ReadTillNewSpace(),$
	OutputAmp:			ReadTillNewSpace(),$
	PreAmpGain:			ReadTillNewSpace(),$
	Serial:				ReadTillNewSpace(),$
	NumPulses:			ReadTillNewSpace(),$
	mFrameTransferAcqMode:ReadTillNewSpace(),$
	unstabilizedTemperature:ReadTillNewSpace(),$
	mBaseLineClamp:		ReadTillNewSpace(),$
	mPreScan:			ReadTillNewSpace(),$
	mEMRealGain:		'',$
	mBaseLineOffset:	'',$
	null0:				'',$
	head_model:			ReadLengthString(),$
	null1:				ReadTillNewLine(),$
	null2:				ReadTillNewSpace(),$
	detector_format_x:	ReadTillNewSpace(),$
	detector_format_y:	ReadTillNewSpace(),$
	filename:			ReadLengthString(),$
	null3:				ReadTillNewLine(),$
	TUserTextParams:	TUserText(),$
	TShutterParams:		TShutter(version),$
	TShamrockSaveParams:	TShamrockSave()  }
endif
if version eq '65555' then begin
InstaParams={		$
	type:				ReadTillNewSpace(),$
	active:				ReadTillNewSpace(),$
	structure:			ReadTillNewSpace(),$
	timedate:			ReadTillNewSpace(),$
	temperature:		ReadTillNewSpace(),$
	head:				byte(ReadTillNewSpace()),$
	store_type:			byte(ReadTillNewSpace()),$
	data_type:			byte(ReadTillNewSpace()),$
	mode:				byte(ReadTillNewSpace()),$
	trigger_source:		byte(ReadTillNewSpace()),$
	trigger_level:		ReadTillNewSpace(),$
	exposure_time:		ReadTillNewSpace(),$
	delay:				ReadTillNewSpace(),$
	integration_cycle_time:	ReadTillNewSpace(),$
	no_integrations:	ReadTillNewSpace(),$
	sync:				byte(ReadTillNewSpace()),$
	kinetic_cycle_time:	ReadTillNewSpace(),$
	pixel_readout_time:	ReadTillNewSpace(),$
	no_points:			ReadTillNewSpace(),$
	fast_track_height:	ReadTillNewSpace(),$
	gain:				ReadTillNewSpace(),$
	gate_delay:			ReadTillNewSpace(),$
	gate_width:			ReadTillNewSpace(),$
	gate_step:			ReadTillNewSpace(),$
	track_height:		ReadTillNewSpace(),$
	series_length:		ReadTillNewSpace(),$
	read_pattern:		byte(ReadTillNewSpace()),$
	shutter_delay:		byte(ReadTillNewSpace()),$
	st_center_row:		ReadTillNewSpace(),$
	mt_offset:			ReadTillNewSpace(),$
	operation_mode:		ReadTillNewSpace(),$
	FlipX:				ReadTillNewSpace(),$
	FlipY:				ReadTillNewSpace(),$
	Clock:				ReadTillNewSpace(),$
	AClock:				ReadTillNewSpace(),$
	MCP:				ReadTillNewSpace(),$
	Prop:				ReadTillNewSpace(),$
	IOC:				ReadTillNewSpace(),$
	Freq:				ReadTillNewSpace(),$
	VertClockAmp:		ReadTillNewSpace(),$
	data_v_shift_speed:	ReadTillNewSpace(),$
	OutputAmp:			ReadTillNewSpace(),$
	PreAmpGain:			ReadTillNewSpace(),$
	Serial:				ReadTillNewSpace(),$
	NumPulses:			ReadTillNewSpace(),$
	mFrameTransferAcqMode:ReadTillNewSpace(),$
	unstabilizedTemperature:ReadTillNewSpace(),$
	mBaseLineClamp:		ReadTillNewSpace(),$
	mPreScan:			ReadTillNewSpace(),$
	mEMRealGain:		ReadTillNewSpace(),$
	mBaseLineOffset:	ReadTillNewSpace(),$
	null0:				ReadTillNewLine(),$
	head_model:			ReadTillNewSpace(),$
	null1:				ReadTillNewSpace(),$
	detector_format_x:	ReadTillNewSpace(),$
	detector_format_y:	ReadTillNewSpace(),$
	null2:				ReadTillNewLine(),$
	filename:			ReadTillNewSpace(),$
	null3:				ReadTillNewLine(),$
	TUserTextParams:	TUserText(),$
	TShutterParams:		TShutter(version),$
	TShamrockSaveParams:	TShamrockSave()  }
end
if version eq '65567' then begin
    InstaParams={   $
      type:       ReadTillNewSpace(),$
      active:       ReadTillNewSpace(),$
      structure:      ReadTillNewSpace(),$
      timedate:     ReadTillNewSpace(),$
      temperature:    ReadTillNewSpace(),$
      head:       byte(ReadTillNewSpace()),$
      store_type:     byte(ReadTillNewSpace()),$
      data_type:      byte(ReadTillNewSpace()),$
      mode:       byte(ReadTillNewSpace()),$
      trigger_source:   byte(ReadTillNewSpace()),$
      trigger_level:    ReadTillNewSpace(),$
      exposure_time:    ReadTillNewSpace(),$
      delay:        ReadTillNewSpace(),$
      integration_cycle_time: ReadTillNewSpace(),$
      no_integrations:  ReadTillNewSpace(),$
      sync:       byte(ReadTillNewSpace()),$
      kinetic_cycle_time: ReadTillNewSpace(),$
      pixel_readout_time: ReadTillNewSpace(),$
      no_points:      ReadTillNewSpace(),$
      fast_track_height:  ReadTillNewSpace(),$
      gain:       ReadTillNewSpace(),$
      gate_delay:     ReadTillNewSpace(),$
      gate_width:     ReadTillNewSpace(),$
      gate_step:      ReadTillNewSpace(),$
      track_height:   ReadTillNewSpace(),$
      series_length:    ReadTillNewSpace(),$
      read_pattern:   byte(ReadTillNewSpace()),$
      shutter_delay:    byte(ReadTillNewSpace()),$
      st_center_row:    ReadTillNewSpace(),$
      mt_offset:      ReadTillNewSpace(),$
      operation_mode:   ReadTillNewSpace(),$
      FlipX:        ReadTillNewSpace(),$
      FlipY:        ReadTillNewSpace(),$
      Clock:        ReadTillNewSpace(),$
      AClock:       ReadTillNewSpace(),$
      MCP:        ReadTillNewSpace(),$
      Prop:       ReadTillNewSpace(),$
      IOC:        ReadTillNewSpace(),$
      Freq:       ReadTillNewSpace(),$
      VertClockAmp:   ReadTillNewSpace(),$
      data_v_shift_speed: ReadTillNewSpace(),$
      OutputAmp:      ReadTillNewSpace(),$
      PreAmpGain:     ReadTillNewSpace(),$
      Serial:       ReadTillNewSpace(),$
      NumPulses:      ReadTillNewSpace(),$
      mFrameTransferAcqMode:ReadTillNewSpace(),$
      unstabilizedTemperature:ReadTillNewSpace(),$
      mBaseLineClamp:   ReadTillNewSpace(),$
      mPreScan:     ReadTillNewSpace(),$
      mEMRealGain:    ReadTillNewSpace(),$
      mBaseLineOffset:  ReadTillNewSpace(),$
      null0:        ReadTillNewLine(),$
      head_model:     ReadTillNewSpace(),$
      null1:        ReadTillNewSpace(),$
      detector_format_x:  ReadTillNewSpace(),$
      detector_format_y:  ReadTillNewSpace(),$
      null2:        ReadTillNewLine(),$
      filename:     ReadTillNewSpace(),$
      null3:        ReadTillNewLine(),$
      TUserTextParams:  TUserText(),$
      TShutterParams:   TShutter(version),$
      TShamrockSaveParams:  TShamrockSave()  }
endif
return, InstaParams
end
; -------------------------------------
Function TCalibImage
version=			ReadTillNewSpace()
print,'TcalibImage vesrion:', version
if version eq '65539' then begin
TCalibImageParams={	$
	x_type:			byte(ReadTillNewSpace()),$
	x_unit:			byte(ReadTillNewSpace()),$
	y_type:			byte(ReadTillNewSpace()),$
	y_unit:			byte(ReadTillNewSpace()),$
	z_type:			byte(ReadTillNewSpace()),$
	z_unit:			byte(ReadTillNewLine()),$
	x_cal0:			ReadTillNewSpace(),$
	x_cal1:			ReadTillNewSpace(),$
	x_cal2:			ReadTillNewSpace(),$
	x_cal3:			ReadTillNewLine(),$
	y_cal0:			ReadTillNewSpace(),$
	y_cal1:			ReadTillNewSpace(),$
	y_cal2:			ReadTillNewSpace(),$
	y_cal3:			ReadTillNewLine(),$
	z_cal0:			ReadTillNewSpace(),$
	z_cal1:			ReadTillNewSpace(),$
	z_cal2:			ReadTillNewSpace(),$
	z_cal3:			ReadTillNewLine(),$
	rayleigh_wavelength:	ReadTillNewLine(),$
	pixel_length:	ReadTillNewLine(),$
	pixel_height:	ReadTillNewLine(),$
	xtext:			ReadLengthString(),$
	ytext:			ReadLengthString(),$
	ztext:			ReadLengthString() }
	endif
	if version eq '65540' then begin
	TCalibImageParams={  $
	  x_type:     byte(ReadTillNewSpace()),$
	  x_unit:     byte(ReadTillNewSpace()),$
	  y_type:     byte(ReadTillNewSpace()),$
	  y_unit:     byte(ReadTillNewSpace()),$
	  z_type:     byte(ReadTillNewSpace()),$
	  z_unit:     byte(ReadTillNewLine()),$
	  x_cal0:     ReadTillNewSpace(),$
	  x_cal1:     ReadTillNewSpace(),$
	  x_cal2:     ReadTillNewSpace(),$
	  x_cal3:     ReadTillNewLine(),$
	  y_cal0:     ReadTillNewSpace(),$
	  y_cal1:     ReadTillNewSpace(),$
	  y_cal2:     ReadTillNewSpace(),$
	  y_cal3:     ReadTillNewLine(),$
	  z_cal0:     ReadTillNewSpace(),$
	  z_cal1:     ReadTillNewSpace(),$
	  z_cal2:     ReadTillNewSpace(),$
	  z_cal3:     ReadTillNewLine(),$
	  rayleigh_wavelength:  ReadTillNewLine(),$
	  pixel_length: ReadTillNewLine(),$
	  pixel_height: ReadTillNewLine(),$
	  nothing: ReadTillNewLine(),$
	  xtext:      ReadLengthString(),$
	  ytext:      ReadLengthString(),$
	  ztext:      ReadLengthString() }
	  endif else begin print, 'version not expected in TCalibImageParams'
	  endelse

	;print,TCalibImageParams
return,TCalibImageParams
end
; -------------------------------------
Function TSubImage
TSubImageParams={	$
	left:				ReadTillNewSpace(),$
	top:				ReadTillNewSpace(),$
	right:				ReadTillNewSpace(),$
	bottom:				ReadTillNewSpace(),$
	vertical_bin:		ReadTillNewSpace(),$
	horizontal_bin:		ReadTillNewSpace() }
return,TSubImageParams
end
; -------------------------------------
Function ParseConfig,pth_config,Prm_name,Prm_index
Prm_array=fltarr(Prm_index+1)
Prm_name_to_read = Prm_name+'='
Prm_name_read = Prm_name_to_read
a=0b
openr,5,pth_config
readu,5,Prm_name_read
while (Prm_name_read ne Prm_name_to_read) and ~EOF(5) do begin
	readu,5,a
	Prm_name_read=strmid(Prm_name_read,1,strlen(Prm_name_to_read)-1)+string(a)
	endwhile
if ~EOF(5) then readf,5,Prm_array
close,5
return,Prm_array[Prm_index]
end

; -------------------------------------
Pro TImage, this_chunk, number_of_chunks, TImageParams, InstaParams, ThisData, pth, filen
reffilename=pth+filen+'.dat'			;the filename with extension .dat
;********* Status Bar Initialization  ******************
oStatusBar = obj_new('PALM_StatusBar', COLOR=[0,0,255], TEXT_COLOR=[255,255,255],   TITLE='Converting SIF file...', TOP_LEVEL_BASE=tlb)
fraction_complete_last=0.0
pr_bar_inc=0.01
TImage_version=ReadTillNewSpace()

;if version ne '65538' then message,version+' TImage version not expected'
close,3
if reffilename ne '' then openw,3,reffilename
TImageParams={	$
	image_format_left:	ReadTillNewSpace(),$
	image_format_top:	ReadTillNewSpace(),$
	image_format_right:	ReadTillNewSpace(),$
	image_format_bottom:	ReadTillNewSpace(),$
	no_images:			ReadTillNewSpace(),$
	no_sub_images:		ReadTillNewSpace(),$
	total_length:		ReadTillNewSpace(),$
	image_length:		ReadTillNewLine() }
for i=0,(long(TImageParams.no_sub_images)-1L) do begin
	version = ReadTillNewSpace()
	if version ne '65538' then message,version+' T(SUB)Image version not expected'
	TSubImageParams = TSubImage()
	subImage_offset = ReadTillNewLine()
endfor
time_stamp=strarr(long(TImageParams.no_images))
xpix=abs(fix(TSubImageParams.right)-fix(TSubImageParams.left))+1
ypix=abs(fix(TSubImageParams.bottom)-fix(TSubImageParams.top))+1

for i=0L,(long(TImageParams.no_images)-1L) do time_stamp[i] =	ReadTillNewLine()

	;; added RS according Pierre			; start of removed by GS
	;null = ReadTillNewLine()
	;for i=0L,(long(TImageParams.no_images)-1L) do $
	;  null = ReadTillNewLine()
	;  ;End RS								; send of removed by GS

	; added by GS to be executed only if  TImage_version ge 65541
	if TImage_version ge 65541 then begin
		;calculate the data start position
		sif_info=file_info(pth+filen+'.sif')
		frame_size=4UL*xpix*ypix

		; this was here before 04.02.2019 - GS (probably for NIH/FEI ?)
		;start_position=sif_info.size-282-frame_size*ulong(TImageParams.no_images)
		; this was added instead on 04.02.2019 - GS
		;start_position=sif_info.size-1116-frame_size*ulong(TImageParams.no_images)

		; Now combine the two of the above with selector on head_model - GS 08.2022
		start_shift= 1116  ;  works for old cameras, head_model='DU897_EXF'
		if instaparams.head_model eq 'DU897_EXF' then start_shift = 282
		print, 'Head Model :',instaparams.head_model, '  start_shift=', start_shift
		start_position=sif_info.size - start_shift - frame_size*ulong(TImageParams.no_images)

		print,TImageParams.no_images, xpix, ypix, start_position

		print,'Timage version: ',TImage_version, '       Calculated start position: ',start_position
		point_lun,1,start_position
	endif else print,'Timage version: ',TImage_version

number_of_chunks = number_of_chunks < long(TImageParams.no_images)	; make sure that the number of chunks does not
																; exceed the number of images
;print,'number_of_chunks=',number_of_chunks
this_chunk = fix(this_chunk < number_of_chunks,type=14)>0L		;make sure requested chunk is in range
frames_in_chunk = fix(long(TImageParams.no_images)/number_of_chunks,type=14)
if TImage_version le 65541 then data=fltarr(long(TImageParams.image_length),frames_in_chunk) else data=lonarr(long(TImageParams.image_length),frames_in_chunk)
for i=0,number_of_chunks-1 do begin						;read data in chunks, & put this_chunk of frames into thisdata
	readu,1,data
	if i eq this_chunk then thisdata=data
	if reffilename ne '' then begin
		writedata=reform(uint(round(data)),xpix,ypix,frames_in_chunk)
	 	writeu,3,writedata
	 	fraction_complete=float(frames_in_chunk)*(i+1)/TImageParams.no_images
		oStatusBar -> UpdateStatus, fraction_complete
		wait,0.5
	 endif
endfor
frames_in_remainder=long(TImageParams.no_images) - number_of_chunks*frames_in_chunk

if frames_in_remainder ge 1 then begin						; always read the remainder frames
	if TImage_version le 65541 then data=fltarr(long(TImageParams.image_length),frames_in_remainder) else data=lonarr(long(TImageParams.image_length),frames_in_remainder)
	readu,1,data
	if reffilename ne '' then begin
		writedata=reform(uint(round(data)),xpix,ypix,frames_in_remainder)
	 	writeu,3,writedata
	endif
endif
;if remaining frames requested then set them to be Thisdata to be returned
if (this_chunk ge number_of_chunks) and (TImageParams.no_images gt number_of_chunks*frames_in_chunk) then begin
	frames_in_chunk=frames_in_remainder
	thisdata=data
endif
thisdata=reform(thisdata,xpix,ypix,frames_in_chunk)
if reffilename ne '' then close,3
obj_destroy, oStatusBar ;********* Status Bar Close ******************
return
end
;
; -------------------------------------
;
Pro Read_Sif,pth,reffilename,config_path,ini_filename, thisfitcond
close,1
openr,1,pth+reffilename
def_wind=!D.window
if (size(thisfitcond))[2] ne 8 then LoadThiFitCond,ini_filename,thisfitcond

text=ReadTillNewLine()
version=ReadTillNewSpace()
if version ne '65538' then message,version+' Main Sif version not expected'
SignalPresent=ReadTillNewLine()
;print,"Signal Present=",SignalPresent
If Fix(SignalPresent) eq 1 then begin
filen=strmid(reffilename,0,strlen(reffilename)-4)
	InstaParams=TInstaImage()
	null0_string=strsplit(InstaParams.null0,' ',/EXTRACT)
	sensitivity_from_SIF_header=0
	null0_nums=double(strtrim(null0_string,2))
	if (InstaParams.gain gt 0) and (n_elements(null0_nums) ge 25) then begin
		if (null0_nums[11] ne 0)  then sensitivity_from_SIF_header=InstaParams.gain/null0_nums[11]
	endif
	TCalibImageParams=TCalibImage()
	this_chunk = 0L
	number_of_chunks = 25L

	TImage, this_chunk, number_of_chunks, TImageParams, InstaParams, Data, pth, filen
   ;print,"TImage Conversion Complete"
	; ****************** create .txt file ******************************
		npix=size(Data)
		xpix=npix[1]
		ypix=npix[2]
		if sensitivity_from_SIF_header gt 0 then begin
			print,"SIF file header contains information on camera sensitivity and dark count"
			CCD_Sensitivity = sensitivity_from_SIF_header ;if the SIF header had valid sensitivity data, then use it.
			CCD_Dark_Count = (InstaParams.mBaseLineClamp eq '0')	?	1000.00	:	100.00
		endif else begin
			print,"SIF file header does not contain information on camera sensitivity"
			sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
			configfilename=config_path+sep+'Andor_iXon_SN_'+InstaParams.Serial+'.ini'
			conf_info=file_info(configfilename)
			if ~conf_info.exists then begin	; if the SN is garbled then try SN=1880
				print,'The Camera config file ',configfilename,' does not exist, will try ..\Andor_iXon_SN_1880.ini  instead'
				configfilename=config_path+sep+'Andor_iXon_SN_1880.ini'
				conf_info=file_info(configfilename)
			endif
			if conf_info.exists then begin
				print,'The Camera config file ',configfilename,' does exist, will parse it to extract CCD sensitivity information"
				Head_model=fix(ParseConfig(configfilename,'Head',0))
				Pix_readout_rate=fix(1/float(InstaParams.pixel_readout_time)/1.0e6)
				case Pix_readout_rate of
					1: CCD_Prm='1MHz_16bit_EM_CCD_Sensitivity'
					3: CCD_Prm='3MHz_14bit_EM_CCD_Sensitivity'
					5: CCD_Prm='5MHz_14bit_EM_CCD_Sensitivity'
					10: CCD_Prm='10MHz_14bit_EM_CCD_Sensitivity'
				else:	CCD_Prm='3MHz_14bit_EM_CCD_Sensitivity'
				endcase
				case Pix_readout_rate of
					1: CCD_Prm2='1MHz_16bit_EM_CCD_Base_level'
					3: CCD_Prm2='3MHz_14bit_EM_CCD_Base_level'
					5: CCD_Prm2='5MHz_14bit_EM_CCD_Base_level'
					10: CCD_Prm2='10MHz_14bit_EM_CCD_Base_level'
				else:	CCD_Prm2='3MHz_14bit_EM_CCD_Base_level'
				endcase
				case Pix_readout_rate of
					1: CCD_Prm3='1MHz_16bit_EM_CCD_Gain_at_255'
					3: CCD_Prm3='3MHz_14bit_EM_CCD_Gain_at_255'
					5: CCD_Prm3='5MHz_14bit_EM_CCD_Gain_at_255'
					10: CCD_Prm3='10MHz_14bit_EM_CCD_Gain_at_255'
				else:	CCD_Prm3='3MHz_14bit_EM_CCD_Gain_at_255'
				endcase

				CCD_Prm_index= (InstaParams.PreAmpGain le 1.5) ?	0 : ((InstaParams.PreAmpGain ge 4) ? 2 : 1)

				if Head_model eq 887 then CCD_Prm_index=0
				CCD_Sensitivity_File_Data=ParseConfig(configfilename,CCD_Prm,CCD_Prm_index)
				CCD_Dark_Count=ParseConfig(configfilename,CCD_Prm2,CCD_Prm_index)

				if Head_model eq 887 then begin
					; For older 887 cameras the data sheets have only one CCD sensitivity value
					; for "optimal PreAmg gain setting". It is assumed here that CCD sensitivity in
					; electrons/count is inversely proportional to PreAmp gain setting.
					; The numbers from the factory datasheets have been used to calculate CCD sensitivity
					; for PreAmp Gain setting of 1, and this value is saved in Andor_iXon_SN_XXXX.ini files
					; To get the actual CCD sesnitivity for a given PreAmp gain setting one needs to devide
					; it by the value of PreAmp Gain.
					; Now, to get the CCD Sensitivity for the maximum EM Gain (only max value is used),
					; one needs to divide the above value by tabulated value of EM Gain for EM gain Setting=255
					; Finally, the CCD Sensitivity in A/D Counts/electron is inverse of teh previous quantity.
					EM_gain=ParseConfig(configfilename,CCD_Prm3,0)
					CCD_Sensitivity=EM_Gain /(CCD_Sensitivity_File_Data/float(InstaParams.PreAmpGain))
				endif else begin
					; For newer 897 cameras the values of CCD sensitivity for all PreAmp values have been
					; measured and tabulated. To get the Gain value for appropriate value of EM gain
					; one needs to devide it by the EM gain and invert.
					EM_gain=float(InstaParams.Gain)

					CCD_Sensitivity = EM_Gain / CCD_Sensitivity_File_Data
					CCD_Dark_Count = (InstaParams.mBaseLineClamp eq '0')	?	CCD_Dark_Count	:	100.00	; if the BaseClamp is OFF, use tabulated CCD base level, if BaseClamp is ON , use 100
				endelse
			endif else begin
				print,"The Camera config file ",configfilename," does not exist"
				CCD_Sensitivity=58.3
				CCD_Dark_Count=1000
			endelse
	endelse
		print,"Setting CCD_Sensitivity=",CCD_Sensitivity
		print,"Setting CCD_dark_count=",CCD_Dark_Count
		ref_filename=(pth+filen+'.txt')
		thisfitcond.f_info = filen						; filename wo extension
		thisfitcond.zerodark = CCD_Dark_Count			; Zero Dark Counts
		thisfitcond.xsz = xpix							; x-size (pixels)
		thisfitcond.ysz = ypix							; y-size (pixels)
		thisfitcond.Nframesmax = TImageParams.no_images	; max Number of Frames
		thisfitcond.FrmN = TImageParams.no_images-1L	; No Last Frame
		thisfitcond.Cntpere = CCD_Sensitivity			; Counts per electron
		close,2
		openw, 2, ref_filename
		printf, 2, pth
		printf, 2, filen								; filename wo extension
		printf, 2, thisfitcond.zerodark					; Zero Dark Counts
		printf, 2, thisfitcond.xsz						; x-size (pixels)
		printf, 2, thisfitcond.ysz						; y-size (pixels)
		printf, 2, thisfitcond.Nframesmax				; max Number of Frames
		printf, 2, thisfitcond.Frm0						; No First Frame
		printf, 2, thisfitcond.FrmN						; No Last Frame
		printf, 2, thisfitcond.THresholdcriteria		; Threshold Fit Criteria
		printf, 2, thisfitcond.filetype					; File Type
		printf, 2, thisfitcond.LimBotA1					; Min Peak Amplitude
		printf, 2, thisfitcond.LimTopA1					; Max Peak Amplitude
		printf, 2, thisfitcond.LimBotSig				; Min Peak Width
		printf, 2, thisfitcond.LimTopSig				; Max Peak Width
		printf, 2, thisfitcond.LimChiSq					; Limit ChiSq
		printf, 2, thisfitcond.Cntpere					; Counts per electron
		printf, 2, thisfitcond.maxcnt1					; Max peak # in a single frame, Iteration 1
		printf, 2, thisfitcond.maxcnt2					; Max peak # in a single frame, Iteration 2
		printf, 2, thisfitcond.fliphor					; Flip Horisontal
		printf, 2, thisfitcond.flipvert					; Flip Vertical
		printf, 2, thisfitcond.SigmaSym					; SigmaSym
		printf, 2, thisfitcond.MaskSize					; Gaussian Mask (half) size (parameter d)
		printf, 2, thisfitcond.GaussSig
		printf, 2, thisfitcond.MaxBlck
		printf, 2, thisfitcond.LocalizationMethod
		printf, 2, thisfitcond.SparseOversampling
		printf, 2, thisfitcond.SparseLambda
		printf, 2, thisfitcond.SparseDelta
		printf,	2, thisfitcond.SpError
		printf,	2, thisfitcond.SpMaxIter
		close,2
		close,2
print,"Image Conversion Complete"
endif
ReferencePresent=ReadTillNewLine()
print,"Reference Present=",ReferencePresent
If Fix(ReferencePresent) eq 1 then begin
	InstaParams=TInstaImage()
	TCalibImageParams=TCalibImage()
	TImage, this_chunk, number_of_chunks, TImageParams, Data
endif
BackgroundPresent=ReadTillNewLine()
print,"Background Present=",BackgroundPresent
If Fix(BackgroundPresent) eq 1 then begin
	InstaParams=TInstaImage()
	TCalibImageParams=TCalibImage()
	TImage, this_chunk, number_of_chunks, TImageParams, Data
endif
LivePresent=ReadTillNewLine()
print,"Live Present=",LivePresent
If Fix(LivePresent) eq 1 then begin
	InstaParams=TInstaImage()
	TCalibImageParams=TCalibImage()
	TImage, this_chunk, number_of_chunks, TImageParams, Data
endif
SourcePresent=ReadTillNewLine()
;print,"Source Present=",SourcePresent
If Fix(SourcePresent) eq 1 then begin
	InstaParams=TInstaImage()
	TCalibImageParams=TCalibImage()
	TImage, this_chunk, number_of_chunks, TImageParams, Data
endif
close,1
wset,def_wind
return
end
;
; -------------------------------------
;
pro get_camera_head_model, camera_head_HeadModel
;[ReturnCode, HeadModel] = GetAndorSifProperty('c:\andor\image.sif', 'HeadModel', 0)
if !VERSION.OS_family eq 'unix' then ReadSif_dll_location=pref_get('IDL_MDE_START_DIR')+'/Read_SIF_DLL/GetAndorSifProperty.dll'	$
	else	ReadSif_dll_location=pref_get('IDL_WDE_START_DIR')+'\Read_SIF_DLL\GetAndorSifProperty.dll'
r=strarr(2)
r=call_external(ReadSif_dll_location, 'GetAndorSifProperty', 'Z:\PalmClusterTest\PeakSelector_V8.8\Read_SIF_DLL\cell1_lamp.sif','HeadModel', 0, /unload)
print,r
stop
end
;
; -------------------------------------
;
function GetAndorSifProperty

;[ReturnCode, HeadModel] = GetAndorSifProperty('c:\andor\image.sif', 'HeadModel', 0)
;[ReturnCode, ExposureTime] = GetAndorSifProperty('c:\andor\image.sif', 'ExposureTime', 0)
;
;  ReturnCode = 0 if an error occurs
;  ReturnCode = 1 if the function
;
;  Input parameter 1 = a string with the sif file name
;  Input parameter 2 = the property to be retrieved
;  Input parameter 3 = the data type you wish to access
;    0: signal
;    1: ref
;    2: background
;    3: source
;    4: live
;
;The full list of properties which can be retrieved are:-
;
;AClock
;Active
;BaselineClamp
;BaselineOffset
;CentreRow
;Clock
;DataType
;Delay
;DetectorFormatX
;DetectorFormatZ
;EMRealGain
;ExposureTime
;FileName
;FlipX
;FlipY
;FormattedTime
;FrameTransferAcquisitionMode
;Frequency
;Gain
;Head
;HeadModel
;IntegrationCycleTime
;IOC
;KineticCycleTime
;MCP
;Mode
;ModeFullName
;NumberImages
;NumberIntegrations
;NumberPulses
;Operation
;OutputAmplifier
;PixelReadOutTime
;PreAmplifierGain
;PreScan
;ReadPattern
;ReadPatternFullName
;RowOffset
;SIDisplacement
;SINumberSubFrames
;StoreType
;SWVersion
;SWVersionEx
;Temperature
;Time
;TrackHeight
;TriggerLevel
;TriggerSource
;TriggerSourceFullName
;Type
;UnstabalizedTemperature
;Version
;VerticalClockAmp
;VerticalShiftSpeed

;  ReturnCode = 0 if an error occurs
;  ReturnCode = 1 if the function
;
;[ReturnCode, HeadModel] = GetAndorSifProperty('c:\andor\image.sif', 'HeadModel', 0)

;a=1L;
;b=2L;
;c=lonarr(2);
;print, c
;r=call_external('add.dll', 'add', a, b, c, /unload)

returncode=0UL
HeadModel='headmodel'
result=[ReturnCode,returncode]
result0=0UL
par1='C:\IDL\PeakSelector_V8.9\Read_SIF_DLL\gp100508_02_FAPP1PH_EGFP.sif'
par2='HeadModel'
par3=0UL

par10=byte(par1)
result0=call_external('C:\IDL\PeakSelector_V8.9\Read_SIF_DLL\SIFReaderSDK\ATSIFIO.dll', 'ATSIF_ReadFromFile', par10,/AUTO_GLUE)
print,result0

print,result
return, result
end