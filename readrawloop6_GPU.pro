;CGroupParametersGP[0,*] - Peak Base Level (Offset)
;CGroupParametersGP[1,*] - Peak Amplitude
;CGroupParametersGP[2,*] - Peak X  Position
;CGroupParametersGP[3,*] - Peak Y  Position
;CGroupParametersGP[4,*] - Peak X Gaussian Width
;CGroupParametersGP[5,*] - Peak Y Gaussian Width
;CGroupParametersGP[6,*] - Number of Photons in the Peak
;CGroupParametersGP[7,*] - Chi Squared
;CGroupParametersGP[8,*] - Original FitOK
;CGroupParametersGP[9,*] - frame number
;CGroupParametersGP[10,*] - peak index in the frame
;CGroupParametersGP[11,*] - peak global index
;CGroupParametersGP[12,*] - (Peak X Gaussian Width-const)*(Peak X Gaussian Width-const)
;CGroupParametersGP[13,*] - sigma amplitude
;CGroupParametersGP[14,*] - Peak X Sigma (based on Nph only)
;CGroupParametersGP[15,*] - Peak Y Sigma (based on Nph only)
;CGroupParametersGP[16,*] - Peak X Full Sigma
;CGroupParametersGP[17,*] - Peak Y Full Sigma
;CGroupParametersGP[18,*] - group #
;CGroupParametersGP[19,*] - Group X Position
;CGroupParametersGP[20,*] - Group Y Position
;CGroupParametersGP[21,*] - Group X Full Sigma
;CGroupParametersGP[22,*] - Group Y Full Sigma
;CGroupParametersGP[23,*] - Total Number of Photons in the Group
;CGroupParametersGP[24,*] - Total Number of Peaks in the Group
;CGroupParametersGP[25,*] - Frame Index in the Group
;CGroupParametersGP[26,*] - Label Number
;CGroupParametersGP[27,*] - Label 1 Amplitude
;CGroupParametersGP[28,*] - Label 2 Amplitude
;CGroupParametersGP[29,*] - Label 3 Amplitude
;CGroupParametersGP[30,*] - Fit OK labels
;CGroupParametersGP[31,*] - Label 1 Amplitude Sigma
;CGroupParametersGP[32,*] - Label 2 Amplitude Sigma
;CGroupParametersGP[33,*] - Label 3 Amplitude Sigma
;CGroupParametersGP[34,*] - Peak Z Position
;CGroupParametersGP[35,*] - Sigma Z
;CGroupParametersGP[36,*] - Coherence
;CGroupParametersGP[37,*] - Group L1 Amplitude
;CGroupParametersGP[38,*] - Group L2 Amplitude
;CGroupParametersGP[39,*] - Group L3 Amplitude
;CGroupParametersGP[40,*] - Group Z Position
;CGroupParametersGP[41,*] - Group Sigma Z
;CGroupParametersGP[42,*] - Group Coherence
;CGroupParametersGP[43,*] - XY Ellipticity
;CGroupParametersGP[44,*] - Peak Z Position (Phase unwrapped)
;CGroupParametersGP[45,*] - Peak Z Position Error (Phase unwrapped)
;CGroupParametersGP[46,*] - Group XY Ellipticity
;CGroupParametersGP[47,*] - Group Z Position (Phase unwrapped)
;CGroupParametersGP[48,*] - Group Z Position (Phase unwrapped)
;
pro WriteInfoFile					;Template for *.txt file descibing datafile.dat
common InfoFit, pth, filen, thisfitcond, saved_pks_filename
ref_filename=pth+filen+'.txt'
openw,1,ref_filename
printf,1,pth
printf,1,filen
printf,1,thisfitcond.zerodark
printf,1,thisfitcond.xsz
printf,1,thisfitcond.ysz
printf,1,thisfitcond.Nframesmax
printf,1,thisfitcond.Frm0
printf,1,thisfitcond.FrmN
printf,1,thisfitcond.THresholdcriteria
printf,1,thisfitcond.filetype
printf,1,thisfitcond.LimBotA1
printf,1,thisfitcond.LimTopA1
printf,1,thisfitcond.LimBotSig
printf,1,thisfitcond.LimTopSig
printf,1,thisfitcond.LimChiSq
printf,1,thisfitcond.Cntpere
printf,1,thisfitcond.maxcnt1
printf,1,thisfitcond.maxcnt2
printf,1,thisfitcond.fliphor
printf,1,thisfitcond.flipvert
printf,1,thisfitcond.SigmaSym
close,1
return
end
;
;-----------------------------------------------------------------------------------
;
pro ReadInfoFile	;Read *.txt file for basic file processing info
common InfoFit, pth, filen, thisfitcond, saved_pks_filename
thisfitcond={			$
f_info,					$
zerodark:360,			$
xsz:256,				$
ysz:256,				$
Nframesmax:1500ul,		$
Frm0:0ul,				$
FrmN:1499ul,			$
Thresholdcriteria:10.0,	$
filetype:0,				$
LimBotA1:12.6,			$
LimTopA1:10000.,		$
LimBotSig:0.5,			$
LimTopSig:3.5,			$
LimChiSq:1500.,			$
Cntpere:15.1,			$
maxcnt1:125,			$
maxcnt2:0,				$
fliphor:0,				$
flipvert:0,				$
SigmaSym:0}
reffilename = Dialog_Pickfile(/read,get_path=fpath,filter=['*.sif','*.txt','*.dat'],title='Select *.sif, *.txt or *.dat data/info file')
if reffilename eq '' then return
cd,fpath
pth=fpath						;nnnn
fname=strmid(reffilename,strlen(fpath))

WIDGET_CONTROL,/HOURGLASS
if strmatch(reffilename,'*.sif',/fold_case) then begin
	conf_example='Andor_iXon_SN_1880.ini'
	ini_path=file_which(conf_example,/INCLUDE_CURRENT_DIR)
	if !VERSION.OS_family eq 'unix' then	configpath=pref_get('IDL_MDE_START_DIR')+'/Andor_iXon_ini' else configpath=pref_get('IDL_WDE_START_DIR')+'\Andor_iXon_ini'
	Read_sif,fpath,fname,configpath,thisfitcond.SigmaSym
	reffilename=strmid(reffilename,0,strlen(reffilename)-4)+'.txt'
endif

if strmatch(reffilename,'*.txt',/fold_case) then begin
	ReadThisFitCond, reffilename, pth, filen, thisfitcond
	pth1=file_which(fpath,strmid(fname,0,strlen(fname)-4)+'.dat')
	if pth1 ne '' then pth=fpath
endif
if strmatch(reffilename,'*.dat',/fold_case) then begin
	filen=strmid(reffilename,strlen(fpath))			;the filename
	filen=strmid(filen,0,strlen(filen)-4)			;the filename wo extension
endif
return
end
;
;------------------------------------------------------------------------------------
;
pro ReadThisFitCond, reffilename, pth, filen, thisfitcond		; Reads .TXT file with fit conditions
	pth=''
	filen=''
	thisfitcond={			$
	f_info,					$
	zerodark:1000,			$
	xsz:256,				$
	ysz:256,				$
	Nframesmax:1500ul,		$
	Frm0:0ul,				$
	FrmN:1499ul,			$
	Thresholdcriteria:37.8,	$
	filetype:0,				$
	LimBotA1:12.6,			$
	LimTopA1:10000.,		$
	LimBotSig:0.5,			$
	LimTopSig:3.5,			$
	LimChiSq:1500.,			$
	Cntpere:13.878,			$
	maxcnt1:125,			$
	maxcnt2:25,				$
	fliphor:0,				$
	flipvert:0,				$
	SigmaSym:0}

	openr,1,reffilename
	readf,1,pth
	readf,1,filen
	readf,1,zerodark	&	thisfitcond.zerodark=	zerodark
	readf,1,xsz			&	thisfitcond.xsz		=	xsz
	readf,1,ysz			&	thisfitcond.ysz		=	ysz
	readf,1,Nframesmax	&	thisfitcond.Nframesmax=	Nframesmax
	readf,1,Frm0		&	thisfitcond.Frm0	=	Frm0
	readf,1,FrmN		&	thisfitcond.FrmN	=	FrmN
	readf,1,Thresholdcriteria	&	thisfitcond.Thresholdcriteria	=	Thresholdcriteria
	readf,1,filetype	&	thisfitcond.filetype	=	filetype
	readf,1,LimBotA1	&	thisfitcond.LimBotA1	=	LimBotA1
	readf,1,LimTopA1	&	thisfitcond.LimTopA1	=	LimTopA1
	readf,1,LimBotSig	&	thisfitcond.LimBotSig	=	LimBotSig
	readf,1,LimTopSig	&	thisfitcond.LimTopSig	=	LimTopSig
	readf,1,LimChiSq	&	thisfitcond.LimChiSq	=	LimChiSq
	readf,1,Cntpere		&	thisfitcond.Cntpere		=	Cntpere
	readf,1,maxcnt1		&	thisfitcond.maxcnt1		=	maxcnt1
	readf,1,maxcnt2		&	thisfitcond.maxcnt2		=	maxcnt2
	if (~ EOF(1)) then readf,1,fliphor else fliphor = 0		& thisfitcond.fliphor = fliphor
	if (~ EOF(1)) then readf,1,flipvert else flipvert = 0	& thisfitcond.flipvert = flipvert
	if (~ EOF(1)) then readf,1,SigmaSym else SigmaSym = 0	& thisfitcond.SigmaSym = SigmaSym
	close,1
return
end
;
;------------------------------------------------------------------------------------
;
function ReadData,thefile_no_exten,thisfitcond,framefirst,Nframes	;Reads thefile & returns data (units of photons)
framelast=framefirst+Nframes-1
thefile = thefile_no_exten+'.dat'
zerodark=thisfitcond.zerodark										;zero dark count in CCD counts
xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz													;number of y pixels
filetype=thisfitcond.filetype										;flag for special file read treatment
counts_per_e=thisfitcond.cntpere									;counts per electron CCD sensitivity
data=-1
if (filetype eq 0) then begin
	data=uintarr(xsz,ysz,Nframes)
	openr,1,thefile
	point_lun,1,2ull*xsz*ysz*framefirst
	readu,1,data
	close,1
	data=((float(temporary(data))-zerodark)/counts_per_e)>0.
	;print,'done reading and converting'
endif
return,data
end

;------------------------------------------------------------------------------------
pro Showframes, data,xsz,ysz,mg, Nframes,scl		;Time movie of data
for i =0, 1*Nframes do begin
	tv,rebin(data[*,*,i mod Nframes],mg*xsz,mg*ysz,/sample)*scl < 255
	xyouts,40,40,string(i mod Nframes),/device,charthick=((i mod 4) eq 1)+1
	wait,0.05
endfor
tv,rebin(data[*,*,Nframes/2],mg*ysz,mg*xsz,/sample)*scl < 255
print,'Done with Real space display'
wait,1.0
return
end

;------------------------------------------------------------------------------------
pro ShowIt, image, mag=mg,wait=wtime		;Display image w scalebar
sz=size(image)
tv,bytscl(image[*,0:(sz[2]-1) < 1190])
TVscales,sz[1],sz[2],mg
wait,wtime
return
end
;
;------------------------------------------------------------------------------------
;
pro FindPeaks, clip, totdat, d, threshold, mxcnt, peakxa, peakya, maxpeakcriteria, criteria		;Create and ordered list of peak candidate coordinates
clipsz=size(clip)
Nx=clipsz[1]
Ny=clipsz[2]
mask=fltarr(Nx,Ny)
mask[d+2:Nx-d-1,d+2:Ny-d-1]=1.0
criteriaclip=(clip-0.9*smooth(totdat,3))>0.5
dk=5	&	dl=2
xyvals=findgen(dk)-dl
sigmasq=1.3
gcx=exp(-(xyvals^2)/sigmasq^2/2.)
gausscenter=gcx#gcx
gausscenter=gausscenter-mean(gausscenter)
criteria=(convol(criteriaclip,gausscenter) > 0)*mask
newcriteria=criteria
counter = 0
maxpeakcriter=max(criteria)
if maxpeakcriter gt threshold then begin
while (maxpeakcriter gt threshold) and (counter lt mxcnt) do begin
	maxpeakcriter=max(newcriteria)
	peakloc=where(newcriteria eq maxpeakcriter,count)
	if (count gt 0) and (maxpeakcriter gt threshold) then begin
	 	peakloc2d=ARRAY_INDICES(newcriteria,peakloc)
		peakx=peakloc2d[0]
		peaky=peakloc2d[1]
		if peakx-d lt 0 then stop
		newcriteria[peakx-d:peakx+d,peaky-d:peaky+d] = 0
		counter+=1
		if counter eq 1 then peakxa = [peakx] else peakxa=[[peakxa],peakx]
		if counter eq 1 then peakya = [peaky] else peakya=[[peakya],peaky]
		if counter eq 1 then maxpeakcriteria = [maxpeakcriter] else maxpeakcriteria=[[maxpeakcriteria],maxpeakcriter]
	endif
endwhile
endif
mxcnt=counter
return
end
;
;------------------------------------------------------------------------------------
;
pro FindPeaksLong, clip, totdat, d, threshold, mxcnt, peakxa, peakya, maxpeakcriteria, criteria, Peaks_Raw		;Create and ordered list of peak candidate coordinates
clipsz=size(clip)
Peaks_Raw = uintarr(2*d+1,2*d+1, mxcnt)
Nx=clipsz[1]
Ny=clipsz[2]
mask=fltarr(Nx,Ny)
mask[d+2:Nx-d-1,d+2:Ny-d-1]=1.0
criteriaclip=(clip-0.9*smooth(totdat,3))>0.5
dk=5	&	dl=2
xyvals=findgen(dk)-dl
sigmasq=1.3
gcx=exp(-(xyvals^2)/sigmasq^2/2.)
gausscenter=gcx#gcx
gausscenter=gausscenter-mean(gausscenter)
criteria=(convol(criteriaclip,gausscenter) > 0)*mask
newcriteria=criteria
counter = 0
maxpeakcriter=max(criteria)
if maxpeakcriter gt threshold then begin
while (maxpeakcriter gt threshold) and (counter lt mxcnt) do begin
	maxpeakcriter=max(newcriteria)
	peakloc=where(newcriteria eq maxpeakcriter,count)
	if (count gt 0) and (maxpeakcriter gt threshold) then begin
	 	peakloc2d=ARRAY_INDICES(newcriteria,peakloc)
		peakx=peakloc2d[0]
		peaky=peakloc2d[1]
		Peaks_Raw[*,*,counter]=clip[peakx-d:peakx+d,peaky-d:peaky+d]
		if peakx-d lt 0 then stop
		newcriteria[peakx-d:peakx+d,peaky-d:peaky+d] = 0
		counter+=1
		if counter eq 1 then peakxa = [peakx] else peakxa=[[peakxa],peakx]
		if counter eq 1 then peakya = [peaky] else peakya=[[peakya],peaky]
		if counter eq 1 then maxpeakcriteria = [maxpeakcriter] else maxpeakcriteria=[[maxpeakcriteria],maxpeakcriter]
	endif
endwhile
endif
mxcnt=counter
Peaks_Raw = Peaks_Raw[*,*,0:counter-1]
return
end
;
;------------------------------------------------------------------------------------
;
Pro FindnWackaPeak, clip, d, peakparams, fita, result, thisfitcond, DisplaySet, peakx, peaky, criteria, Dispxy, SigmaSym		;Find,fit,remove target peak & return fit parameters
;peakparams = {twinkle,frameindex:0l,peakindex:0l,fitOK:0,peakx:0.0,peaky:0.0,A:fltarr(7),sigma1:fltarr(7),sigma2:fltarr(7),chisq:0.0,Nphot:0l}
		;DisplaySet=0				;set to -1 - no display while fitting(fastest)   0 - min display   1 - some display	2 - full display
mod4clip = fix(clip *0)				;create array for changed pixels modified by >= 4 photons
tempclip=clip
clipsize=size(clip)
peakx=(peakx<(clipsize[1]-d-1))>d
peaky=(peaky<(clipsize[2]-d-1))>d
region=clip[peakx-d:peakx+d,peaky-d:peaky+d]							;define 2d+1 region around peak
dd=2*d+1
A=peakparams.A
A[0]=(region[0,0]+region[0,2*d]+region[2*d,0]+region[2*d,2*d])/4.		;set A[0] to averaged value of base
A[1]=max(region[d-1:d+1,d-1:d+1]-A[0]) > 1								;set A[1] to peak amplitude
A1=A[1]
peakparams.A=A															;write intial A into peakparms.A
A=A[0:5]
if SigmaSym eq 1 then result=gauss2Dfithh(region,A,sigma=sigma, chisq=chisq, FITA = fita, STATUS = status, ITMAX = 100)	;do 2D fit
if SigmaSym eq 0 then begin				;SigmaSym eq 0 is the flag for Radially symmetric gaussian fit
	AR=A[0:4]
	AR[3:4]=A[4:5]						;Collapse sigma x and sigma y into one variable sigma r
	fita=[fita[0:2],fita[4:5]]
	;print,'AR=',Ar, 'peakx=',peakx,'peaky=',peaky
	result=gauss2RDfithh(region,AR,sigma=sigmaR, chisq=chisq, FITA = fita, STATUS = status, ITMAX = 100)	;do 2D fit

	A[0:2]=AR[0:2]
	A[3:5]=AR[2:4]
	sigma=fltarr(6)
	sigma[0:2]=sigmar[0:2]
	sigma[3:5]=sigmar[2:4]
endif
A[1]=A[1] > 0.0				;75*thisfitcond.LimBotA1
A[2:3]=abs(A[2:3])
fitOK = (A[2] gt thisfitcond.LimBotSig) and (A[2] lt thisfitcond.LimTopSig) and $
		(A[3] gt thisfitcond.LimBotSig) and (A[3] lt thisfitcond.LimTopSig) and $
		(chisq lt thisfitcond.LimChiSq) and (A[1] gt thisfitcond.LimBotA1)
fitreallybad = (chisq gt 2*thisfitcond.LimChiSq) or (A[1] lt thisfitcond.LimBotA1/2) or (A[1] gt thisfitcond.LimTopA1*2) or $
		(A[2] lt thisfitcond.LimBotSig/2) or (A[2] gt thisfitcond.LimTopSig*2) or $
		(A[3] lt thisfitcond.LimBotSig/2) or (A[3] gt thisfitcond.LimTopSig*2) or $
		(A[4] lt (d-2)) or (A[4] gt (d+2)) or (A[5] lt (d-2)) or (A[5] gt (d+2))
fitOKq=(A[2] le thisfitcond.LimBotSig) + 2*(A[2] ge thisfitcond.LimTopSig)+4*(A[3] le thisfitcond.LimBotSig)+8*(A[3] ge thisfitcond.LimTopSig)+$
	16*(chisq ge thisfitcond.LimChiSq)+32*(A[1] le thisfitcond.LimBotA1)+64*(chisq gt 2*thisfitcond.LimChiSq)+128*(A[1] lt thisfitcond.LimBotA1/2)+$
	256*(A[1] gt thisfitcond.LimTopA1*2) + 512*(A[2] lt thisfitcond.LimBotSig/2)+1024*(A[2] gt thisfitcond.LimTopSig*2)+$
	2048*(A[3] lt thisfitcond.LimBotSig/2)+4096*(A[3] gt thisfitcond.LimTopSig*2)+$
	8192*(A[4] lt (d-2)) or (A[4] gt (d+2))or (A[5] lt (d-2)) or (A[5] gt (d+2))
;print,'status - fitok - fitreallybad - A =  ',status,fitok,fitreallybad,A
if (status eq 0) and (fitreallybad eq 0) then begin						;if status eq 0 i.e. a good fit then subtract out the fitted gaussian
	result=result-A[0]*replicate(1,2*d+1,2*d+1)
	;clip[PeakX-d:PeakX+d,PeakY-d:PeakY+d]=region-result>0
	dxx=round(((((A[2]*4)>5.0)<15.0)<PeakX)<(clipsize[1]-PeakX-1))			&	dyy=round(((((A[3]*4)>5.0)<15.0)<PeakY)<(clipsize[2]-PeakY-1))							;if the peak is very asymetric - subtract a properly wider area
	xarr=exp(-0.5*((indgen(2*dxx+1)-dxx+d-A[4])/A[2])^2)	&	yarr=exp(-0.5*((indgen(2*dyy+1)-dyy+d-A[5])/A[3])^2)
	full_result=A[1]*xarr#yarr
	clip[PeakX-dxx:PeakX+dxx,PeakY-dyy:PeakY+dyy]=temporary(clip[PeakX-dxx:PeakX+dxx,PeakY-dyy:PeakY+dyy])-full_result
	if fitOK eq 0 then fitOK = 3										;if fitOK eq 0 but not really bad then report as suspicious ie fitOK = 3
endif else begin
	if total(fita) ge 5 then begin						;if fit not OK and gt 4 fit variables then do 2nd fit with sigma constrained gaussian and subtract it out
		A=peakparams.A[0:5]											;reset A parameters
		fita = [1,1,0,0,1,1]
		constrainresult=gauss2Dfithh(region,A,sigma=sigma, chisq=chisq, FITA = fita, STATUS = status, ITMAX = 100)
		if status eq 2 then constrainresult=gauss2Dfithh(region,A,sigma=sigma, chisq=chisq, FITA = fita, STATUS = status, ITMAX=200)
		fitOK = (A[1] le thisfitcond.LimTopA1) and (A[1] ge thisfitcond.LimBotA1) and $
				((status eq 0) or (status eq 2)) and (chisq lt thisfitcond.LimChiSq/2)
		A[1] = A[1] > 0.75*thisfitcond.LimBotA1
		if fitok eq 1 then 	begin							;second iteration with sigmas fix has worked, --- so subtract out fit, mark as fitok = 2
			result=constrainresult-A[0]*replicate(1,2*d+1,2*d+1)
			clip[PeakX-d:PeakX+d,PeakY-d:PeakY+d]=region-result>0
			fitok = 2
		endif
	endif else 	A[0:1]=peakparams.A[0:1]				;reset A parameters if only 2 fit parameters and no refitting done
	if fitok eq 0 then begin							;fit fails, ---so subtract out fake gaussian from data, mark as fitok = 0, and move on
		peakparams.A[1]=A1								;use initial parameters
		position=shift(dist(dd),d,d)
		fakegauss=peakparams.A[1]*exp(-(position/peakparams.A[2])^2/2.)
		clip[PeakX-d:PeakX+d,PeakY-d:PeakY+d]=region-fakegauss>0
	endif
endelse
fitOK = fitOK + 4*(total(mod4clip[PeakX-d:PeakX+d,PeakY-d:PeakY+d]) gt 1)		;add 4 to fitOK if any pixel was previously modified
		;fitOK = 0 status = 1 (diverge) or 2 (no converge) or fitreallybad = 1 (true) and refit failed if more than 4 variables fitted
		;fitOK = 1 status = 0 (converge) and fitreallybad = 0 (false) and (initial)fitOK = 1 (sigma & amplitude within bounds) ---report good fit values
		;fitOK = 2 status = 0 (converge) and fitreallybad = 0 (false) and (initial)fitOK = 0 (sigma or amplitude out of bounds) ---second fit iteration worked
		;fitOK = 3 status = 0 (converge) and (fitOK = 0) and fitreallybad = 0 (false) ---- report fit values but indicate as suspicious
		;so a peak with fitOK = 1 is best,		 = 2 is OK,		 = 3 is suspect,		 = 0 is bad
		;fitOK ge 4 then like above (fitOK mod 4) but fit used previously modified (>=4) pixel
peakparams.fitOK=fitOK										;assign data to structure
peakparams.peakx=peakx+A[4]-d+0.5
peakparams.peaky=peaky+A[5]-d+0.5
peakparams.A=A[0:5]
sigma=(sigma[0:5] > 0.001) < 0.2
peakparams.sigma1=sigma
peakparams.chisq=chisq
peakparams.Nphot=total(result) > 0
Nback=A[0]*!pi*A[2]*A[3]
Nphot=peakparams.Nphot > 1
Ntot=Nback+Nphot
;newsigx=Nback/Ntot*A[2]/sqrt(Nback+1) + Nphot/Ntot*(A[2]/sqrt(Nphot) + sigma(4))		;old version
;newsigy=Nback/Ntot*A[3]/sqrt(Nback+1) + Nphot/Ntot*(A[3]/sqrt(Nphot) + sigma(5))
;print,'T1= ',Nback/Ntot*A[2]/sqrt(Nback+1),' T2= ',Nphot/Ntot*(A[2]/sqrt(Nphot)), ' T3= ',Nphot/Ntot*sigma(4),$
;		'   Ntot, Nback, Nphot, A2, Sigma4  ',Ntot,Nback,Nphot,A[2],sigma[4]
Region_edge=fltarr(8*d)
Region_edge[0:2*d]=region[0,*]  &  Region_edge[2*d+1:4*d-1]=region[1:2*d-1,0]
Region_edge[4*d:6*d]=region[2*d,*]  &  Region_edge[6*d+1:8*d-1]=region[1:2*d-1,2*d]
back_noise=stddev(region_edge)

newsigx=1.0*sqrt((A[2]^2+1./12.)/Nphot + 8.*!pi*A[2]^4*back_noise^2/Nphot^2 + sigma[4]^2)   ;from Webb's paper 2D with with 1.00 factor
newsigy=1.0*sqrt((A[3]^2+1./12.)/Nphot + 8.*!pi*A[3]^4*back_noise^2/Nphot^2 + sigma[5]^2)
Err=sqrt(Nphot)
update=[sqrt(peakparams.A[1]), newsigx, newsigy, peakparams.A[2]/Err, peakparams.A[3]/Err, 1./Err]
peakparams.sigma2[1:6]=update

mod4clip[PeakX-d:PeakX+d,PeakY-d:PeakY+d]=abs(clip[PeakX-d:PeakX+d,PeakY-d:PeakY+d]-region) ge 4		;report if changed pixels >= 4 photons
;----------------------									;display find and wack prpgress
if DisplaySet eq 2 then begin			;1 then begin
	scl=1.0
	tempclip[PeakX-d:PeakX+d,PeakY-d]=255								;draw underline under peak
	tv,scl*tempclip<255,0,0											;show it
	waittime=0.25
	if n_elements(criteria) gt 1 then tvscl,criteria,512,0
	scl=3.0
	ofs=(min(tempclip) - 50) > 0
	tempclip=tempclip-ofs
	tempclip[PeakX-d:PeakX+d,PeakY-d]=255								;draw underline under peak
	tv,scl*tempclip<255,0,0											;show it
	wait,waittime
	tempclip=clip-ofs
	tempclip[PeakX-d:PeakX+d,PeakY-d]=255								;draw underline under removed peak
	tv,scl*tempclip<255,0,0											;show it
	if (peakparams.peakindex mod 10) eq 1 then print,$
			'frame#, peak#, fitOKq,   A0,        A1,       deltax,     deltay,     posx,      posy,    chi(sq),    sigx,     sigy'
	print,peakparams.frameindex,peakparams.peakindex, fitOKq, A[0:5],sqrt(chisq),newsigx,newsigy,format='(3i8,7f10.3,2f9.4)'
	gscl=4.
								;Show peaks one by one
	xtvpeak=(dd*gscl*dispxy[0] mod (fix(1024.0/dd/gscl)*dd*gscl))
	ytvpeak=512+(3*dispxy[1])*dd*gscl
	tv,50+scl*rebin(region-min(region),dd*gscl,dd*gscl,/sample)<255,xtvpeak,ytvpeak				;tv slected peak region
	tv,50+scl*rebin(result-min(result),dd*gscl,dd*gscl,/sample)<255,xtvpeak,ytvpeak+dd*gscl		;tv resulting fit
	plots,gscl*(A[4]+0.5)+xtvpeak,gscl*(A[5]+0.5)+ytvpeak,psym=1,/device,col=0	;mark the center of data peak with plus
	plots,gscl*(A[4]+0.5)+xtvpeak,gscl*(A[5]+0.5)+ytvpeak,psym=3,/device		;mark center of data peak, put dot in middle
	plots,gscl*(A[4]+0.5)+xtvpeak,gscl*(A[5]+0.5)+ytvpeak+dd*gscl,psym=1,/device,col=0	;mark center of peak fit
	plots,gscl*(A[4]+0.5)+xtvpeak,gscl*(A[5]+0.5)+ytvpeak+dd*gscl,psym=3,/device		;mark center of peak fit
	xpos=(findgen(dd*gscl)/gscl-peakparams.A[4]-0.5)#replicate(1,dd*gscl)
	ypos=replicate(1.,dd*gscl)#(findgen(dd*gscl)/gscl-peakparams.A[5]-0.5)
	gausscenter=peakparams.A[1]*exp(-((xpos/newsigx)^2+(ypos/newsigy)^2)/2.)
	tvscl,gausscenter<255,xtvpeak,0
	minclp=min(clip[PeakX-d:PeakX+d,PeakY-d:PeakY+d])
	tv,50+scl*rebin(clip[PeakX-d:PeakX+d,PeakY-d:PeakY+d]-minclp,dd*gscl,dd*gscl,/sample)<255,xtvpeak,ytvpeak+2*dd*gscl		;tv slected residual after peak removal
	xyouts,xtvpeak+dd*gscl/2,ytvpeak,string(fitOKq),/device,align=0.75
endif
return
end
;------------------------------------------------------------------------------------
pro Showpeakparams, Apeakparams,Frameindx		;Create 3x3 matrix of plots of peak fitting params of a frame
thisframe=where(Apeakparams.frameindex eq Frameindx,Npeaks)
if thisframe[0] eq -1 then print, 'No Frames of that index'
!P.multi=[0,3,3]
title=['Base','Ph/pix amp','sigx','sigy','dx','dy','pix error','Nphot','chisq']
rmx=[100.,200.,2.,2.,10.,10.,0.2,2000.,200.]
for i=0,5 do begin
	plot,findgen(Npeaks),[Apeakparams[thisframe].A[i] > 0] < 200,title=title[i],yrange=[0.,rmx[i]],psym=3
	oploterr,findgen(Npeaks),Apeakparams[thisframe].A[i], Apeakparams[thisframe].sigma1[i]
	notOKindex=where(Apeakparams[thisframe].fitOK eq 0)
	if notOKindex[0] ge 1 then oplot,notOKindex,([Apeakparams[thisframe[notOKindex]].A[i]] > 0) < 200,col=180,psym=1
endfor
sigma_x_pos=sqrt(1./Apeakparams.Nphot+Apeakparams[thisframe].sigma1[4]^2)
sigma_y_pos=sqrt(1./Apeakparams.Nphot+Apeakparams[thisframe].sigma1[5]^2)
sigma_x_pos=Apeakparams[thisframe].sigma2[2]
sigma_y_pos=Apeakparams[thisframe].sigma2[3]
plot,findgen(Npeaks),[sigma_x_pos > 0] < 100,title=title[i],yrange=[0.,rmx[i]],psym=3
if notOKindex[0] ge 1 then oplot,notOKindex,(sigma_x_pos > 0) < 100,col=180,psym=2
oplot,findgen(Npeaks),[sigma_y_pos > 0] < 100,psym=3
if notOKindex[0] ge 1 then oplot,notOKindex,([Apeakparams[notOKindex].sigma2[5]] > 0) < 100,col=180,psym=2
plot,findgen(Npeaks),[Apeakparams[thisframe].Nphot > 0] < 100,title=title[i+1],yrange=[0.,rmx[i+1]],psym=3
oploterr,findgen(Npeaks),[Apeakparams[thisframe].Nphot],Sqrt([Apeakparams[thisframe].Nphot])
if notOKindex[0] ge 1 then oplot,notOKindex,([Apeakparams[notOKindex].Nphot] > 0) < rmx[i+1],col=180,psym=2
plot,findgen(Npeaks),[Apeakparams[thisframe].chisq > 0] < rmx[i+2],title=title[i+2],yrange=[0.,rmx[i+2]],psym=3
if notOKindex[0] ge 1 then oplot,notOKindex,([Apeakparams[notOKindex].chisq] > 0) < rmx[i+2],col=180,psym=2
!p.multi=[0,0,0]
return
end
;
;------------------------------------------------------------------------------------
;
Function ParamsofShortStackofFrames,data,DisplayType,thisfitcond,framefirst,SigmaSym
xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz													;number of y pixels
Thresholdcriteria = thisfitcond.Thresholdcriteria					;threshold for smallest extracted and fitted peak
szdata=size(data)
if szdata[0] eq 2 then Nframes=1 else Nframes=szdata[3]
if Nframes gt 1 then totdat=total(data[*,*,0:Nframes-1],3)/Nframes else totdat=data
d=5.															;half size of region of a selected peak
																;setup twinkle structure with all the fitted peak information
peakparams = {twinkle,frameindex:0l,peakindex:0l,fitOK:0,peakx:0.0,peaky:0.0,A:fltarr(7),sigma1:fltarr(7),sigma2:fltarr(7),chisq:0.0,Nphot:0l}
peakparams.A=[0.0,1.0,1.2,1.2,d,d,0.]
Peakparamsinitial=peakparams
Apeakparams=peakparams
trash=''
if (DisplayType eq 0) then begin
	trash=string(framefirst)+'#'
	xyouts,0.85,0.92,trash,/normal		;0.85,0.92
	xyouts,0.05,0.02,trash,/normal		;0.85,0.92
endif
for frameindx=0,Nframes-1 do begin
	if (DisplayType eq -1) and (0 eq (frameindx mod 50)) then print,'Frameindex=',frameindx
	if Nframes gt 1 then clip=reform(data[*,*,frameindx]) else clip=data
	if (DisplayType ge 1) and ((frameindx mod 200) eq 0) then DisplaySet=2 else DisplaySet=DisplayType
	if DisplaySet gt 1 then begin
		mg=2*(1 + (xsz lt 250))
		newframe=rebin(clip,xsz*mg,ysz*mg,/sample)
		ShowIt,newframe,mag=mg,wait=0.0
	endif
	If (DisplaySet gt 1) or (DisplaySet ge 1)*((frameindx mod 20) eq 1) then begin
		xyouts,0.85,0.92,trash,/normal,col=0
		xyouts,0.05,0.02,trash,/normal,col=0
		trash=string(frameindx+framefirst)+'#'
		xyouts,0.85,0.92,trash,/normal		;0.85,0.92
		scl=4.															;intensity scaling Range = scl* # electrons
		tv,scl*clip<255
		xyouts,0.05,0.02,trash,/normal		;0.85,0.92
	endif
	mxcnt=thisfitcond.maxcnt1
	FindPeaks, clip, totdat, d, Thresholdcriteria, mxcnt, peakxa, peakya, maxpeakcriteria, criteria
	NP=n_elements(peakxa)
	if NP ge 1 then begin
		A1peakparams=replicate(peakparams,NP)
		for k=0, NP-1  do begin								;Fit the first NP peaks
			peakparams=peakparamsinitial
			peakparams.frameindex=frameindx+framefirst
			peakparams.peakindex=k
			peakx=peakxa[k]
			peaky=peakya[k]
			fita = [1,1,1,1,1,1]
			Dispxy=[k,0]					;index of chosen frame
			FindnWackaPeak, clip, d, peakparams, fita, result, thisfitcond, DisplaySet, peakx, peaky, criteria, Dispxy, SigmaSym		;Find, fit, and remove the peak with biggest criteria
			A1peakparams[k]=peakparams
		endfor
		if frameindx eq 0 then Apeakparams=A1peakparams else Apeakparams=[Apeakparams,A1peakparams]
		mxcnt=thisfitcond.maxcnt2
		FindPeaks, clip, totdat, d, thresholdcriteria, mxcnt, peakxa, peakya, maxpeakcriteria, criteria
		NP2=n_elements(peakxa) < thisfitcond.maxcnt2
		if NP2 ge 1 then begin								;If there are more peak then fit them too.
			A2peakparams=replicate(peakparams,NP2)
			for k=0, NP2-1  do begin
				peakparams=peakparamsinitial
				peakparams.frameindex=frameindx+framefirst
				peakparams.peakindex=k+NP
				peakx=peakxa[k]
				peaky=peakya[k]
				fita = [1,1,1,1,1,1]
				Dispxy=[k,0]					;index of chosen frame
				FindnWackaPeak, clip, d, peakparams, fita, result, thisfitcond, DisplaySet, peakx, peaky, criteria, Dispxy, SigmaSym		;Find, fit, and remove the peak with biggest criteria
				A2peakparams[k]=peakparams
			endfor
			Apeakparams=[Apeakparams,A2peakparams]
		endif

		if (DisplaySet gt 1) or ((DisplaySet eq 1) and ((frameindx mod 200) eq 0)) then begin
			Showpeakparams, Apeakparams,frameindx+framefirst
			wait,2*(DisplaySet eq 3)
			tv,bytarr(1024,512)
		endif
	endif
endfor
if (DisplayType ne -1) then begin
	xyouts,0.85,0.92,trash,/normal,col=0
	xyouts,0.05,0.02,trash,/normal,col=0
endif
return,Apeakparams
end
;
;------------------------------------------------------------------------------------
;
Function ParamsofLongStackofFrames,data,DisplayType,thisfitcond,framefirst,SigmaSym;		GPU version
xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz													;number of y pixels
Thresholdcriteria = thisfitcond.Thresholdcriteria					;threshold for smallest extracted and fitted peak
szdata=size(data)
if szdata[0] eq 2 then Nframes=1 else Nframes=szdata[3]
if Nframes gt 1 then totdat=total(data[*,*,0:Nframes-1],3)/Nframes else totdat=data
d=5.															;half size of region of a selected peak
																;setup twinkle structure with all the fitted peak information
peakparams = {twinkle,frameindex:0l,peakindex:0l,fitOK:0,peakx:0.0,peaky:0.0,A:fltarr(7),sigma1:fltarr(7),sigma2:fltarr(7),chisq:0.0,Nphot:0l}
peakparams.A=[0.0,1.0,1.2,1.2,d,d,0.]
Peakparamsinitial=peakparams
Apeakparams=peakparams
trash=''

NP_tot=0
mxcnt=thisfitcond.maxcnt1
Peak_Data = uintarr(2*d+1,2*d+1, mxcnt*Nframes)
PeakX = uintarr(mxcnt*Nframes)
PeakY = uintarr(mxcnt*Nframes)

t0 = SYSTIME(/SECONDS)
print,'started convolutions'
for frameindx=0,Nframes-1 do begin
	if (frameindx mod 100) eq 0 then print,'Frame',frameindx
	if Nframes gt 1 then clip=reform(data[*,*,frameindx]) else clip=data
	FindPeaksLong, clip, totdat, d, Thresholdcriteria, mxcnt, peakxa, peakya, maxpeakcriteria, criteria, Peaks_Raw
	NP=n_elements(peakxa)
	;stop
	Peak_Data [*,*,NP_tot:NP_tot+NP-1] = Peaks_Raw
	PeakX[NP_tot:NP_tot+NP-1] = peakxa
	PeakY[NP_tot:NP_tot+NP-1] = peakya
	NP_tot+=NP
endfor
t1 = SYSTIME(/SECONDS)
print,'finished convolutions, elapsed time (sec) =',t1-t0

if NP_tot ge 1 then begin
	constraints = float([0,500,100,1500,0,5,0,5,0,10,0,10])
	imax = 30
	p = cu_gauss2d(Peak_Data, imax, constraints, 0)
	A1peakparams=replicate(peakparams,NP_tot)
	A1peakparams[0:1,*]=p[0:1,*]
	A1peakparams[2,*]=PeakX[*]+p[4,*]-d
	A1peakparams[3,*]=PeakY[*]+p[5,*]-d
	A1peakparams[4:5,*]=p[2:3,*]
endif
t2 = SYSTIME(/SECONDS)
print,'finished CUDA swarm fit, elapsed time (sec) =',t2-t1

return,Apeakparams
end
;
;------------------------------------------------------------------------------------
;
Pro ReadRawLoopCluster			;Master program to read data and loop through processing for cluster
common InfoFit, pth, filen, thisfitcond, saved_pks_filename
restore,'temp/temp.sav'
print,'sh '+idl_pwd+'/runme.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd		;Spawn workers in cluster
spawn,'sh '+idl_pwd+'/runme.sh '+strtrim(nloops,2)+' '+curr_pwd+' '+idl_pwd		;Spawn workers in cluster
thefile_no_exten=pth+filen
for nlps=0,nloops-1 do begin			;reassemble little pks files from all the workers into on big one
	framefirst=	thisfitcond.Frm0 + (nlps)*increment						;first frame in batch
	framelast=((thisfitcond.Frm0 + (nlps+1)*increment-1)<thisfitcond.Nframesmax) < thisfitcond.FrmN
	test1=file_info(pth+'/temp/'+filen+'_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'_IDL.pks')
	if ~test1.exists then stop
	restore,filename=pth+'/temp/'+filen+'_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'_IDL.pks'
	if (size(Apeakparams))[2] ne 0 then begin
		if (size(Apeakparams_tot))[2] eq 0 then begin
			Apeakparams_tot=Apeakparams
			totaldata_tot=totaldata
			image_tot=image
		endif else begin
			Apeakparams_tot=[Apeakparams_tot, Apeakparams]
			totaldata_tot=(totaldata_tot*nlps + totaldata)/(nlps+1)
			image_tot=(image_tot*nlps+image)/(nlps+1)
		endelse
	endif
	file_delete,pth+'/temp/'+filen+'_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'_IDL.pks'
endfor
Apeakparams=Apeakparams_tot
totaldata=totaldata_tot
image=image_tot
saved_pks_filename=pth+filen+'_'+strtrim(thisfitcond.Frm0,2)+'-'+strtrim(framelast,2)+'_IDL.pks'
save,Apeakparams,image,xsz,ysz,totaldata,thefile_no_exten, filename=saved_pks_filename
print,'Wrote file '+saved_pks_filename
return
end
;
;------------------------------------------------------------------------------------
;
Pro	ReadRawWorker,nlps,data_dir						;spawn mulitple copies of this programs for cluster
Nlps=ulong((COMMAND_LINE_ARGS())[0])
data_dir=(COMMAND_LINE_ARGS())[1]
cd,data_dir
restore,'temp/temp.sav'
thefile_no_exten=pth+filen
DisplayType=-1													;set to no displays
xsz=thisfitcond.xsz												;number of x pixels
ysz=thisfitcond.ysz												;number of y pixels
framefirst=	thisfitcond.Frm0 + nlps*increment					;first frame in batch
framelast=((thisfitcond.Frm0 + (nlps+1)*increment-1)<thisfitcond.Nframesmax) < thisfitcond.FrmN
Nframes=(framelast-framefirst+1) 								;number of frames to extract in file
print,'Started processinmg frames  '+strtrim(framefirst,2)+'-'+strtrim(framelast,2)
data=float(ReadData(thefile_no_exten,thisfitcond,framefirst,Nframes))	;Reads thefile and returns data (bunch of frames) in (units of photons)

print,'Read the data'
wxsz=1024 & wysz=1024
dsz=xsz>ysz
mg=((wxsz<wysz))/dsz

if Nframes gt 1 then totdat=total(data[*,*,0:Nframes-1],3)/Nframes else totdat=data
totaldata=rebin(totdat*Nframes,xsz*mg,ysz*mg,/sample)
Apeakparams = ParamsofShortStackofFrames(data,DisplayType,thisfitcond,framefirst,SigmaSym)
mg=2
loc=fltarr(xsz*mg,ysz*mg)
filter=((Apeakparams.fitok eq 1) or (Apeakparams.fitok eq 2))
loc[[mg*Apeakparams.peakx],[mg*Apeakparams.peaky]]=255*filter
image=float(loc)
save,Apeakparams,image,xsz,ysz,totaldata,filename=pth+'/temp/'+filen+'_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'_IDL.pks',thefile_no_exten
spawn,'sync'
spawn,'sync'
print,'Wrote file '+pth+'/temp/'+filen+'_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'_IDL.pks'
return
end
;
;------------------------------------------------------------------------------------
;
pro ReadRawLoop6, DisplayType, SigmaSym				;Master program to read data and loop through processing
;DisplayType set to 0 - min display while fitting 1 - some display, 2 - full display,  3 - Cluster, 4 - GPU
common InfoFit, pth, filen, thisfitcond, saved_pks_filename
										;use SigmaSym as a flag to indicate xsigma and ysigma are not independent and locked together in the fit
common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, RawFilenames, MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate
COMMON managed,	ids, $		; IDs of widgets being managed
  			names, $	; and their names
			modalList	; list of active modal widgets
TopID=ids[min(where(names eq 'WID_BASE_0_PeakSelector'))]
Event1={ WIDGET, ID:TopID, TOP:TopID, HANDLER:TopID }

if (!VERSION.OS_family eq 'unix') and (DisplayType eq 4) then begin
	if (scope_varfetch('haveCUDA', LEVEL=1)) then begin
	print,'start DLM load'
    	dlm_load, 'CU_GAUSS2D'
    endif else DisplayType=2	;(If selected GPU but no GPU is present - go for full display)
endif

if !VERSION.OS_family eq 'windows' and (DisplayType eq 4) then DisplayType=2

thefile_no_exten=pth+filen
DisplaySet=0
xsz=thisfitcond.xsz													;number of x pixels
ysz=thisfitcond.ysz													;number of y pixels

xy_sz=sqrt(float(xsz)*float(ysz))
min_frames_per_node=fix(max((thisfitcond.FrmN-thisfitcond.Frm0)/500.00))>50
increment = long(fix(500*(256.0/xy_sz)))>long(min_frames_per_node)				;setup loopback conditions to write multiple files
if DisplayType eq 4 then increment=5000
if thisfitcond.FrmN le 500 then increment=thisfitcond.FrmN-thisfitcond.Frm0+1

wxsz=1024 & wysz=1024
dsz=xsz>ysz
mgw=((wxsz<wysz))/dsz

n_cluster_nodes_max = 128
nloops = Fix((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/increment) < n_cluster_nodes_max			;nloops=Fix((framelast-framefirst)/increment)
;don't allow to use more then n_cluster_nodes_max cluster cores
increment = long(floor((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/nloops))
nloops = fix(ceil((thisfitcond.FrmN-thisfitcond.Frm0+1.0)/increment))

if DisplayType eq 3 then begin 	;set to 3 (--> -1) - Cluster
	DisplayType=-1			;turns of all displays during processing
	if !VERSION.OS_family eq 'unix' then	idl_pwd=pref_get('IDL_MDE_START_DIR') else idl_pwd=pref_get('IDL_WDE_START_DIR')
	cd,current=curr_pwd
	FILE_MKDIR,curr_pwd+'/temp'
	save,curr_pwd,idl_pwd,pth, filen, thisfitcond, increment, nloops, SigmaSym, filename='temp/temp.sav'		;save variables for cluster cpu access
	ReadRawLoopCluster
	file_delete,'temp/temp.sav'
	file_delete,'temp'
	cd,curr_pwd
	restore,saved_pks_filename
endif else begin
for nlps=0L,nloops-1 do begin											;loop for all file chunks
	framefirst=	thisfitcond.Frm0 + nlps*increment					;first frame in batch
	framelast=((thisfitcond.Frm0 + (nlps+1L)*increment-1L)<thisfitcond.Nframesmax) < thisfitcond.FrmN
	Nframes=(framelast-framefirst+1) 								;number of frames to extract in file
	data=float(ReadData(thefile_no_exten,thisfitcond,framefirst,Nframes))	;Reads thefile and returns data (bunch of frames) in (units of photons)
	mg=2
	scl=4.															;intensity scaling Range = scl* # electrons
	if DisplayType eq 2 then Showframes, data,xsz,ysz,mg, Nframes,scl						;Shows time movie of data

	mg=mgw
	if Nframes gt 1 then totdat=total(data[*,*,0:Nframes-1],3)/Nframes else totdat=data
	totaldata=rebin(totdat*Nframes,xsz*mg,ysz*mg,/sample)
	if (DisplayType ge 1) then ShowIt,totaldata,mag=mg,wait=1.0
															;Get parameters of one bunch of frames
	if DisplayType lt 4 then Apeakparams=ParamsofShortStackofFrames(data,DisplayType,thisfitcond,framefirst,SigmaSym)
	if DisplayType eq 4 then begin
		print,'loaded data block',nlps+1,'
		t0 = SYSTIME(/SECONDS)
		Apeakparams=ParamsofLongStackofFrames(data,DisplayType,thisfitcond,framefirst,SigmaSym)

	endif
	;Apeakparams[*].frameindex+=framefirst						;adjust frame index to include batch offset
	mg=2
	loc=fltarr(xsz*mg,ysz*mg)
	filter=((Apeakparams.fitok eq 1) or (Apeakparams.fitok eq 2))
	loc[[mg*Apeakparams.peakx],[mg*Apeakparams.peaky]]=255*filter
	image=float(loc)

	If DisplayType ge 1 then begin
		ShowIt,totaldata,mag=mg,wait=1.0
		ShowIt,loc,mag=mg,wait=1.0
	endif
	;--------------------------
	if nlps eq 0 then save,Apeakparams,image,xsz,ysz,totaldata,filename=thefile_no_exten+'_'+strtrim(framefirst,2)+'-'+strtrim(framelast,2)+'_IDL.pks',thefile_no_exten
	if nlps gt 0 then begin
		NApeakparams=Apeakparams
		Ntotaldata=totaldata
		Nimage=image
		restore,filename=thefile_no_exten+'_'+strtrim(thisfitcond.Frm0,2)+'-'+strtrim(framefirst-1,2)+'_IDL.pks'
		Apeakparams=[Apeakparams,NApeakparams]
		totaldata=(totaldata*nlps + Ntotaldata)/(nlps+1)
		image=(image*nlps+Nimage)/(nlps+1)
		save,Apeakparams,image,xsz,ysz,totaldata,filename=thefile_no_exten+'_'+strtrim(thisfitcond.Frm0,2)+'-'+strtrim(framelast,2)+'_IDL.pks',thefile_no_exten
		file_delete,thefile_no_exten+'_'+strtrim(thisfitcond.Frm0,2)+'-'+strtrim(framefirst-1,2)+'_IDL.pks'
	endif
	saved_pks_filename=thefile_no_exten+'_'+strtrim(thisfitcond.Frm0,2)+'-'+strtrim(framelast,2)+'_IDL.pks'
	print,'Wrote file '+saved_pks_filename
	if framelast gt thisfitcond.FrmN then return
endfor
endelse

xydsz=[xsz,ysz]
sz=size(Apeakparams)
CGroupParams=dblarr(CGrpSize,sz[1])
CGroupParams[0:1,*]=ApeakParams.A[0:1]
CGroupParams[2,*]=ApeakParams.peakx
CGroupParams[3,*]=ApeakParams.peaky
CGroupParams[4:5,*]=ApeakParams.A[2:3]
CGroupParams[6,*]=ApeakParams.NPhot
CGroupParams[7,*]=ApeakParams.ChiSq
CGroupParams[8,*]=ApeakParams.FitOK
CGroupParams[9,*]=ApeakParams.FrameIndex
CGroupParams[10,*]=ApeakParams.PeakIndex
CGroupParams[11,*]=dindgen(sz[1])
CGroupParams[12,*]=ApeakParams.A[2]*ApeakParams.A[3]
CGroupParams[13,*]=ApeakParams.Sigma2[1]
CGroupParams[14:15,*]=ApeakParams.Sigma2[4:5]
CGroupParams[16:17,*]=ApeakParams.Sigma2[2:3]
TotalRawData = TotalData
if CGrpSize ge 49 then CGroupParams[43,*]=(ApeakParams.A[2]-ApeakParams.A[3])/(ApeakParams.A[2]+ApeakParams.A[3])

FlipRotate=replicate({frt,present:0B,transp:0B,flip_h:0B,flip_v:0B},3)
NFrames=thisfitcond.Nframesmax    ;   long64(max(CGroupParams[9,*]))
GuideStarDrift=replicate({present:0B,xdrift:dblarr(Nframes+1),ydrift:dblarr(Nframes+1),zdrift:dblarr(Nframes+1)},3)
FiducialCoeff=replicate({fidcoef,present:0U,P:dblarr(2,2),Q:dblarr(2,2)},3)
RawFilenames=strarr(3)
RawFilenames[0]=thefile_no_exten

return
end
;
