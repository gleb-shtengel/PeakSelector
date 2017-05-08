;$File: //depot/gsg/HHMI/Phase2/src/palmgr3dmodel__define.pro $
;$Revision: #52 $
;$Change: 150764 $
;$DateTime: 2010/02/05 11:10:17 $
;$Author: rkneusel $


;------------------------------------------------------------------------------
;+
; This lifecycle method is called during the destruction of the PALMgr3DModel
; object via OBJ_DESTROY.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALMgr3DModel::Cleanup
    self -> Destruct
end


;------------------------------------------------------------------------------
;+
;  Hide/Show EM and/or molecule volumes
;-
;------------------------------------------------------------------------------
pro PALMgr3DModel::ShowVolumes, vols
    compile_opt idl2, logical_predicate

    case vols of
        'em':
        'mol':
        'both':
        else:  return  ; ignore completely
    endcase

    ;  Update the 3D display... [TODO]
end


;------------------------------------------------------------------------------
;+
;  Remove and destroy the EM volume object.
;-
;------------------------------------------------------------------------------
pro PALMgr3DModel::ClearEMVolume
    compile_opt idl2, logical_predicate

    ;  remove the object from the model...
    obj_destroy, self.oEMVolume
    ;  force 3D display to redraw...
end


;------------------------------------------------------------------------------
;+
; This method scales the volume to fit in the viewplane
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALMgr3DModel::ConvertCoords

    self.oVolume -> GetProperty, XRANGE=xRange, $
                                 YRANGE=yRange, $
                                 ZRANGE=zRange
    ranges = [[xRange], [yRange], [zRange]]
    max_range = max(ranges[1,*]-ranges[0,*], indexMax)
    refRange = ranges[*,indexMax]
    xs = [-(xRange[0])+((refRange[1]-refRange[0])/2-(xRange[1]-xRange[0])/2),1]/(refRange[1]-refRange[0])
    ys = [-(yRange[0])+((refRange[1]-refRange[0])/2-(yRange[1]-yRange[0])/2),1]/(refRange[1]-refRange[0])
    zs = [-(zRange[0])+((refRange[1]-refRange[0])/2-(zRange[1]-zRange[0])/2),1]/(refRange[1]-refRange[0])
    xs[0] = xs[0] + (-0.5)
    ys[0] = ys[0] + (-0.5)
    zs[0] = zs[0] + (-0.5)
    self.oVolume->SetProperty, XCOORD_CONV=xs, YCOORD_CONV=ys, ZCOORD_CONV=zs

end


;------------------------------------------------------------------------------
;+
; Cleans up object references and pointers
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALMgr3DModel::Destruct

    ptr_free, self.pFilterList, $
        self.pParamLimits, $
        self.pRawData, $
        self.pVolume,  $
        self.pLabelSet, $
        self.pParams
    obj_destroy, self.oVolume
    obj_destroy, self.oEMVolume
    self -> IDLgrModel::Cleanup

end


;------------------------------------------------------------------------------
;+
; This method filters out all points outside the data range
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @param ParameterLimits {in}{type=5x45 array}{required}
;   The peak parameters read from PeakSelector.
;
; @param CGroupParams {in}{type=matrix}{required}
;   Peak information, from PeakSelector.
;
; @param CGrpSize {in}{type=integer}{required}
;   Group size.
;
; @keyword GROUP {in}{type=boolean}{optional}
;   If set, do groups versus all peaks
;-
;------------------------------------------------------------------------------
function PALMgr3DModel::Filter, ParameterLimits, CGroupParams, CGrpSize, $
    GROUP=group

    compile_opt idl2, hidden
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return, -1
    endif

    tlb = PALM_Dialog( $
        TEXT='  *** Filtering Data ***  ', $
        TITLE='Initializing...')

    doGroup = keyword_set(group)
    GroupParamSize=size(CGroupParams)
    indexAll=indgen(CGrpSize)

    indexTemp = doGroup ? $
        (CGrpSize ge 43) ? [9,indexAll[18:24],26,37] : [9,indexAll[18:24],26] : $
        [indexAll[0:17],indexAll[26:(36<(CGrpSize-1))]]

    if n_elements(ParameterLimits) EQ 0 then $
        ParameterLimits = ParamLimits

    indexNonZero = where(ParameterLimits[indexTemp,1] NE 0)
    index=(temporary(indexTemp))[temporary(indexNonZero)]
    pl0=ParameterLimits[index,0]#replicate(1,GroupParamSize[2])
    pl1=ParameterLimits[index,1]#replicate(1,GroupParamSize[2])

    filter0=(CGroupParams[index,*] GE pl0) AND $
        (CGroupParams[index,*] LE pl1)
    newFilter=floor(total(temporary(filter0),1)/n_elements(temporary(index)))
    if doGroup then $
        newFilter *= (CGroupParams[25,*] EQ 1)

    widget_control, tlb, /DESTROY
    return, where(temporary(newFilter))
end


;------------------------------------------------------------------------------
;+
;  Decide if the dataset has multiple molecules or not.
;
;  @returns True (1) if there are multiple data sets, false (0) otherwise.
;-
;------------------------------------------------------------------------------
function PALMgr3DModel::HasMultipleMolecules
    compile_opt idl2, logical_predicate
    on_error, 2

    if ~ptr_valid(self.pLabelSet) then  $
        return, 0b

    ;  If label set min != max, more than one type
    mn = min(*self.pLabelSet, MAX=mx)
    return, (mn ne mx)
end


;------------------------------------------------------------------------------
;+
;  Generate a volume for the given set of peaks.
;
; @returns  A volume generate from the given set of peaks.
;
; @param peaks {in}{type=6xN}{required}
;   The peaks from which the volume is to be generated.  [x,y,z,sx,sy,sz] per row.
;
; @keyword CANCEL {out}{type=boolean}{optional}
;   If set, user canceled the volume generation.
;
; @keyword KEEP_RAW {in}{type=boolean}{optional}
;   If set, keep the raw data (necessary for multiple moluecules).
;
; @keyword CONSTANT_INTENSITY {in}{type=boolean}{optional}
;   If set, scale the volume data in an attempt to make up for spreading the
;   Gaussian values across ever more voxels as one zooms in.
;-
;------------------------------------------------------------------------------
function PALMgr3DModel::GenerateVolumeFromPeaks, peaks, CANCEL=cancel, KEEP_RAW=keep_raw,  $
                                                 CONSTANT_INTENSITY=const
    compile_opt idl2, logical_predicate

    self.xRange = [min((*self.pRawData)[0,*], MAX=maxVal), maxVal]
    self.yRange = [min((*self.pRawData)[1,*], MAX=maxVal), maxVal]
    self.zRange = [min((*self.pRawData)[2,*], MAX=maxVal), maxVal]

    xDiff = self.xRange[1]-self.xRange[0]
    yDiff = self.yrange[1]-self.yrange[0]
    zDiff = self.zRange[1]-self.zRange[0]
    diff = [xDiff, yDiff, zDiff]

    zScale = self.zScaleFactor

    ; Define the volume dimensions
    void = max(diff, indexMax)
    case 1 of
        (diff[indexMax] EQ xDiff): begin
            xDim = self.MaxVolumeDimension
            df = xDiff/double(self.MaxVolumeDimension)
            yDim = round(yDiff/df)
            zDim = round(zScale*zDiff/df)
        end
        (diff[indexMax] EQ yDiff): begin
            yDim = self.MaxVolumeDimension
            df = yDiff/double(self.MaxVolumeDimension)
            xDim = round(xDiff/df)
            zDim = round(zScale*zdiff/df)
        end
        (diff[indexMax] EQ zDiff): begin
        ; Ignore z-scaling
            zScale = 1.0
            zDim = self.MaxVolumeDimension
            df = zDiff / double(self.MaxVolumeDimension)
            xDim = round(xDiff/df)
            yDim = round(yDiff/df)
        end
    endcase

    ;  Envelope or sum
    doEnvelope = strupcase(self.Accumulation) NE 'SUM'

    ;  Fill in the data volume
    nPeaks = (size(peaks, /DIM))[1]

    ;  Define the volume data
    volData = fltarr(xDim,yDim,zDim)

    SubVolFitRad=fix(0.5D0+self.SubVolumeWidth*1.D0/df)   ;Set Gauss render radius (in Voxels) = SubVolFitRad*10 nm
    subvolrange=2*SubVolFitRad+1    ;x,y,z slope in volume pixels zero at center (odd number count)
    f = findgen(subvolrange) - SubVolFitRad
    x_axis_subVol=f;findgen(subvolrange)-SubVolFitRad
    y_axis_subVol=f;findgen(subvolrange)-SubVolFitRad
    z_axis_subVol=temporary(f);findgen(subvolrange)-SubVolFitRad
    Unity_axis_sub_Vol=replicate(1.0,subvolrange)

    modVal = Ceil(nPeaks/20.)
    oStatusBar = obj_new('PALM_StatusBar', $
        COLOR=[0,0,255], $
        TEXT_COLOR=[255,255,255], $
        TITLE='Generating Volume...', $
        TOP_LEVEL_BASE=tlb)

    cancel = 0

    ;
    ;  Loop over all peaks
    ;
    start=systime(2)
    ssx = reform(peaks[3,*])
    ssy = reform(peaks[4,*])
    ssz = reform(peaks[5,*])
    amp = (1.0d / ((2.*!pi)^1.5*(ssx/df)*(ssy/df)*(ssz/df)))<1.d

    sgx = (ssx/df)^2
    sgy = (ssy/df)^2
    sgz = (ssz*zscale/df)^2

    xsc = self.xRange[0]/df
    ysc = self.yRange[0]/df
    zsc1 = zscale/df
    zsc2 = self.zRange[0]*zsc1
    xdm1 = xdim - 1L
    ydm1 = ydim - 1L
    zdm1 = zdim - 1L
    SubVolFitRad2 = SubVolFitRad*2
    for i = 0LL, nPeaks-1 do begin


        ;  Get the position and standard deviation of this peak
        xyz = reform(peaks[*,i])
        xp = xyz[0]  &  yp = xyz[1]  &  zp = xyz[2]

        ;  Map the position onto the volume grid, ensure within the data volume
        ;  This is only to define the subvolume to compute over
        ip = long(xp/df - xsc)
        jp = long(yp/df - ysc)
        kp = long(zp*zsc1 - zsc2)

        ip = (ip > 0) < xdm1
        jp = (jp > 0) < ydm1
        kp = (kp > 0) < zdm1

        ;-----------------------------------------------------------------------------
        ;  Define the range of the voxels in a sub-region around the peak position and
        ;  deal with truncation if it is clipped by the edge
        ;  i0-i1, j0-j1, k0-k1 are ranges for full volData array and i0b-i0e, j0b-j0e,
        ;  k0b-k0e are ranges for smaller subregion around peak
        i0 = (ip - SubVolFitRad) > 0            &   i0b = (SubVolFitRad - ip) > 0
        j0 = (jp - SubVolFitRad) > 0            &   j0b = (SubVolFitRad - jp) > 0
        k0 = (kp - SubVolFitRad) > 0            &   k0b = (SubVolFitRad - kp) > 0
        i1 = (ip + SubVolFitRad) < xdm1     &   i0e = SubVolFitRad2 + ((xdm1 - (SubVolFitRad + ip)) < 0)
        j1 = (jp + SubVolFitRad) < ydm1     &   j0e = SubVolFitRad2 + ((ydm1 - (SubVolFitRad + jp)) < 0)
        k1 = (kp + SubVolFitRad) < zdm1     &   k0e = SubVolFitRad2 + ((zdm1 - (SubVolFitRad + kp)) < 0)
 ;       if ((i1-i0) ne (i0e-i0b)) || ((j1-j0) ne (j0e-j0b)) || ((k1-k0) ne (k0e-k0b)) then begin
 ;           print,i0,i1,j0,j1,k0,k1,i0b,i0e,j0b,j0e,k0b,k0e
 ;           stop
 ;       endif
        dx = i1 - i0 + 1
        dy = j1 - j0 + 1
        dz = k1 - k0 + 1

        xpFracShift=x_axis_subVol - ((xp/df mod 1.d) -0.5d)*Unity_axis_sub_Vol
        ypFracShift=y_axis_subVol - ((yp/df mod 1.d) -0.5d)*Unity_axis_sub_Vol
        zpFracShift=z_axis_subVol - ((zp/df mod 1.d) -0.5d)*Unity_axis_sub_Vol

        expA= exp(-0.5d*(xpFracShift[i0b:i0e]^2/sgx[i]))
        ;expB= exp(-0.5d*(xpFracShift[j0b:j0e]^2/sgy[i]))
        expB= exp(-0.5d*(ypFracShift[j0b:j0e]^2/sgy[i]))
        expC= exp(-0.5d*(zpFracShift[k0b:k0e]^2/sgz[i]))
        Gauss2D = reform(expB#(amp[i]*expC),dy*dz)
        R = reform(expA#Gauss2D, dx, dy, dz)
        if doEnvelope then begin
            volData[i0,j0,k0] = volData[i0:i1,j0:j1,k0:k1] > R
        endif else begin
            volData[i0,j0,k0] = volData[i0:i1,j0:j1,k0:k1] + R
        endelse
        ;-----------------------------------------------------------------------------
        if ~(i MOD modVal) then begin
            event = widget_event(tlb, /NOWAIT)
            if tag_names(event, /STRUCT) NE 'WIDGET_NOEVENT' then begin
                cancel = 1
                obj_destroy, oStatusBar
                self->Destruct
                return, 1
            endif

            oStatusBar -> UpdateStatus, float(i)/nPeaks
        endif

    endfor
    ;-----------------------------------------------------------

    print,systime(2)-start,' seconds render time'

    ; Don't need the raw data anymore
    if (~keyword_set(keep_raw)) then  $
        ptr_free, self.pRawData

    if (obj_valid(oStatusBar)) then  $
        obj_destroy, oStatusBar

    if ptr_valid(self.pVolume) then $
        ptr_free, self.pVolume

    ;  Scale for constant intensity if desired
    if (keyword_set(const)) then begin
        meanVal = mean(volData)
        self.oMainGUI->GetProperty, MEAN_VOXEL_BRIGHTNESS=meanVoxBrightness
        if (meanVoxBrightness eq 0.0) then  meanVoxBrightness = meanVal
        volData = temporary(volData)*(meanVoxBrightness / meanVal)
        self.meanVal = meanVal
    endif

    return, volData
end


;------------------------------------------------------------------------------
;+
; Filter the peaks for the selected molecule types.
;
; @returns The peaks filtered by the selected molecule types.
;
; @param arg {in}{type=matrix}{required}
;   Input peaks to be filtered.
;
; @param labels {in}{type=vector}{required}
;   Molecule numbers to be kept.
;-
;------------------------------------------------------------------------------
function PALMgr3DModel::FilterPeaksBySelectedMoleculeType, arg, labels
    compile_opt idl2, logical_predicate

    selected = widget_info(self.oMainGUI->Get('MM_SHOW_MOL'), /DROPLIST_SELECT)

    case (selected) of
        0:  ans = arg  ;  all
        1: begin
            idx = where(labels eq 4, count)  ;  4 only
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
        2: begin
            idx = where(labels eq 5, count)  ;  5 only
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
        3: begin
            idx = where(labels eq 6, count)  ;  6 only
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
        4: begin
            idx = where(labels eq 7, count)  ;  7 only
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
        5: begin
            idx = where((labels eq 4) or (labels eq 5), count)  ;  4 & 5
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
        6: begin
            idx = where((labels eq 4) or (labels eq 6), count)  ;  4 & 6
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
        7: begin
            idx = where((labels eq 4) or (labels eq 7), count)  ;  4 & 7
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
        8: begin
            idx = where((labels ge 4) and (labels le 6), count)  ;  4, 5, 6
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
        9: begin
            idx = where((labels eq 4) or (labels eq 5) or (labels eq 7), count)  ;  4, 5, 7
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
       10: begin
            idx = where((labels eq 4) or (labels eq 6) or (labels eq 7), count)  ;  4, 6, 7
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
       11: begin
            idx = where((labels eq 5) or (labels eq 6), count)  ;  5 & 6
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
       12: begin
            idx = where((labels eq 5) or (labels eq 7), count)  ;  5 & 7
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
       13: begin
            idx = where((labels ge 5) and (labels le 7), count)  ;  5, 6, 7
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
       14: begin
            idx = where((labels eq 6) or (labels eq 7), count)  ;  6 & 7
            ans = (count ne 0) ? reform(arg[*,idx])  $
                               : 0
        end
    endcase

    return, ans
end


;------------------------------------------------------------------------------
;+
;  There are multiple molecules in the input data.  Display them according to
;  the current GUI settings.
;
; @keyword CANCEL {out}{type=boolean}{optional}
;   Set if user cancels the volume generation.
;
; @keyword BRIGHT {in}{type=number}{required}
;   Brightness slider value.
;
; @keyword GAMMA_RENDER {in}{type=number}{required}
;   Gamma slider value.
;
; @keyword COLORTABLE {in}{type=3x256 array}{required}
;   The color table to use, if not using hue.
;
; @keyword USE_HUE {in}{type=boolean}{required}
;   If set, use hue to represent Z position.
;
; @keyword USE_EDM {in}{type=boolean}{required}
;   If set, use Euclidean distance mapping in the volume generation.
;
; @keyword CONSTANT_INTENSITY {in}{type=boolean}{required}
;   If set, apply constant intensity scaling to keep the intensity constant
;   when zooming.
;-
;------------------------------------------------------------------------------
pro PALMgr3DModel::DisplayMultipleMolecules, CANCEL=cancel, BRIGHT=bright, GAMMA_RENDER=gammaRender, $
                            COLORTABLE=ColorTable, USE_HUE=useHue, USE_EDM=useEDM, CONSTANT_INTENSITY=const
    compile_opt idl2, logical_predicate

    if (~widget_info(self.oMainGUI->Get('MM_HUE'), /BUTTON_SET)) then begin
        ;  Filter peaks by selected molecule types
        peaks = self->FilterPeaksBySelectedMoleculeType(*self.pRawData, *self.pLabelSet)

        if (n_elements(peaks) eq 1) then begin
            self.oMainGUI->GetProperty, TLB=tlb
            void = dialog_message(/ERROR, 'No peaks match the selected molecule types.',  $
                                  DIALOG_PARENT=tlb)
            cancel = 1
            return
        end

        ;  Generate the volume for these peaks
        volData = self->GenerateVolumeFromPeaks(peaks, CANCEL=cancel, CONSTANT=const)
        volData = (temporary(volData)*bright) < 1.0
        volData = temporary(volData)^(gammaRender)
        self.maxVolumeElement = max(volData)
        self.pVolume = ptr_new(volData, /NO_COPY)

        if n_elements(ColorTable) NE 768 then begin
            oPalette = obj_new('IDlgrPalette')
            oPalette -> LoadCT, 3
            oPalette -> GetProperty, BLUE=blue, GREEN=green, RED=red
            ColorTable = [[red],[green],[blue]]
        endif else begin
            ColorTable = reform(byte(ColorTable),256,3)
        endelse

        self.useEDM = (n_elements(useEDM) eq 0) ? 1 : useEDM
        void = self->IDLgrModel::Init(/SELECT_TARGET)

        if (useHue ne 1) then begin
            self.oVolume = obj_new('IDLgrVolume', *self.pVolume, $
                /INTERPOLATE, HINTS=2+self.useEDM, NAME='VolumeObject')
        endif

        ;  Generate a normal or hue volume
        if (useHue eq 1) then begin
            self->UpdateHueVolume
        endif else begin
            self->UpdateVolume
            if n_elements(ColorTable) GT 0 then begin
                self->SetProperty, COLOR_TABLE=ColorTable
            endif
        endelse
    endif else begin
        ;  Fancy multiple molecule display
        ;allLabels = ['4 5 6 7','4','5','6','7','4 5','4 6','4 5 6','4 5 7','4 6 7','5 6','5 7','5 6 7','6 7']
        allLabels = ['1 2 3 4','1','2','3','4','1 2','1 3','1 2 3','1 2 4','1 3 4','2 3','2 4','2 3 4','3 4']
        selected = widget_info(self.oMainGUI->Get('MM_SHOW_MOL'), /DROPLIST_SELECT)
        labels = fix(strsplit(allLabels[selected],/EXTRACT))

        ;  Get the hue values for the molecule types
        widget_control, self.oMainGUI->Get('MM_DRAW1_SLIDER'), GET_VALUE=hue1
        widget_control, self.oMainGUI->Get('MM_DRAW2_SLIDER'), GET_VALUE=hue2
        widget_control, self.oMainGUI->Get('MM_DRAW3_SLIDER'), GET_VALUE=hue3
        widget_control, self.oMainGUI->Get('MM_DRAW4_SLIDER'), GET_VALUE=hue4
        hues = [hue1, hue2, hue3, hue4]

        ;  Loop over all the labels
        for i=0, n_elements(labels)-1 do begin
            ;  Get all the peaks for this molecule type, ignoring empty types
            idx = where(*self.pLabelSet eq labels[i], count)
            if (count eq 0) then continue

            ;  Generate a volume from this (apply gamma and brightness)
            volData = self->GenerateVolumeFromPeaks((*self.pRawData)[*,idx], CANCEL=cancel, /KEEP_RAW, CONSTANT=const)
            volData = (temporary(volData)*bright) < 1.0
            volData /= max(volData)
            volData = temporary(volData)^(gammaRender)

            ;  Convert this volume to HSV using the hue for this molecule type
            dims = size(volData, /DIM)
            hvol = replicate(hues[labels[i]-1], dims)
            vvol = temporary(volData) < 1.0
            svol = replicate(1.0, dims)

            ;  Convert to RGB volumes
            color_convert, hvol, svol, vvol, rvol, gvol, bvol, /HSV_RGB

            ;  Add into output volume
            if (i eq 0) then begin
                outr = fix(temporary(rvol))
                outg = fix(temporary(gvol))
                outb = fix(temporary(bvol))
                outv = temporary(vvol)
            endif else begin
                outr += temporary(rvol)
                outg += temporary(gvol)
                outb += temporary(bvol)
                outv += temporary(vvol)
            endelse
        endfor

        ;  Set the output volume as if it were an ordinary hue volume
        outr = byte(temporary(outr) < 255)
        outg = byte(temporary(outg) < 255)
        outb = byte(temporary(outb) < 255)
        outv = outv < 1
        self.maxVolumeElement = max(outv)
        self.pVolume = ptr_new(outv, /NO_COPY)

        self.useEDM = (n_elements(useEDM) eq 0) ? 1 : useEDM
        void = self->IDLgrModel::Init(/SELECT_TARGET)

        ;  Set the volume object for display
        obj_destroy, self.oVolume
        self.oVolume = obj_new('IDLgrVolume', DATA0=outr, DATA1=outg, DATA2=outb, DATA3=bytscl(*self.pVolume),  $
                               VOLUME_SELECT=2, COMPOSITE=self.Composite, NAME='VolumeObject', $
                               HINTS=2+self.useEDM, /INTERPOLATE)
        self->ConvertCoords
        self.hue = 1b
        ptr_free, self.pRawData
    endelse
end


;------------------------------------------------------------------------------
;+
; This method generates the volume from the selected peak data.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @keyword AUTO {in}{type=boolean}{optional}
;   Unused.
;
; @keyword CANCEL {out}{type=boolean}{optional}
;   Set if the user cancels the volume generation.
;
; @keyword DPMAX {in}{type=boolean}{optional}
;   Unused.
;
; @keyword DPMIN {in}{type=boolean}{optional}
;   Unused.
;
; @keyword LOG {in}{type=boolean}{optional}
;   If set, use log.
;
; @keyword GAMMA_RENDER {in}{type=number}{optional}
;   The gamma value.
;
; @keyword BRIGHTNESS {in}{type=number}{optional}
;   The brightness value.
;
; @keyword ERROR {out}{type=string}{optional}
;   The error message, if any.
;
; @keyword COLORTABLE {in}{type=3x256 matrix}{optional}
;   The color table to use, if not hue.
;
; @keyword USE_HUE {in}{type=boolean}{optional}
;   If set, use hue for Z position.
;
; @keyword USE_EDM {in}{type=boolean}{optional}
;   If set, use Euclidean distance mapping when creating the volume.
;
; @keyword CONSTANT_INTENSITY {in}{type=boolean}{optional}
;   If set, use a constant intensity multiplier to account for zooming.
;-
;------------------------------------------------------------------------------
function PALMgr3DModel::GenerateVolume, $
    AUTO=auto, $
    CANCEL=cancel,  $
    DPMAX=dpmax,  $
    DPMIN=dpmin,  $
    LOG=log,      $
    GAMMA_RENDER=gamma0,  $
    BRIGHTNESS=bright0, $
    ERROR=ec,  $
    COLORTABLE=ColorTable, $
    USE_HUE=useHue,  $
    USE_EDM=useEDM,  $
    CONSTANT_INTENSITY=const

    compile_opt idl2, hidden
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return, 0
    endif

    ec = 1b

    bright = (n_elements(bright0) eq 0) ? 1.0 : float(bright0)
    gammaRender = (n_elements(gamma0) eq 0) ? 1.0 : float(gamma0)

    ;  Display
    if (self->HasMultipleMolecules()) then begin
        ;
        ;  Multiple molecule types
        ;
        self->DisplayMultipleMolecules, CANCEL=cancel, BRIGHT=bright, GAMMA_RENDER=gammaRender, $
                                        COLORTABLE=ColorTable, USE_HUE=useHue, USE_EDM=useEDM,  $
                                        CONSTANT=const
    endif else begin
        ;
        ;  Single molecule type
        ;
        volData = self->GenerateVolumeFromPeaks(*self.pRawData, CANCEL=cancel, CONSTANT=const)
        volData = (temporary(volData)*bright) < 1.0
        volData /= max(volData)
        volData = temporary(volData)^(gammaRender)
        self.maxVolumeElement = max(volData)
        self.pVolume = ptr_new(volData, /NO_COPY)

        if n_elements(ColorTable) NE 768 then begin
            oPalette = obj_new('IDlgrPalette')
            oPalette -> LoadCT, 3
            oPalette -> GetProperty, BLUE=blue, GREEN=green, RED=red
            ColorTable = [[red],[green],[blue]]
        endif else begin
            ColorTable = reform(byte(ColorTable),256,3)
        endelse

        self.useEDM = (n_elements(useEDM) eq 0) ? 1 : useEDM
        void = self->IDLgrModel::Init(/SELECT_TARGET)

        if (useHue ne 1) then begin
            self.oVolume = obj_new('IDLgrVolume', *self.pVolume, $
                /INTERPOLATE, HINTS=2+self.useEDM, NAME='VolumeObject')
        endif

        ;  Generate a normal or hue volume
        if (useHue eq 1) then begin
            self->UpdateHueVolume
        endif else begin
            self->UpdateVolume
            if n_elements(ColorTable) GT 0 then begin
                self->SetProperty, COLOR_TABLE=ColorTable
            endif
        endelse
    endelse

    return, self.oVolume
end


;------------------------------------------------------------------------------
;+
;  Get object properties.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @keyword ACCUMULATION {out}{optional}
;   Get the accumulation mode.
;
; @keyword COLOR_TABLE {out}{optional}
;   Get the volume color table.
;
; @keyword FILTER_INDEX {out}{optional}
;   Get the filter indices.
;
; @keyword FILTER_LIST {out}{optional}
;   Get the filter list.
;
; @keyword FUNCTION_INDEX {out}{optional}
;   Get the function indices.
;
; @keyword HUE {out}{optional}
;   Get if hue for Z set.
;
; @keyword EDM {out}{optional}
;   Get if Euclidean distance mapping set.
;
; @keyword NANOS_PER_CCD_PIXEL {out}{optional}
;   Get CCD pixel size in nm.
;
; @keyword OPACITY_TABLE {out}{optional}
;   Get the volume opacity table
;
; @keyword PARAMETER_LIMITS {out}{optional}
;   Get the peak parameter limits.
;
; @keyword REFERENCE_RANGE {out}{optional}
;   Get the reference range.
;
; @keyword VOLUME_OBJECT {out}{optional}
;   Get the volume object reference.
;
; @keyword VOLUME_XRANGE {out}{optional}
;   Get the volume X size.
;
; @keyword VOLUME_YRANGE {out}{optional}
;   Get the volume Y size.
;
; @keyword VOLUME_ZRANGE {out}{optional}
;   Get the volume Z size.
;
; @keyword VOLUME_MAX {out}{optional}
;   Get the max volume element.
;
; @keyword X_RANGE {out}{optional}
;   Get the X range.
;
; @keyword XCOORD_CONV {out}{optional}
;   Get the X coordinate conversion.
;
; @keyword Y_RANGE {out}{optional}
;   Get the Y range.
;
; @keyword YCOORD_CONV {out}{optional}
;   Get the Y coordinate conversion.
;
; @keyword Z_RANGE {out}{optional}
;   Get the Z range.
;
; @keyword ZCOORD_CONV {out}{optional}
;   Get the Z coordinate conversion.
;
; @keyword VOLUME_POINTER {out}{optional}
;   Get a pointer to the volume data.
;-
;------------------------------------------------------------------------------
pro PALMgr3DModel::GetProperty, $
    ACCUMULATION=Accumulation, $
    COLOR_TABLE=ColorTable, $
    FILTER_INDEX=FilterIndex, $
    FILTER_LIST=FilterList, $
    FUNCTION_INDEX=FunctionIndex, $
    HUE=hue, $
    EDM=edm, $
    NANOS_PER_CCD_PIXEL=Nanos, $
    OPACITY_TABLE=OpacityTable, $
    PARAMETER_LIMITS=ParameterLimits, $
    REFERENCE_RANGE=refRange, $
    VOLUME_OBJECT=oVolume, $
    VOLUME_XRANGE=volXRange, $
    VOLUME_YRANGE=volYRange, $
    VOLUME_ZRANGE=volZRange, $
    VOLUME_MAX=volMax, $
    X_RANGE=xRange, $
    XCOORD_CONV=xs, $
    Y_RANGE=yRange, $
    YCOORD_CONV=ys, $
    Z_RANGE=zRange, $
    ZCOORD_CONV=zs, $
    VOLUME_POINTER=vp, $
    _REF_EXTRA=_extra

    compile_opt idl2, hidden
    on_error, 2

    if arg_present(vp) then  $
        vp = self.pVolume
    if arg_present(volMax) then  $
        volMax = self.maxVolumeElement
    if arg_present(hue) then  $
        hue = self.Hue
    if arg_present(edm) then  $
        hue = self.useEDM
    if arg_present(Accumulation) then $
        Accumulation = self.Accumulation
    if arg_present(ColorTable) then $
        self.oVolume -> GetProperty, RGB_TABLE0=ColorTable
    if arg_present(FilterIndex) then $
        FilterIndex = self.FilterIndex
    if arg_present(FilterList) then $
        FilterList = *self.pFilterList
    if arg_present(FunctionIndex) then $
        FunctionIndex = self.FunctionIndex
    if arg_present(Nanos) then $
        Nanos = self.Nanos
    if arg_present(OpacityTable) then $
        self.oVolume -> GetProperty, OPACITY_TABLE0=OpacityTable
    if arg_present(ParameterLimits) then $
        ParameterLimits = *self.pParamLimits
    if arg_present(refRange) then begin
        self.oVolume -> GetProperty, XRANGE=xRange, $
                                     YRANGE=yRange, $
                                     ZRANGE=zRange
        ranges = [[xRange], [yRange], [zRange]]
        max_range = max(ranges[1,*]-ranges[0,*], indexMax)
        refRange = ranges[*,indexMax]
    endif
    if arg_present(xs) then $
        self.oVolume -> GetProperty, XCOORD_CONV=xs
    if arg_present(ys) then $
        self.oVolume -> GetProperty, YCOORD_CONV=ys
    if arg_present(zs) then $
        self.oVolume -> GetProperty, ZCOORD_CONV=zs

    if arg_present(oVolume) then $
        oVolume = self.oVolume

    if arg_present(volXRange) then $
        self.oVolume -> GetProperty, XRANGE=VolXRange
    if arg_present(volYRange) then $
        self.oVolume -> GetProperty, YRANGE=VolYRange
    if arg_present(volZRange) then $
        self.oVolume -> GetProperty, ZRANGE=VolZRange

    if arg_present(xRange) then $
        xRange = self.xRange
    if arg_present(yRange) then $
        yRange = self.yRange
    if arg_present(zRange) then $
        zRange = self.zRange
    if n_elements(_extra) GT 0 then begin
        self -> IDLgrModel::GetProperty, _EXTRA=_extra
        self.oVolume -> GetProperty, _EXTRA=_extra
    endif

end


;------------------------------------------------------------------------------
;+
; Return the volume object
;
; @returns The volume object
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALMgr3DModel::GetVolObj

    if ~obj_valid(self.oVolume) then $
        return, 0

    return, self.oVolume

end


;------------------------------------------------------------------------------
;+
; Return the volume pointer.
;
; @returns A pointer to the volume data.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALMgr3DModel::GetVolPtr

    if ~ptr_valid(self.pVolume) then $
        return, 0

    return, self.pVolume

end


;------------------------------------------------------------------------------
;+
; Initialize a PALMgr3DModel object.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @keyword ACCUMULATION {in}{required}
;   Sum or envelope
;
; @keyword AUTO_FILTER {in}{required}
;   Auto filter or not.
;
; @keyword COLOR_TABLE {in}{required}
;   Default color table
;
; @keyword COMPOSITE_FUNCTION {in}{required}
;   Volume composite function (MIP, etc)
;
; @keyword DATA_RANGE {in}{required}
;   Data ranges (nm)
;
; @keyword FIDUCIAL_CUTOFF {in}{required}
;   Threshold for super "bright" fiducials
;
; @keyword FILTER_INDEX {in}{required}
;   Filter mask for selected peaks
;
; @keyword FUNCTION_INDEX {in}{required}
;   Function vector
;
; @keyword GAMMA_RENDER {in}{required}
;   Gamma value to apply to calculated volume data
;
; @keyword LOG_FILTER {in}{required}
;   If set, log filter the data
;
; @keyword MAXIMUM_VOLUME_DIMENSION {in}{required}
;   Largest possible volume dimension (other two will be less)
;
; @keyword NANOS_PER_CCD_PIXEL {in}{required}
;   CCD pixel size in nm
;
; @keyword PARAMETER_LIMITS {in}{required}
;   Peak limits for all parameters
;
; @keyword SUBVOLUME_WIDTH {in}{required}
;   Volume around a peak in which to calculate the Gaussian (nm)
;
; @keyword VERBOSE {in}{required}
;   If set, output status messages
;
; @keyword VOLUME_TYPE {in}{required}
;   Unused.
;
; @keyword USE_HUE {in}{required}
;   If set, use hue for Z position
;
; @keyword USE_EDM {in}{required}
;   If set, use Euclidean distance mapping in the volume rendering
;
; @keyword CONSTANT_INTENSITY {in}{required}
;   If set, use constant intensity scaling when zooming.
;
; @keyword Z_SCALE_FACTOR {in}{required}
;   Scale factor for expanding the Z axis
;
; @keyword GROUP_PARAMS {in}{required}
;   Unused.
;
; @keyword GROUP_SIZE {in}{required}
;   Unused.
;
; @keyword BRIGHTNESS {in}{required}
;   Brightness value applied to volume data.
;
; @keyword MAIN_GUI {in}{required}
;   Reference to the main GUI object.
;-
;------------------------------------------------------------------------------
function PALMgr3DModel::Init, $
    ACCUMULATION=Accumulation, $
    AUTO_FILTER=doAutoFilter, $
    COLOR_TABLE=ColorTable, $
    COMPOSITE_FUNCTION=Composite, $
    DATA_RANGE=dataRangeIn, $
    FIDUCIAL_CUTOFF=FiducialCutoff, $
    FILTER_INDEX=filterIndex, $
    FUNCTION_INDEX=functionIndex, $
    GAMMA_RENDER=gammaRender, $
    LOG_FILTER=doLogFilter, $
    MAXIMUM_VOLUME_DIMENSION=MaxVolumeDimension, $
    NANOS_PER_CCD_PIXEL=nanos, $
    PARAMETER_LIMITS=ParameterLimits, $
    SUBVOLUME_WIDTH=SubvolumeWidth, $
    VERBOSE=verbose, $
    VOLUME_TYPE=VolumeType, $
    USE_HUE=UseHue, $
    USE_EDM=UseEDM, $
    CONSTANT_INTENSITY=const,  $
    Z_SCALE_FACTOR=zScaleFactor,  $
    GROUP_PARAMS=GroupParams,  $
    GROUP_SIZE=GrpSize, $
    BRIGHTNESS=bright, $
    MAIN_GUI=oMainGUI, $
    PARAMS=vParams

    compile_opt idl2, hidden
    on_error, 2

	common  SharedParams, CGrpSize, CGroupParams, ParamLimits, filter, Image, b_set, xydsz, TotalRawData, DIC, RawFilenames, SavFilenames,  MLRawFilenames, GuideStarDrift, FiducialCoeff, FlipRotate

    ;  Set up the parameters to use for x,y,z and sx,sy,sz
    self.defParams = [19,20,40,21,22,41]
    if (n_elements(vParams) eq 0) then begin
        self.pParams = (filterIndex ne 1) ? ptr_new([2,3,34,16,17,35])    $
                                          : ptr_new([19,20,40,21,22,41])
    endif else begin
        self.pParams = ptr_new(vParams)
    endelse

    ;  Trap for multiple molecules (development)
    if (getenv('PALM_MM') ne '') then begin
        print
        print, '***********************************************************************'
        print, 'Restore the sample multiple molecule dataset, then type .cont'
        print, '***********************************************************************'
        print
        stop
    endif

    ;  A reference to the main GUI
    self.oMainGUI = (obj_valid(oMainGUI)) ? oMainGUI : obj_new()

    self.verbose = keyword_set(verbose)

    ;  Use keyword values in place of common block, if given
    if (n_elements(GroupParams) ne 0) then begin
        CGroupParams = GroupParams
    endif
    if (n_elements(GrpSize) ne 0) then begin
        CGrpSize = GrpSize
    endif
    if (n_elements(ParameterLimits) ne 0) then begin
        ParamLimits = ParameterLimits
    endif

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        self -> Destruct
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return, 0
    endif

    if n_elements(filterIndex) EQ 0 OR $
       n_elements(functionIndex) EQ 0 OR $
       n_elements(nanos) EQ 0 OR $
       n_elements(ParameterLimits) EQ 0 then begin

        void = dialog_message(/ERROR, $
            ['The following keywords must be set for initialization:', $
             'FILTER_INDEX', $
             'FUNCTION_INDEX', $
             'NANOS_PER_CCD_PIXEL'])
        return, 0

    endif

    self.FilterIndex = FilterIndex[0]
    self.FunctionIndex = FunctionIndex[0]
    self.Nanos = Nanos
    self.pParamLimits = ptr_new(ParameterLimits)
    FilterList = self -> Filter(ParameterLimits, CGroupParams, CGrpSize, $
        GROUP=(FilterIndex EQ 1))
    if FilterList[0] EQ -1 then begin
        self -> Destruct
        return, 0
    endif
    self.pFilterList = ptr_new(FilterList, /NO_COPY)
    self.Accumulation = n_elements(Accumulation) GT 0 ? Accumulation[0] : 'Sum'
    self.Composite = n_elements(Composite) GT 0 ? Composite[0] : 0
    self.FiducialCutoff = n_elements(FiducialCutoff) GT 0 ? FiducialCutoff[0] : 100
    self.MaxVolumeDimension = n_elements(MaxVolumeDimension) GT 0 ? $
        MaxVolumeDimension[0] : 800
    self.SubvolumeWidth = n_elements(SubvolumeWidth) GT 0 ? SubvolumeWidth[0] : 60
    self.zScaleFactor = n_elements(zScaleFactor) GT 0 ? zScaleFactor[0] : 1.0

    ; Prepare the raw data
    nparams = (size(CGroupParams,/DIM))[0]
    void = where(*self.pParams ge nparams, count)
    if (count eq 0) then begin
        RawData = CGroupParams[*self.pParams,*]
    endif else begin
        ;  Params out of range, switch to group Z
        self.oMainGUI->GetProperty, TLB=tlb
        void = dialog_message(DIALOG_PARENT=tlb, 'Z parameter undefined, switching to Group Z')
        RawData = CGroupParams[self.defParams,*]
    endelse
    RawData = RawData[*,*self.pFilterList]

    ;  Keep the label set vector (for multiple molecules)
    self.pLabelSet = ptr_new(reform(CGroupParams[26, *self.pFilterList]))

    RawData[[0,1,3,4],*] *= nanos
    self.pRawData = ptr_new(RawData, /NO_COPY)
    dataRange = n_elements(dataRangeIn) GT 1 ? dataRangeIn[0:1] : [0.0,50.0]

    self.oVolume = self->GenerateVolume(AUTO=doAutoFilter, CANCEL=cancel, DPMAX=dataRange[1],           $
                                        DPMIN=dataRange[0], LOG=doLogFilter, GAMMA_RENDER=gammaRender,  $
                                        BRIGHTNESS=bright, ERROR=ec, COLORTABLE=ColorTable,             $
                                        USE_HUE=useHue, USE_EDM=useEDM, CONSTANT=const)

    ;  Set the mean voxel brightness
    if obj_valid(self.oMainGUI) then begin
        self.oMainGUI->GetProperty, MEAN_VOXEL_BRIGHTNESS=meanVoxBrightness
        if (meanVoxBrightness eq 0.0) then begin
            self.oMainGUI->SetProperty, MEAN_VOXEL_BRIGHTNESS=self.meanVal
        endif
    endif

    if ~ec then begin
        void = dialog_message(/ERROR, 'Unable to generate volume')
        self->Destruct
        return, 0
    end
    if cancel then $
        return, 0

    self-> Add, self.oVolume
    return, 1

end


;------------------------------------------------------------------------------
;+
; Scale the output volume data.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @keyword LOG {in}{type=boolean}{optional}
;   If set, log the volume data
;
; @keyword DPMIN {in}{type=number}{optional}
;   Minimum percentage
;
; @keyword DPMAX {in}{type=number}{optional}
;   Maximum percentage
;
; @keyword AUTO {in}{type=boolean}{optional}
;   If set, auto scale.
;-
;------------------------------------------------------------------------------
pro PALMgr3DModel::ScaleOutputData, LOG=log, DPMIN=dpmin, DPMAX=dpmax, AUTO=auto
    compile_opt idl2

    if ~ptr_valid(self.pVolume) then $
        return

    if keyword_set(auto) then begin
        ;  Scale to 100*mean value
        threshold = 100*mean(*self.pVolume)
        idx = where(*self.pVolume gt threshold, count)
        if (count ne 0) then  (*self.pVolume)[idx] = threshold
    endif else begin
        if n_elements(dpMin) EQ 0 then $
            dpMin = 0.0
        if n_elements(dpMax) EQ 0 then $
            dpMax = 100.0

        ;  Validate cutoffs
        dpMin = (dpMin>0.0)<100.0
        dpMax = (dpMax>0.0)<100.0
        if (dpmin gt dpmax) then begin
            t = dpmin
            dpmin = dpmax
            dpmax = t
        endif

        ;  Find the threshold values
        wmin = min(*self.pVolume, MAX=wmax)
        tmin = (dpmin/100.0)*(wmax-wmin) + wmin
        tmax = (dpmax/100.0)*(wmax-wmin) + wmin

        ;  Set data values outside this range
        ;  to the threshold values
        idx = where(*self.pVolume lt tmin, count)
        if (count ne 0) then  (*self.pVolume)[idx] = tmin
        idx = where(*self.pVolume gt tmax, count)
        if (count ne 0) then  (*self.pVolume)[idx] = tmax
        if keyword_set(log) then $
            *self.pVolume = alog10(1+(*self.pVolume))
    endelse
end


;------------------------------------------------------------------------------
;+
; Set object properties
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @keyword COLOR_TABLE
;   Set the volume color table.
;
; @keyword HIDE
;   If set, hide the volume
;
; @keyword OPACITY_TABLE
;   Set the volume object opacity table.
;
; @keyword MAIN_GUI
;   Set the reference to the main GUI object.
;-
;------------------------------------------------------------------------------
pro PALMgr3DModel::SetProperty, $
    COLOR_TABLE=ColorTable, $
    HIDE=hide, $
    OPACITY_TABLE=OpacityTable, $
    MAIN_GUI = oMainGUI,  $
    EM_VOLUME=oEMVolume, $
    _EXTRA=_extra

    compile_opt idl2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    if (n_elements(oMainGUI) ne 0) then begin
        self.oMainGUI = oMainGUI
    endif

    if n_elements(ColorTable) EQ 768 then begin
        self.oVolume -> SetProperty, RGB_TABLE0 = reform(ColorTable,256,3)
    endif

    if n_elements(hide) GT 0 then $
        self.oVolume -> SetProperty, HIDE=hide[0]

    if n_elements(OpacityTable) EQ 256 then begin
        self.oVolume -> SetProperty, OPACITY_TABLE0=OpacityTable
    endif

    if (n_elements(oEMVolume) ne 0) then begin
        self->Remove, self.oEMVolume  ; remove the old one
        obj_destroy, self.oEMVolume   ; ensure we don't leak memory
        self.oEMVolume = oEMVolume    ; add the new one
        self->Add, oEMVolume
    endif

    if n_elements(_extra) GT 0 then begin
        self -> IDLgrModel::SetProperty, _EXTRA=_extra
        self.oVolume->SetProperty, _EXTRA = _extra
    endif
end


;------------------------------------------------------------------------------
;+
; Update the volume properties
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALMgr3DModel::UpdateVolume

    compile_opt idl2

    if ~obj_valid(self.oVolume) then $
        return
    if ~obj_isa(self.oVolume, 'IDLgrVolume') then $
        return
    if ~ptr_valid(self.pVolume) then $
        return

    self->GetProperty, COLOR_TABLE=ColorTable

    self.oVolume -> SetProperty, $
        COMPOSITE=self.Composite, $
        DATA0 = bytscl(*self.pVolume),  $
        RGB_TABLE0=ColorTable,          $
        HINTS=2+self.useEDM

    self -> ConvertCoords
    self.hue = 0b
end


;------------------------------------------------------------------------------
;+
;  Set the IDLgrVolume to display a hue colored volume
;-
pro PALMgr3DModel::UpdateHueVolume
    compile_opt idl2

    if ~ptr_valid(self.pVolume) then $
        return

    ;  HSV volumes
    dims = size(*self.pVolume, /DIM)
    hvol = fltarr(dims)
    ;vvol = (*self.pVolume)/max(*self.pVolume)
    vvol = (*self.pVolume) < 1.0
    svol = replicate(1.0, dims)

    ;  Fill in the hue according to z position
    v = 320.0*(1+findgen(dims[2]))/dims[2]  ;##HFH
    for i=0, dims[2]-1 do begin
        hvol[*,*,i] = v[i]
    endfor

    ;  Convert to RGB volumes
    color_convert, hvol, svol, vvol, rvol, gvol, bvol, /HSV_RGB

    ;  Set the volume object for display
    obj_destroy, self.oVolume
    self.oVolume = obj_new('IDLgrVolume', DATA0=rvol, DATA1=gvol, DATA2=bvol, DATA3=bytscl(vvol),  $
                           VOLUME_SELECT=2, COMPOSITE=self.Composite, NAME='VolumeObject', $
                           HINTS=2+self.useEDM, /INTERPOLATE)
    self->ConvertCoords
    self.hue = 1b
end


;------------------------------------------------------------------------------
;+
; Class definition
;
; @field Accumulation
;   Sum or envelope
;
; @field Composite
;   Volume composite value
;
; @field FiducialCutoff
;   Threshold for bright fiducials
;
; @field FilterIndex
;   Mask for selected peaks
;
; @field FunctionIndex
;   Function index
;
; @field Hue
;   Set if displaying Z with hue
;
; @field useEDM
;   Set if volume object using Euclidean distance mapping
;
; @field MaxVolumeDimension
;   Largest allowed volume dimension
;
; @field Nanos
;   CCD pixel size (nm)
;
; @field oMainGUI
;   Reference to the main GUI object
;
; @field oVolume
;   Calculated volume object
;
; @field maxVolumeElement
;   Largest volume value
;
; @field pFilterList
;   Pointer to the filter mask
;
; @field pParamLimits
;   Pointer to the parameter limits
;
; @field pRawData
;   Pointer to the raw peak data
;
; @field pLabelSet
;   Pointer to the molecule label
;
; @field pVolume
;   Pointer to the volume data
;
; @field SubvolumeWidth
;   Dimensions of volume in which to calculate the 3D Gaussian (nm)
;
; @field verbose
;   Output status message if set
;
; @field xRange
;   X range, low, high (nm)
;
; @field yRange
;   Y range, low, high (nm)
;
; @field zRange
;   Z range, low, high (nm)
;
; @field zScaleFactor
;   Z axis scale factor
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALMgr3DModel__Define
    void = {PALMgr3DModel, $
            inherits IDLgrModel, $

            Accumulation       : '',        $
            Composite          : 0,         $
            FiducialCutoff     : 0,         $
            FilterIndex        : 0,         $
            FunctionIndex      : 0,         $
            Hue                : 0b,        $
            useEDM             : 0b,        $
            MaxVolumeDimension : 0,         $
            Nanos              : 0D,        $
            oMainGUI           : obj_new(), $  ;  do not delete
            oVolume            : obj_new(), $
            oEMVolume          : obj_new(), $
            maxVolumeElement   : 0.0d,      $
            pFilterList        : ptr_new(), $
            pParamLimits       : ptr_new(), $
            pRawData           : ptr_new(), $
            pLabelSet          : ptr_new(), $
            pVolume            : ptr_new(), $
            pParams            : ptr_new(), $
            defParams          : lonarr(6), $
            meanVal            : 0.0d,      $
            SubvolumeWidth     : 0,         $
            verbose            : 0B,        $
            xRange             : dblarr(2), $
            yRange             : dblarr(2), $
            zRange             : dblarr(2), $
            zScaleFactor       : 0.0        $
           }
end
