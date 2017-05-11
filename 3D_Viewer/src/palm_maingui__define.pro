;$File: //depot/gsg/HHMI/Phase2/src/palm_maingui__define.pro $
;$Revision: #75 $
;$Change: 150764 $
;$DateTime: 2010/02/05 11:10:17 $
;$Author: rkneusel $
;
;  file:  palm_maingui__define.pro
;
;  Builds the 3D Viewer user interface
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;--------------------------------------------------------------
;+
;  Last modification date.
;-
;--------------------------------------------------------------
function MOD_DATE
    return, '05-Feb-2010'
end

;------------------------------------------------------------------------------
;+
; This method handles events from the main base widget
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from the PALM_MainGUI base widget.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::BaseEvent, event, $
    ERROR_MESSAGE=errMsg, $
    NO_UPDATE=noUpdate, $
    NO_XYZ_DRAW=noXYZDraw

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        widget_control, self.tlb, /UPDATE
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    widget_control, self.tlb, UPDATE=0

    ; Resize the top level base
    if strupcase(!version.OS_Family) EQ 'UNIX' then $
        if n_elements(event) GT 0 then $
            widget_control, self.tlb, SCR_XSIZE=event.x, SCR_YSIZE=event.y
    ; Resize the draw widget
    wGeom = widget_info(self.tlb, /GEOMETRY)
    wLabelBase = self -> Get('info_label_base')
    wLabelGeom = widget_info(wLabelBase, /GEOMETRY)
    menuGeom = widget_info(self -> Get('menu_base'), /GEOM)
    wToolbar = self -> Get('toolbar_base')
    widget_control, wToolbar, SCR_XSIZE=wGeom.xSize
    wToolGeom = widget_info(self->Get('toolbar_base'), /GEOMETRY)
    wControlBase = self -> Get('ControlBase')
    controlGeom = widget_info(wControlBase, /GEOM)
    wMiddleBase = self -> get('MiddleBase')
    wMiddleGeom = widget_info(wMiddleBase, /GEOMETRY)
    wDraw = self -> Get('MiddleDraw')
    wDrawGeom = widget_info(wDraw, /GEOMETRY)
    xSize = wGeom.xSize - 4*wGeom.yPad

    if widget_info(self->Get('menu_view_control'), /BUTTON_SET) then $
        xSize = xSize - controlGeom.scr_xSize

    ySize = wGeom.ySize - $
            menuGeom.scr_ySize - $
            wMiddleGeom.scr_ySize - $
            wLabelGeom.scr_ySize

    if widget_info(self->get('menu_view_toolbar'), /BUTTON_SET) then begin
    ; Button set
        ySize = ySize - wToolGeom.scr_ySize - wToolGeom.yPad
        wMainDraw = self -> Get('DrawBase')
        widget_control, wMainDraw, YOFFSET=wToolGeom.scr_ySize+wToolGeom.yPad
    endif else begin
    ; Button not set
        wMainDraw = self -> Get('DrawBase')
        widget_control, wMainDraw, YOFFSET=0
    endelse

    widget_control, wMiddleBase, YOFFSET=ySize*self.winPct[1]
    widget_control, self->Get('BottomBase'), YOFFSET=ySize*self.winPct[1]+wMiddleGeom.scr_ySize
    widget_control, wDraw, XOFFSET=xSize*self.winPct[0]

    uNames = 'PALM_' + ['Z','X','Y','3D'] + 'Window'
    for i = 0, n_elements(uNames)-1 do begin
        wDraw = self -> Get(uNames[i])
        widget_control, wDraw, $
            DRAW_XSIZE=xSize * (i MOD 2 ? (1-self.winPct[0]) : self.winPct[0]), $
            DRAW_YSIZE=ySize * (i LT 2 ? self.winPct[1] : (1-self.winPct[1])), $
            XOFFSET=(i MOD 2 ? xSize*self.winPct[0] + wDrawGeom.xSize : 0)
    endfor

    ; Resize the information labels
    nLabels = widget_info(wLabelBase, /N_CHILDREN)
    wLabels = widget_info(wLabelBase, /ALL_CHILDREN)
    for i = 0, nLabels-1 do $
        widget_control, wLabels, XSIZE=(wGeom.xSize-(5+nLabels)*wGeom.xPad)/nLabels

    ; Explicitly position the control base (UNIX)
    widget_control, wControlBase, XOFFSET=wGeom.scr_xSize-controlGeom.xSize

    widget_control, self.tlb, /UPDATE

    ; Update the trackball and viewplane rectangle
    if ~keyword_set(noUpdate) then begin
        self.oXYZDisplay->SetViews, NO_DRAW=noXYZDraw
        self.o3DDisplay->SetViewplane, /NO_DRAW, /UPDATE_TRACKBAL
    endif

end


;------------------------------------------------------------------------------
;+
;  Toggle the sensitivity of the "hue for Z" checkbox
;
; @param event {in}{type=structure}{required}
;   The button event.
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::MultMolButtonEvent, event
    compile_opt idl2, logical_predicate
    on_error, 2

    state = widget_info(event.id, /BUTTON_SET)

    if (state) then begin
        widget_control, self->Get('UseHue'), /SET_BUTTON
        widget_control, self->Get('UseHueMPR'), /SET_BUTTON
    endif

    widget_control, self->Get('UseHue'), SENSITIVE=~state
    widget_control, self->Get('UseHueMPR'), SENSITIVE=~state
end

;------------------------------------------------------------------------------
;+
;  Load an EM volume and display it
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::LoadEMVolume, volData, $
    NO_DRAW=noDraw, $
    OPACITY_TABLE=OpacityTable

    compile_opt idl2, logical_predicate

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if obj_valid(oSave) then $
            obj_destroy, oSave
        help, /LAST
        return
    endif

    if n_elements(volData) EQ 0 then begin
        file = dialog_pickfile( $
            DIALOG_PARENT=self.tlb, $
            GET_PATH=path, $
            /MUST_EXIST, $
            TITLE='Select an EM volume...')
        if (file EQ '') then $
            return
        if ~file_test(file, /READ) then begin
            void = dialog_message(/ERROR, $
                ['Unable to read EM volume file: ', file])
            return
        endif
        oSave = obj_new('IDL_Savefile', file)
        varNames = oSave->Names(COUNT=nVars)
        for i = 0, nVars-1 do begin
            if (oSave->Size(varNames[i]))[0] EQ 3 then begin
                oSave->Restore, varNames[i]
                volData = scope_varfetch(varNames[i])
                break
            endif
        endfor
        obj_destroy, oSave
        if n_elements(volData) EQ 0 then begin
            void = dialog_message(/ERROR, $
                'EM volume file contains no volumetric data')
            return
        endif
    endif

    hide = ~widget_info(self->Get('EM_SHOW_EM'), /BUTTON_SET)
    void = self.o3DDisplay->AddEMVolume(volData, $
        HIDE=hide, $
        /NO_DRAW, $ ; Drawing will be handled by self->EMVolumeColorTable, $
        OPACITY_TABLE=OpacityTable)
    self->EMVolumeColorTable, NO_DRAW=noDraw
    if ~void then begin
        self->DisableEMVolume
        return
    endif
    widget_control, self->Get('EM_IMAGE_OPACITY'), $
        GET_VALUE=opacity
    self.oXYZDisplay->AddEMVolume, ptr_new(volData), $
        HIDE=hide, $
        OPACITY=float(opacity)/100
    self->EnableEMVolume

end


;------------------------------------------------------------------------------
;+
;  Handle an event from an EM volume button widget
;
;  @param event {in}{type=button event}{required}
;    The event structure.
;
;  @param components {in}{type=string vector}{required}
;    Any extra components in the widget UNAME.
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::EMVolumeButtonEvent, event, components
    compile_opt idl2, logical_predicate

    case components[0] of
        'clear': begin
            self.o3DDisplay->ClearEMVolume
            self.oXYZDisplay->ClearEMVolume
            self->DisableEMVolume
            return
        end
        'load': self->LoadEMVolume
        'opacity': begin
            self->VolumeOpacityEvent, /EM_VOLUME
        end
        'show/hide': begin
            hideEM = ~widget_info(self->Get('EM_SHOW_EM'), /BUTTON_SET)
            hideMol = ~widget_info(self->Get('EM_SHOW_MOL'), /BUTTON_SET)
            self.o3DDisplay->SetProperty, $
                EM_VOLUME_HIDE=hideEM, $
                HIDE_VOLUME=hideMol
            self.oXYZDisplay->SetProperty, $
                EM_HIDE=hideEM, $
                MOLECULE_HIDE=hideMol
            self.oXYZDisplay->RenderScene
            self.o3DDisplay->RenderScene
        end
        else: begin
            message, /CONTINUE, 'Unknown component: ' + components[0]
            return
        end
    endcase

end


;------------------------------------------------------------------------------
;+
; This method handles events from button widgets
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from a button on the PALM_MainGUI widget.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::ButtonEvent, event, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    widget_control, event.id, GET_UVALUE=uval
    if n_elements(uval) EQ 0 then $
        return
    components = strtok(uval, '_', /EXTRACT)
    case components[0] of
        '3dmouse': self -> VolMouseMenuEvent, event, components[1:*]
        'control': self -> ControlEvent, event, components[1:*]
        'menu': self -> MenuEvent, event, components[1:*]
        'XYZMouse': self -> XYZMouseMenuEvent, event, components[1:*]
        'toolbar': self -> ToolbarEvent, components[1:*]
        'volume' : self->VolumeExport, event
        'mpr'    : self->MPRMovie, components[1:*]
        'mm'     : self->MultMolButtonEvent, event
        'em'     : self->EMVolumeButtonEvent, event, components[1:*]
        else: message, /CONTINUE, 'Unknown component: ' + components[0]
    endcase

end


;------------------------------------------------------------------------------
;+
;  Make an MPR movie
;
; @param animationType {in}{type=string}{required}
;   A string naming the type of animation to generate.
;-
pro PALM_MainGUI::MPRMovie, animationType
    compile_opt idl2

    cd, CURRENT = savedPath
    catch, errorNumber
    if (errorNumber ne 0) then begin
        catch, /cancel
        cd, savedPath
        if (n_elements(oMJ2K) ne 0) then begin
            obj_destroy, oMJ2K
        endif
        if (n_elements(keep) ne 0 && ~keep && n_elements(nIter) ne 0) then begin
            self->UpdateInformationLabel, 'Cleaning up frames', /APPEND, /REMOVE_PERCENT
            for i=0LL, nIter-1 do begin
                file_delete, 'frame_'+string(i,FORMAT='(I04)')+'.png', /ALLOW_NONEXISTENT
            endfor
            if (n_elements(newFrames) ne 0) then Begin
                for i=0LL, newFrames-1 do begin
                    file_delete, 'saframe_'+string(i,FORMAT='(I04)')+'.png', /ALLOW_NONEXISTENT
                endfor
            endif
        endif
        self->UpdateInformationLabel, ' '
        message, /REISSUE_LAST
    endif
    if (animationType eq 'movie') then begin
        ;  Get the output file name
        file = dialog_pickfile(TITLE='Select an output file...', GET_PATH=path, $
            /OVERWRITE, PATH=self.LastFileDirectory, /WRITE)
        if (file eq '') then return
        self.LastFileDirectory = path
        cd, path

        fileType = strtok(file, '.', /EXTRACT, COUNT=nComponents)
        if (nComponents lt 2) then Begin
            void = dialog_message('Need a file name extension')
            cd, savedPath
            return
        endif
        fileType = strlowcase(fileType[nComponents - 1])
    endif

    ;  Get the plane, frame rate and whether to keep the frames or not
    if (widget_info(self->Get('MPR_X'), /BUTTON_SET)) then  plane = 0
    if (widget_info(self->Get('MPR_Y'), /BUTTON_SET)) then  plane = 1
    if (widget_info(self->Get('MPR_Z'), /BUTTON_SET)) then  plane = 2

    keep = widget_info(self->Get('MPR_KEEP_FRAMES'), /BUTTON_SET)
    n = widget_info(self->Get('MPR_FPS'), /DROPLIST_SELECT)
    fps = strtrim(([5,10,15,20,25,30])[n], 2)
    n = widget_info(self->Get('MPR_SCALE_FACTOR'), /DROPLIST_SELECT)
    widget_control, self->Get('MPR_SKIP_AND_AVERAGE'), GET_VALUE=skipAndAverage
    scale = ([1,2,3,4,5])[n]

    ;  Get hue settings
    UseHue = widget_info(self->Get('UseHueMPR'), /BUTTON_SET) &&  $
             widget_info(self->Get('UseHue'), /BUTTON_SET)

    ;  Do we animate on the screen at the same time?
    ScreenAnimation = widget_info(self->Get('MPR_ANIMATION'), /BUTTON_SET) || (animationtype eq 'animate')

    ;  Generate the frames
    oModel = self.o3DDisplay->GetObjectByName('PALM')
    if ~obj_valid(oModel) then  begin
        cd, savedPath
        return
    endif
    oModel->GetProperty, VOLUME_POINTER=vp
    if ~ptr_valid(vp) then  $
        return
    vol = *vp
    mvol = max(vol)
    dims = size(vol, /DIM)

    ;  Byte scale range
    self.oXYZDisplay->GetProperty, BS_MAX=bsMax, BS_MIN=bsMin

    ;  And the palette
    self.oXYZDisplay->GetProperty, COLOR_TABLE=rgb
    red = reform(rgb[*,0])
    green = reform(rgb[*,1])
    blue = reform(rgb[*,2])

    if (animationType eq 'movie') then Begin
        ; Write to a temp file, then rename it to override problems ffmpeg has with network drive paths.
        filebase = file_basename(file)
        tempfile = filepath(filebase, /TMP)
        if (file_test(tempfile)) then begin
            file_delete, tempfile
        endif

        ; For the purposes of testing, I look in the source path directory for the ffmpeg executable first.
        ffmpegexecutable = filepath('ffmpeg*', ROOT = sourcepath())
        mpegexecutable = file_search(ffmpegexecutable, COUNT=count)

        ; If I don't see the executable here, then I assume it's in PATH.
        ffmpeg = count eq 1 ? $
            '"' + mpegexecutable[0] + '"': $
            'ffmpeg'
    endif

    nIter = dims[plane]
    count = (dims[plane]/50)>1
    case plane of
        0 : begin
            xindex = 1
            yindex = 2
        end
        1 : begin
            xindex = 0
            yindex = 2
        end
        2 : begin
            xindex = 0
            yindex = 1
            end
        else :
    endcase
    dimsx = dims[xindex]*scale
    dimsy = dims[yindex]*scale
    if (UseHue) then begin
        rgb = bytarr(3, dimsx, dimsy, /NOZERO)
    endif
    s = replicate(1., dimsx, dimsy)
    if (UseHue) then begin
        switch plane of
            0 :
            1 : begin
                h = replicate(1., dimsx) # 320.0*(1.+findgen(dimsy))/dimsy   ;##HFH##
                break
                end
            2 : begin
                break
                end
            else :
        endswitch
    endif
    self->UpdateInformationLabel, 'Animating '

    oStatusBar = obj_new('PALM_StatusBar', $
        COLOR=[0,111,0], $
        TEXT_COLOR=[255,255,255], $
        TITLE='Generating frames...', $
        TOP_LEVEL_BASE=tlb)

    for i = 0L, nIter - 1 do begin

        event = widget_event(tlb, /NOWAIT)
        if tag_names(event, /STRUCT) NE 'WIDGET_NOEVENT' then begin
            self->ShowPlane, plane, i, /UPDATE_3D
            self->UpdateInformationLabel, ' '
            if ~keep then begin
                for i=0LL, nIter-1 do begin
                    file_delete, 'frame_'+string(i,FORMAT='(I04)')+'.png', /ALLOW_NONEXISTENT
                endfor
            endif
            obj_destroy, oStatusBar
            return
        endif

        oStatusBar->UpdateStatus, float(i)/nIter
        if (i mod count eq 0) then begin
            self->UpdateInformationLabel, /APPEND, /REMOVE_PERCENT, $
                '. <'+strtrim(100*i/nIter,2)+'%>'
        endif
        if (ScreenAnimation) then begin
            self->ShowPlane, plane, i
            if (animationType eq 'animate') then begin
                continue
            endif
        endif
        case plane of
            0 : image = vol[i, *, *]
            1 : image = vol[*, i, *]
            2 : image = vol[*, *, i]
            else :
        endcase

        image = reform(256*temporary(image)<255)                 ;unity = full byte scaling ##HFH
        image = congrid(temporary(image), dimsx, dimsy, /INTERP)
        if (UseHue) then begin
            v = image/255. <1.0     ;(image*float(mvol)/255.)<1.0   ;same as original vol data ##HFH
            case plane of                              ;##HFH##  use case not switch
                0 : begin
                    hh = (320.0*(1.+findgen(dims[2])/dims[2]))        ;##HFH
                    h = replicate(1.0,dims[1])#hh             ;##HFH
                    end
                1 : begin
                    hh = (320.0*(1.+findgen(dims[2])/dims[2]))        ;##HFH
                    h = replicate(1.0,dims[0])#hh             ;##HFH
                    end
                2 : begin
                    hh = (320.0*(1.+findgen(dims[2])/dims[2]))[i]     ;##HFH##  remove reverse
                    h = replicate(hh, dimsx, dimsy)
                    end
            endcase                                   ;##HFH

            color_convert, h,s,v, r,g,b, /HSV_RGB
            rgb[0,0,0] = reform(temporary(r), 1, dimsx, dimsy)
            rgb[1,0,0] = reform(temporary(g), 1, dimsx, dimsy)
            rgb[2,0,0] = reform(temporary(b), 1, dimsx, dimsy)
        endif
        if (animationType eq 'movie') then Begin
            if (fileType eq 'mj2' && skipAndAverage eq 0) then begin
                if (i eq 0) then begin
                    ; Create the MJ2K file object on the first frame.
                    tickspersecond = 30000.
                    oMJ2K = obj_new('idlffmjpeg2000', file, /WRITE, /REVERSIBLE, $
                        FRAME_PERIOD = tickspersecond/fps, TIMESCALE = tickspersecond)
                endif
                if (~UseHue) then begin
                    newimage = bytarr(3, dimsx, dimsy)
                    for j = 0L, 2 do begin
                        newimage[j, 0, 0] = reform((reform(rgb[*, j]))[image], 1, dimsx, dimsy)
                    endfor
                    R = oMJ2K->SetData(temporary(newImage))
                endif else begin
                    R = oMJ2K->SetData(rgb)
                endelse
            endif else begin
                if (UseHue) then begin
                    write_png, 'frame_'+string(i,FORMAT='(I04)')+'.png', rgb
                endif else begin
                    write_png, 'frame_'+string(i,FORMAT='(I04)')+'.png', image, red, green, blue
                endelse
            endelse
        endif
    endfor

    self->ShowPlane, plane, nIter-1, /UPDATE_3D

    if (obj_valid(oStatusBar)) then  $
        obj_destroy, oStatusBar

    self->UpdateInformationLabel, ' '
    if (animationType ne 'movie' && screenAnimation) then begin
        return
    endif
    if (animationType eq 'movie' && skipAndAverage ne 0) then begin
        self->UpdateInformationLabel, 'Averaging frames '
        frameBase = 'saframe'
        newFrames = 0L
        av = skipAndAverage + 1
        for i = 0l, nIter - skipAndAverage + 1, skipAndAverage do begin
            if (fileType eq 'mj2' && i eq 0) then begin
                ; Create the MJ2K file object on the first frame.
                tickspersecond = 30000.
                oMJ2K = obj_new('idlffmjpeg2000', file, /WRITE, /REVERSIBLE, $
                    FRAME_PERIOD = tickspersecond/fps, TIMESCALE = tickspersecond)
            endif
            c = 0L
            for a = 0L, av - 1 do begin
                if (a + i lt nIter - 1) then begin
                    n = strtrim(string(a + i, FORMAT = '(I04)'), 2)
                    frame = 'frame_' + n + '.png'
                    image = read_png('frame_' + n + '.png', rr, gg, bb)
                    if (size(image, /N_DIMENSIONS) ne 3) then begin
                        imageSize = size(image, /DIMENSIONS)
                        dx = imageSize[0]
                        dy = imageSize[1]
                        newImage = bytarr(3, dx, dy)
                        newImage[0, 0, 0] = reform(rr[image], 1, dx, dy)
                        newImage[1, 0, 0] = reform(gg[image], 1, dx, dy)
                        newImage[2, 0, 0] = reform(bb[image], 1, dx, dy)
                        image = temporary(newimage)
                    endif
                    s = a eq 0 ? float(image) : temporary(s) + image
                    c++
                endif
            endfor
            if (n_elements(s) ne 0) then begin
                if (i mod count eq 0) then begin
                    self->UpdateInformationLabel, /APPEND, /REMOVE_PERCENT, $
                        '. <'+strtrim(100*i/nIter,2)+'%>'
                endif

                s = byte(temporary(s)<255)
                if (fileType eq 'mj2') then begin
                    R = oMJ2K->SetData(s)
                endif else begin
                    write_png, frameBase + '_' + string(newframes, FORMAT='(I04)') + '.png', s
                endelse
            endif
            newframes++
        endfor
    endif else begin
        frameBase = 'frame'
    endelse

    switch fileType of
        'mj2' : begin
            status = oMJ2K->Commit(10000L)
            if (status eq 0) then begin
                message, 'Error writing MJPEG2000 file to disk.', /TRACEBACK
            endif
            obj_destroy, oMJ2K
            ;  Clean up the frames
            if (~keep) then begin
                self->UpdateInformationLabel, 'Cleaning up frames', /APPEND, /REMOVE_PERCENT
                for i=0LL, nIter-1 do begin
                    file_delete, 'frame_'+string(i,FORMAT='(I04)')+'.png', /ALLOW_NONEXISTENT
                endfor
            endif
            cd, savedPath
            self -> UpdateInformationLabel, ' '
            return
            break
         end
         else : begin
            break
         end
    endswitch

    if (animationType eq 'movie') then begin
        ;  Get the file extension and make the movie
        t = strsplit(file, '.', /EXTRACT)
        ext = t[n_elements(t)-1]
        if (ext eq 'mpg') then begin
            n = fix(fps)
            if (n lt 20) then begin
                fps = '20'
            endif else begin
                fps = '30'
            endelse
            command = ffmpeg + ' -qscale 5 -y -r ' + fps + ' -i ' + frameBase + '_%04d.png ' + '"' + tempfile + '"'
        endif else begin
            command = ffmpeg + ' -qscale 5 -y -r ' + fps + ' -b 3600 -i ' + frameBase + '_%04d.png ' + '"' + tempfile + '"'
        endelse
        if (!version.os_family eq 'Windows') then begin
            spawn, command, output, error, /HIDE, /NOSHELL
        endif else begin
            spawn, command, output, error
        endelse

        ;  Clean up the frames
        if (~keep) then begin
            self->UpdateInformationLabel, 'Cleaning up frames', /APPEND, /REMOVE_PERCENT
            for i=0LL, nIter-1 do begin
                file_delete, 'frame_'+string(i,FORMAT='(I04)')+'.png', /ALLOW_NONEXISTENT
            endfor
            if (n_elements(newFrames) ne 0) then Begin
                for i=0LL, newFrames-1 do begin
                    file_delete, 'saframe_'+string(i,FORMAT='(I04)')+'.png', /ALLOW_NONEXISTENT
                endfor
            endif
        endif
        if (file_test(tempfile)) then begin

            ;  We use a copy then a delete since FILE_MOVE doesn't let us move across filesystems.
            ;  This may be an issue if we're trying to move from a local drive to a networked drive, for example.
            file_copy, tempfile, file, /OVERWRITE, /ALLOW_SAME
            file_delete, tempfile
        endif else begin
            if (n_elements(error) gt 0 && error[0] ne '') then begin
                v = dialog_message(error, /error)
            endif else begin
                v = dialog_message('An unknown error prevented the movie file from being created.', /error)
            endelse
        endelse
    endif
    self->UpdateInformationLabel, ' '
    cd, savedPath
end


;------------------------------------------------------------------------------
;+
;  Export the volume
;
; @param event {in}{type=structure}{required}
;   The button event structure
;-
pro PALM_MainGUI::VolumeExport, event
    compile_opt idl2

    ;  Get the volume
    oModel = self.o3DDisplay->GetObjectByName('PALM')
    if ~obj_valid(oModel) then  $
        return
    oModel->GetProperty, VOLUME_POINTER=vp
    if ~ptr_valid(vp) then  $
        return

    ;  Get the type
    type = widget_info(self->Get('VOLUME_EXPORT_TYPE'), /COMBOBOX_GETTEXT)

    ;  Get the output file
    file = dialog_pickfile(TITLE='Select an output file...', GET_PATH=path, $
        /OVERWRITE, PATH=self.LastFileDirectory)
    if (file eq '') then  $
        return
    self.LastFileDirectory = path
    cd, path
    widget_control, /HOURGLASS

    ;  Get the actual volume data
    vol = *vp

    ;  Incorporate the volume dimensions into the file name
    dims = size(vol, /DIM)
    ds = '_'+strtrim(dims[0],2)+'_'+strtrim(dims[1],2)+'_'+strtrim(dims[2],2)+'.dat'
    fname = file + ds

    ;  Output according to type
    case (type) of
        'IDL .sav': begin
            save, vol, FILE=file
        end
        'Float (little)': begin
            swap_endian_inplace, vol, /SWAP_IF_BIG_ENDIAN
            openw, u, fname, /GET_LUN
            writeu, u, vol
            free_lun, u
        end
        'Float (big)': begin
            swap_endian_inplace, vol, /SWAP_IF_LITTLE_ENDIAN
            openw, u, fname, /GET_LUN
            writeu, u, vol
            free_lun, u
        end
        'Byte Scaled': begin
            vol = byte(256*vol) <255       ;bytscl(temporary(vol))  ##HFH keep 256 x unity scale
            openw, u, fname, /GET_LUN
            writeu, u, vol
            free_lun, u
        end
        else:  message, 'Unknown volume export type: ' + type
    endcase
end


;------------------------------------------------------------------------------
;+
; This lifecycle method is called during the destruction of the PALM_MainGUI
; object via OBJ_DESTROY.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::Cleanup
    self -> Destruct
end


;------------------------------------------------------------------------------
;+
; This method starts the PALM color table selection dialog
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::ColorTableEvent

    oColorTableDialog = obj_new('PALM_ColorTableDialog', $
        self.tlb, $
        self.o3DDisplay, $
        self.oXYZDisplay, $
        INVERT_TABLE=widget_info(self->Get('menu_view_InvertColorTable'), /BUTTON_SET), $
        VERBOSE=self.verbose)
    obj_destroy, oColorTableDialog

end


;------------------------------------------------------------------------------
;+
; This method builds the context menu for the XYZ window
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_MainGUI::ConstructContextMenus, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
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

    ; Construct the context menu for the XYZ-Displays
    if (obj_valid(self.oXYZWindowObserver)) then $
        self.oXYZWindowObserver->GetProperty, CURSOR_MODE=XYZCursorMode $
    else $
        XYZCursorMode = 'Scroll'
    strXYZ = ['X','Y','Z']
    uNames = 'PALM_' + strXYZ + 'Window'
    for i = 0, n_elements(uNames)-1 do begin
        wBase = widget_base(widget_info(self.tlb, FIND_BY_UNAME=uNames[i]), $
            /CONTEXT_MENU, $
            UNAME = uNames[i]+'ContextBase')
        wButton = widget_button(wBase, $
            /CHECKED_MENU, $
            UNAME = 'XYZMousePanModeButton', $
            UVALUE = 'XYZMouse_Pan', $
            VALUE='Pan')
        widget_control, wButton, $
            SET_BUTTON=strcmp(XYZCursorMode, 'Pan', /FOLD_CASE)
;        wButton = widget_button(wBase, $
;            /CHECKED_MENU, $
;            UNAME = uNames[i]+'WindowLevel', $ ;'XYZMouseWindowLevelModeButton', $
;            UVALUE = 'XYZMouse_WindowLevel', $
;            VALUE='Window Level')
;        widget_control, wButton, $
;            SET_BUTTON=strcmp(XYZCursorMode, 'WindowLevel', /FOLD_CASE)
        wButton = widget_button(wBase, $
            /CHECKED_MENU, $
            UNAME = uNames[i]+'SliceScroll', $ ;'XYZSliceScrollModeButton', $
            UVALUE = 'XYZMouse_SliceScroll', $
            VALUE='Scroll Through Slices')
        widget_control, wButton, $
            SET_BUTTON=strcmp(XYZCursorMode, 'Scroll', /FOLD_CASE)
        wButton = widget_button(wBase, $
            /CHECKED_MENU, $
            UNAME = 'XYZMouseZoomModeButton', $
            UVALUE = 'XYZMouse_Zoom', $
            VALUE='Zoom')
        widget_control, wButton, $
            SET_BUTTON=strcmp(XYZCursorMode, 'Zoom', /FOLD_CASE)
        wExport = widget_button(wBase, $
            /MENU, $
            /SEPARATOR, $
            UVALUE='XYZMouse_Export', $
            VALUE='Export As')
        wImage = widget_button(wExport, $
            /MENU, $
            VALUE='Image')
        wButton = widget_button(wImage, $
            UVALUE='XYZMouse_Image_BMP', $
            VALUE='BMP')
        wButton = widget_button(wImage, $
            UVALUE='XYZMouse_Image_TIFF', $
            VALUE='TIFF')
        wMotion = widget_button(wExport, $
            /MENU, $
            VALUE='Motion')
        wButton = widget_button(wMotion, $
            UVALUE='XYZMouse_Motion_MPEG', $
            VALUE='MPEG')
        wReset = widget_button(wBase, $
            /MENU, $
            UVALUE='XYZMouse_Reset_'+strXYZ[i], $
            VALUE='Reset')
        wButton = widget_button(wReset, $
            UVALUE='XYZMouse_Reset_XYZ_'+strXYZ[i], $
            VALUE='Scale/Translation')
        wButton = widget_button(wReset, $
            UVALUE='XYZMouse_Reset_XYZ_windowlevel', $
            VALUE='Window Level (All XYZ-Windows)')
    endfor

    ; Construct the context menu for the 3D display
    wBase = widget_base(widget_info(self.tlb, FIND_BY_UNAME='PALM_3DWindow'), $
        /CONTEXT_MENU, $
        UNAME='PALM_3DWindowContextBase')
    wButton = widget_button(wBase, $
        /CHECKED_MENU, $
        UNAME='3dmouse_select', $
        UVALUE='3dmouse_select', $
        VALUE='Select')
     widget_control, wButton, /SET_BUTTON
     wButton = widget_button(wBase, $
        /CHECKED_MENU, $
        UNAME='3dmouse_rotate', $
        UVALUE='3dmouse_rotate', $
        VALUE='Rotate')
    wButton = widget_button(wBase, $
        /CHECKED_MENU, $
        UNAME='3dmouse_pan', $
        UVALUE='3dmouse_pan', $
        VALUE='Pan')
    wButton = widget_button(wBase, $
        /CHECKED_MENU, $
        UNAME='3dmouse_zoom', $
        UVALUE='3dmouse_zoom', $
        VALUE='Zoom')
    wExport = widget_button(wBase, $
        /MENU, $
        /SEPARATOR, $
        UVALUE='3dmouse_Export', $
        VALUE='Export As')
    wImage = widget_button(wExport, $
        /MENU, $
        UVALUE='3dmouse_Image', $
        VALUE='Image')
    wButton = widget_button(wImage, $
        UVALUE='3dmouse_Image_BMP', $
        VALUE='BMP')
    wButton = widget_button(wImage, $
        UVALUE='3dmouse_Image_TIFF', $
        VALUE='TIFF')
    wMotion = widget_button(wExport, $
        /MENU, $
        UVALUE='3dmouse_Motion', $
        VALUE='Motion')
    wButton = widget_button(wMotion, $
        UVALUE='3dmouse_Motion_MPEG', $
        VALUE='MPEG')
    wButton = widget_button(wbase, $
        UVALUE='3dmouse_recalculate', $
        VALUE='Recalculate Volume')
    wButton = widget_button(wBase, $
        UVALUE='3dmouse_Reset', $
        VALUE='Reset')
    wSlice = widget_button(wBase, $
        /MENU, $
        UVALUE='3dmouse_Slice', $
        VALUE='Slice Planes')
    wButton = widget_button(wSlice, $
        /CHECKED_MENU, $
        UVALUE='3dmouse_SlicePlanes_none', $
        VALUE='None')
    wButton = widget_button(wSlice, $
        /CHECKED_MENU, $
        UVALUE='3dmouse_SlicePlanes_wire', $
        VALUE='Wire Frame')
    wButton = widget_button(wSlice, $
        /CHECKED_MENU, $
        UVALUE='3dmouse_SlicePlanes_full', $
        VALUE='Full')

    ; Volume opacity
    wButton = widget_button(wBase, $
        UVALUE='3dmouse_VolumeOpacity', $
        VALUE='Volume opacity...')
    widget_control, wButton, /SET_BUTTON
    return, 1

end


;------------------------------------------------------------------------------
;+
; This method builds the controls for the application
;
; @Param
;   wParent {in}{type=long}{required}
;     The widget ID of the base widget on which to place the control base
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @keyword COMPOSITE_FUNCTION {in}{type=number}{required}
;   Volume object composite value
;
; @keyword DATA_RANGE {in}{type=number}{required}
;   Data range (nm)
;
; @keyword MAXIMUM_SCALE {in}{type=number}{required}
;   Unused
;
; @keyword MAXIMUM_VOLUME_DIMENSION {in}{type=number}{required}
;   Largest dimension along any one axis
;
; @keyword SUBVOLUME_WIDTH {in}{type=number}{required}
;   Subvolume over which Gaussian calculated (nm)
;
; @keyword Z_SCALE_FACTOR {in}{type=number}{required}
;   Scale factor applied to Z axis
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_MainGUI::ConstructControlBase, wParent, $
    COMPOSITE_FUNCTION=CompositeFunction, $
    DATA_RANGE=DataRange, $
    ERROR_MESSAGE=ErrMsg, $
    MAXIMUM_SCALE=MaxScale, $
    MAXIMUM_VOLUME_DIMENSION=MaxVolumeDimension, $
    SUBVOLUME_WIDTH=SubVolumeWidth, $
    Z_SCALE_FACTOR=zScaleFactor

    compile_opt idl2
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

    wMainBase = widget_base(wParent, $
        UNAME='ControlBase', $
        XSIZE=strupcase(!version.OS_Family) EQ 'UNIX' ? 370 : 320)
    wControlBase = widget_base(wMainBase, $
        /COL, $
        YOFFSET=(widget_info(self->Get('toolbar_base'),/GEOMETRY)).scr_ySize)
    wTab = widget_tab(wControlBase)
    wControlBase = widget_base(wTab, TITLE=' Page 1 ', /COL, XPAD=0)
    wPage2 = widget_base(wTab, TITLE=' Page 2 ', /COL, XPAD=0)
    wPage3 = widget_base(wTab, TITLE=' Page 3 ', /COL, XPAD=0)
; Volume controls
    ySize = 20
    wBase = widget_base(wControlBase, $
        /COLUMN)
    wSliderBase = widget_base(wBase, $
        /ROW, $
        YSIZE=ySize)
    wLabel = widget_label(wSliderBase, VALUE='Increment:')
    wSlider = widget_slider(wSliderBase, $
        /DRAG, $
        MAXIMUM=100, $
        MINIMUM=1, $
        /SUPPRESS_VALUE, $
        UVALUE='control_increment', $
        VALUE=10, $
        YSIZE=ySize)
    wLabel = widget_label(wSliderBase, $
        /SUNKEN_FRAME, $
        UNAME='incrementLabel', $
        VALUE=' 1.0', $
        XSIZE=25)

    wRotateAndShearBase = widget_base(wBase, /Row)
    wRotateOnlyBase = widget_base(wRotateAndShearBase, /COLUMN)
    wRotateBase = widget_base(wRotateOnlyBase, $
        /ROW, $
        YSIZE=ySize)
    wLabel = widget_label(wRotateBase, $
        VALUE='Rotate (X):', $
        YSIZE=ySize)
    wButton = widget_button(wRotateBase, $
        UVALUE='control_rotate_x-', $
        VALUE=' - ', $
        YSIZE=ySize)
    wButton = widget_button(wRotateBase, $
        UVALUE='control_rotate_x+', $
        VALUE=' + ', $
        YSIZE=ySize)
    wButton = widget_button(wRotateBase, $
        UVALUE='control_rotate_x0', $
        VALUE='0', $
        YSIZE=ySize)
    wRotateBase = widget_base(wRotateOnlyBase, $
        /ROW, $
        YSIZE=ySize)
    wLabel = widget_label(wRotateBase, $
        VALUE='Rotate (Y):', $
        YSIZE=ySize)
    wButton = widget_button(wRotateBase, $
        UVALUE='control_rotate_y-', $
        VALUE=' - ', $
        YSIZE=ySize)
    wButton = widget_button(wRotateBase, $
        UVALUE='control_rotate_y+', $
        VALUE=' + ', $
        YSIZE=ySize)
    wButton = widget_button(wRotateBase, $
        UVALUE='control_rotate_y0', $
        VALUE='0', $
        YSIZE=ySize)
    wRotateBase = widget_base(wRotateOnlyBase, $
        /ROW, $
        YSIZE=ySize)
    wLabel = widget_label(wRotateBase, $
        VALUE='Rotate (Z):', $
        YSIZE=ySize)
    wButton = widget_button(wRotateBase, $
        UVALUE='control_rotate_z-', $
        VALUE=' - ', $
        YSIZE=ySize)
    wButton = widget_button(wRotateBase, $
        UVALUE='control_rotate_z+', $
        VALUE=' + ', $
        YSIZE=ySize)
    wButton = widget_button(wRotateBase, $
        UVALUE='control_rotate_z0', $
        VALUE='0', $
        YSIZE=ySize)
    wShearOnlyBase = widget_base(wRotateAndShearBase, /COLUMN)
    wShearBase = widget_base(wShearOnlyBase, $
        /ROW, $
        YSIZE=ySize)
    wLabel = widget_label(wShearBase, $
        VALUE='Shear (X):', $
        YSIZE=ySize)
    wButton = widget_button(wShearBase, $
        UVALUE='control_shear_x-', $
        VALUE=' - ', $
        YSIZE=ySize)
    wButton = widget_button(wShearBase, $
        UVALUE='control_shear_x+', $
        VALUE=' + ', $
        YSIZE=ySize)
    wButton = widget_button(wShearBase, $
        UVALUE='control_shear_x0', $
        VALUE='0', $
        YSIZE=ySize)
    wShearBase = widget_base(wShearOnlyBase, $
        /ROW, $
        YSIZE=ySize)
    wLabel = widget_label(wShearBase, $
        VALUE='Shear (Y):', $
        YSIZE=ySize)
    wButton = widget_button(wShearBase, $
        UVALUE='control_shear_y-', $
        VALUE=' - ', $
        YSIZE=ySize)
    wButton = widget_button(wShearBase, $
        UVALUE='control_shear_y+', $
        VALUE=' + ', $
        YSIZE=ySize)
    wButton = widget_button(wShearBase, $
        UVALUE='control_shear_y0', $
        VALUE='0', $
        YSIZE=ySize)
    wShearBase = widget_base(wShearOnlyBase, $
        /ROW, $
        YSIZE=ySize)
    wLabel = widget_label(wShearBase, $
        VALUE='Shear (Z):', $
        YSIZE=ySize)
    wButton = widget_button(wShearBase, $
        UVALUE='control_shear_z-', $
        VALUE=' - ', $
        YSIZE=ySize)
    wButton = widget_button(wShearBase, $
        UVALUE='control_shear_z+', $
        VALUE=' + ', $
        YSIZE=ySize)
    wButton = widget_button(wShearBase, $
        UVALUE='control_shear_z0', $
        VALUE='0', $
        YSIZE=ySize)
; Volume settings
    void = widget_droplist(wBase, TITLE='Z parameter to plot: ',   $
                  VALUE=['Group Z', 'Z', 'Unwrapped Group Z', 'Unwrapped Z'],  $
                  UNAME='Z_PARAM', /ALIGN_RIGHT)
    wDroplist = widget_droplist(wBase, $
        /ALIGN_RIGHT, $
        TITLE='Accumulation:', $
        UNAME='Accumulation', $
        VALUE=['Envelope','Sum'])
    widget_control, wDropList, SET_DROPLIST_SELECT=1
    wDroplist = widget_droplist(wBase, $
        /ALIGN_RIGHT, $
        TITLE='Volume Type', $
        UNAME='CompositeFunction', $
        VALUE=['Standard','MIP','Alpha','Average'])
    if n_elements(CompositeFunction) GT 0 then $
        widget_control, wDroplist, SET_DROPLIST_SELECT=CompositeFunction

    ;  Hue volume
    base = widget_base(wBase, /ALIGN_RIGHT, /NONEXCLUSIVE)
    button = widget_button(base, VALUE='Use hue for Z position', UNAME='UseHue')

    ;  Hue in MPR
    base = widget_base(wBase, /ALIGN_RIGHT, /NONEXCLUSIVE)
    button = widget_button(base, VALUE='Use hue for XYZ display', $
        UNAME='UseHueMPR', $
        UVALUE='control_UseHueMPR')

    ;  Calculate EDM in volume generation
    base = widget_base(wBase, /ALIGN_RIGHT, /NONEXCLUSIVE)
    button = widget_button(base, VALUE='Calculate Euclidean distance map', UNAME='UseEDM')
    widget_control, button, /SET_BUTTON

    ;  Use constant intensity scaling when zooming
    base = widget_base(wBase, /ALIGN_RIGHT, /NONEXCLUSIVE)
    button = widget_button(base, VALUE='Use constant intensity scaling', UNAME='UseConstant')
    widget_control, button, /SET_BUTTON

    ;  Gamma slider for volume
    base = widget_base(wBase, /ALIGN_RIGHT, /ROW, YPAD=5)
    void = widget_label(base, VALUE='Volume gamma:', /ALIGN_RIGHT)
    void = widget_slider(base, VALUE=70, MIN=10, MAX=200, /SUPPRESS, /DRAG, $
                         UVALUE='gamma', UNAME='GammaRender')
    void = widget_label(base, VALUE=' 0.7', /SUNK, XSIZE=40, UNAME='GammaRenderValue')

    ;  Brightness slider
    base = widget_base(wBase, /ALIGN_RIGHT, /ROW, YPAD=5)
    void = widget_label(base, VALUE='Brightness:', /ALIGN_RIGHT)
    void = widget_slider(base, VALUE=10, MIN=1, MAX=100, /SUPPRESS, /DRAG, $
                         UVALUE='brightness', UNAME='Brightness')
    void = widget_label(base, VALUE=' 1.0', /SUNK, XSIZE=40, UNAME='BrightnessValue')

    sWidths = ['40','60','80','100','120','140','160','320']
    wDroplist = widget_droplist(wBase, $
        /ALIGN_RIGHT, $
        TITLE='Subvolume Width (nanometers)', $
        UNAME='SubvolumeWidth', $
        VALUE=sWidths)
    if n_elements(SubVolumeWidth) GT 0 then begin
        index = where(sWidths EQ strtrim(SubVolumeWidth[0],2))
        widget_control, wDroplist, SET_DROPLIST_SELECT=index[0]
    endif
    wSliderBase = widget_base(wBase, $
        /ALIGN_RIGHT, $
        /ROW)
    wLabel = widget_label(wSliderbase, $
        VALUE='Volume Dimension:')
    MaxVolDimension = n_elements(MaxVolumeDimension) GT 0 ? $
        (fix(MaxVolumeDimension[0])>100)<2000 : 400
    wSlider = widget_slider(wSliderBase, $
        /DRAG, $
        MAXIMUM=2000, $
        MINIMUM=100, $
        /SUPPRESS_VALUE, $
        UVALUE='control_VolumeDimensions', $
        VALUE=MaxVolDimension, $
        XSIZE=100)
    wLabel = widget_label(wSliderBase, $
        /SUNKEN_FRAME, $
        UNAME='VolumeDimensionsLabel', $
        VALUE=strtrim(maxVolDimension,2), $
        XSIZE=25)
    wSliderBase = widget_base(wBase, $
        /ALIGN_RIGHT, $
        /ROW)
    wLabel = widget_label(wSliderBase, $
        VALUE='Z-Scale Factor:')
    zScale = n_elements(zScaleFactor) GT 0 ? $
        (fix(zScaleFactor[0])>1)<100 : 1
    wSlider = widget_slider(wSliderBase, $
        /DRAG, $
        MAXIMUM=100, $
        MINIMUM=1, $
        /SUPPRESS_VALUE, $
        UVALUE='control_zScaleFactor', $
        VALUE=zScale*10, $
        XSIZE=100)
    wLabel = widget_label(wSliderBase, $
        /SUNKEN_FRAME, $
        UNAME='zScaleFactorLabel', $
        VALUE=string(zScale, FORMAT='(f4.1)'), $
        XSIZE=30)
    wButtonBase = widget_base(wBase, $
        /ALIGN_LEFT, $
        /ROW)
    wButton = widget_button(wButtonBase, $
        UVALUE='control_Recalculate', $
        VALUE='Recalculate Volume')
    void = widget_label(wButtonBase, VALUE='  Dimensions: ')
    void = widget_label(wButtonBase, VALUE='0x0x0', XSIZE=80, /SUNK, UNAME='volume_dims')

; For MPEG output
    wMPEGBase = widget_base(wControlBase, $
        /COL, FRAME=1)
    wLabel = widget_label(wMPEGBase, $
        VALUE='Volume Movies')

    ; Choose between rotation and shear movies; use a bulletin board
    wAnimationTypeBase = widget_base(wMPEGBase, /Row)
    wAnimationTypeLabel = widget_label(wAnimationTypeBase, $
        VALUE='Animation type: ', /ALIGN_LEFT)
    wAnimationType = widget_combobox(wAnimationTypeBase, $
        VALUE=['Flythrough', 'Rotation', 'Shear'], $
        UVALUE='control_animationtype', $
        UNAME='AnimationType')
    widget_control, wAnimationType, SET_COMBOBOX_SELECT=1
    wbbBase = widget_Base(wMPEGBase)
    wFlythroughBase = widget_base(wbbBase, /COLUMN, UNAME='AnimateFlythroughBase')
    wRotationBase = widget_Base(wbbBase, /COLUMN, UNAME='AnimateRotationBase')
    wShearBase = Widget_Base(wbbBase, /COLUMN, UNAME='AnimateShearBase')
    widget_control, wFlythroughBase, MAP=0
    widget_control, wshearbase, map = 0
; Rotation
    wBase = widget_base(wRotationBase, $
        /ROW);, $

    wLabel = widget_label(wRotationBase, $
        VALUE='Rotation Axis: ')
    wExBase = widget_base(wRotationBase, $
        /EXCLUSIVE, $
        /ROW, $
        YSIZE=ySize+1)
    strAxis = ['X','Y','Z']
    for i = 0, n_elements(strAxis)-1 do begin
        wButton = widget_button(wExBase, $
            UNAME='Rotate'+strAxis[i], $
            UVALUE='control_rotate_'+strAxis[i], $
            VALUE=strAxis[i])
        if i EQ 0 then $
            widget_control, wButton, /SET_BUTTON
    endfor
    wSliderBase = widget_base(wRotationBase, $
        /ALIGN_RIGHT, $
        /ROW, $
        YSIZE=ySize)
    wLabel = widget_label(wSliderBase, $
        VALUE='Rotation Angle:')
    wSlider = widget_slider(wSliderBase, $
        /DRAG, $
        MAXIMUM=360, $
        MINIMUM=1, $
        /SUPPRESS_VALUE, $
        UVALUE='control_RotationAngle', $
        VALUE=45, $
        XSIZE=100)
    wLabel = widget_label(wSliderBase, $
        /SUNKEN_FRAME, $
        UNAME='RotationAngle', $
        VALUE='30', $
        XSIZE=25)
;
; Shear
;
    wModeBase = widget_base(wShearBase, /ALIGN_RIGHT, /ROW)
    wLabel = widget_label(wModeBase, VALUE='Shear Mode')
    wDrop = widget_droplist(wModeBase, VALUE=['X','Y','Circle'], $
        UNAME='ShearMode')
    wSliderBase = widget_base(wShearBase, $
        /ALIGN_RIGHT, $
        /ROW, $
        YSIZE=ySize)
    wLabel = widget_label(wSliderBase, $
        VALUE='Shear Scale:')
    wSlider = widget_slider(wSliderBase, $
        /DRAG, $
        MAXIMUM=20, $
        MINIMUM=0, $
        /SUPPRESS_VALUE, $
        UVALUE='control_ShearScale', $
        VALUE=10, $
        XSIZE=100)
    wLabel = widget_label(wSliderBase, $
        /SUNKEN_FRAME, $
        UNAME='ShearScaleLabel', $
        VALUE='1', $
        XSIZE=25)
;
; Flythrough
;
    wFileBase = widget_base(wFlythroughBase, $
        /ALIGN_RIGHT, $
        /ROW)
    wLabel = widget_label(wFileBase, $
        VALUE='Flight Path File:')
    wText = widget_text(wFileBase, $
        /EDITABLE, $
        UNAME='FlythroughFile', $
        VALUE=' ')
    wButton = widget_button(wFileBase, $
        UVALUE='control_FlythroughBrowse', $
        VALUE='Browse')

    wBase = widget_base(wMPEGBase, $
        /ROW)
    wButton = widget_button(wBase, $
        UVALUE='control_animate', $
        VALUE='Animate')
    wButton = widget_button(wBase, $
        UVALUE='control_mpeg', $
        VALUE='Make Movie')
    wButton = widget_button(wBase, UVALUE='control_playback', $
        VALUE='Play Movie')
    wBase = widget_base(wMPEGBase, $
        /NONEXCLUSIVE, $
        /ROW)
    wButton = widget_button(wBase, $
        VALUE='Animate while creating movie', $
        UNAME='animate')

;  ffmpeg output options
    base = widget_base(wMPEGBase, /ALIGN_LEFT, /COLUMN)
    abase = widget_base(base, /ALIGN_LEFT, /EXCLUSIVE)
    v = widget_button(abase, VALUE='Create MPEG output in IDL', UNAME='UseMPEG')
    void = widget_button(abase, VALUE='Use ffmpeg for movie output', UNAME='UseFFMPEG')
    widget_control, void, /SET_BUTTON
    abase = widget_base(base, /ALIGN_LEFT, /NONEXCLUSIVE)
    void = widget_button(abase, VALUE='Do not delete frame files', UNAME='FFMPEG_KEEP_FRAMES')
    void = widget_droplist(base, TITLE='FPS:  ', UNAME='FFMPEG_FPS',  $
                           VALUE=['5','10','15','20','25','30'])
    if (!version.os_family eq 'Windows') then begin
        void = widget_label(base, VALUE=' ')
        void = widget_label(base, VALUE='File extension determines movie type:   ', /ALIGN_LEFT)
        void = widget_label(base, VALUE='     .mj2   - Motion JPEG 2000', /ALIGN_LEFT)
        void = widget_label(base, VALUE='     .mpg   - MPEG-1', /ALIGN_LEFT)
        void = widget_label(base, VALUE='     .mp4   - MPEG-4', /ALIGN_LEFT)
        void = widget_label(base, VALUE='     .mjpeg - Motion JPEG', /ALIGN_LEFT)
        void = widget_label(base, VALUE='     .mov   - Quicktime', /ALIGN_LEFT)
        void = widget_label(base, VALUE='     .wmv   - Windows media', /ALIGN_LEFT)
    endif

;  Page2 & Page3 widgets
    self->LayoutPage2Widgets, wPage2
    self->LayoutPage3Widgets, wPage3

    return, 1
end


;------------------------------------------------------------------------------
;+
;  Enable the EM volume controls
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::EnableEMVolume
    compile_opt idl2, logical_predicate

    uName = [ $
        'EM_CT', $
        'EM_BASE_1', $
        'EM_BASE_2', $
        'EM_BASE_3', $
        'EM_CLEAR' $
        ]
    for i = 0, n_elements(uName)-1 do begin
        widget_control, self->Get(uName[i]), SENSITIVE=1
    endfor

end


;------------------------------------------------------------------------------
;+
;  Disable the EM volume controls
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::DisableEMVolume
    compile_opt idl2, logical_predicate

    uName = [ $
        'EM_CT', $
        'EM_BASE_1', $
        'EM_BASE_2', $
        'EM_BASE_3', $
        'EM_CLEAR' $
        ]
    for i = 0, n_elements(uName)-1 do begin
        widget_control, self->Get(uName[i]), SENSITIVE=0
    endfor

end


;------------------------------------------------------------------------------
;+
;  Layout control tab "Page 3" widgets
;
; @param wBase {in}{type=widget id}{required}
;   Widget id of the page.
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::LayoutPage3Widgets, wBase
    compile_opt idl2, logical_predicate

    ;  EM overlay volume
    void = widget_label(wBase, VALUE=' ', YSIZE=5)
    abase = widget_base(wBase, FRAME=1, /COLUMN)
    void = widget_label(abase, VALUE='EM Volume Overlay')

    base = widget_base(aBase, /ROW)
    void = widget_button(base, VALUE=' Load... ', UNAME='EM_LOAD', UVALUE='em_load',  $
                         TOOLTIP='Load an EM volume from disk')
    void = widget_button(base, VALUE=' Clear ', UNAME='EM_CLEAR', UVALUE='em_clear',  $
                         TOOLTIP='Delete the EM volume')

    base = widget_base(aBase, /ROW)
    loadct, GET_NAMES=ct
    void = widget_droplist(base, TITLE='Color Table:', VALUE=ct, UNAME='EM_CT', UVALUE='em_ct')

    base = widget_base(abase, /ROW, UNAME='EM_BASE_1')
;    void = widget_label(base, VALUE='Alpha (%): ')
;    void = widget_slider(base, XSIZE=150, MIN=0, MAX=100, VALUE=50, UNAME='EM_OPACITY', UVALUE='em_opacity')
;    void = widget_label(base, VALUE=' ')
    void = widget_button(base, $
        UNAME='EM_OPACITY', $
        UVALUE='em_opacity', $
        VALUE='Volume opacity...')
    base = widget_base(aBase, $
        /ROW, $
        UNAME='EM_BASE_2')
    wLabel = widget_label(base, $
        VALUE='Image opacity (%):')
    val = 20
    wSlider = widget_slider(base, $
        /DRAG, $
        MAXIMUM=100, $
        MINIMUM=0, $
        /SUPPRESS_VALUE, $
;        TITLE='Image opacity (%)', $
        UNAME='EM_IMAGE_OPACITY', $
        UVALUE='em_imageOpacity', $
        VALUE=val)
    wLabel = widget_label(base, $
        /DYNAMIC_RESIZE, $
        UNAME='emOpacityLabel', $
        VALUE=strtrim(val,2))

    base = widget_base(abase, /ROW, UNAME='EM_BASE_3')
    bbase = widget_base(base, /COLUMN, /NONEXCLUSIVE)
    void = widget_button(bbase, VALUE=' Show EM volume', UNAME='EM_SHOW_EM', UVALUE='em_show/hide')
    widget_control, void, /SET_BUTTON
    void = widget_button(bbase, VALUE=' Show molecules', UNAME='EM_SHOW_MOL', UVALUE='em_show/hide')
    widget_control, void, /SET_BUTTON

    ;  Start disabled until an EM volume is loaded
    self->DisableEMVolume
end


;------------------------------------------------------------------------------
;+
;  Layout control tab "Page 2" widgets
;
; @param wBase {in}{type=widget id}{required}
;   Widget id of base on which these controls are placed
;-
pro PALM_MainGUI::LayoutPage2Widgets, wBase
    compile_opt idl2

    void = widget_label(wBase, VALUE=' ', YSIZE=5)
    base = widget_base(wBase, FRAME=1, /COLUMN)
    abase = widget_base(base, /ROW)
    void = widget_label(abase, /ALIGN_LEFT, VALUE='Export volume as: ')
    void = widget_combobox(abase, VALUE=['IDL .sav', 'Float (little)', $
                           'Float (big)','Byte Scaled'],  $
                           UNAME='VOLUME_EXPORT_TYPE', XSIZE=125)
    abase = widget_base(base, /COLUMN)
    void = widget_button(abase, VALUE=' Export... ', UNAME='VOLUME_EXPORT', UVALUE='volume_export')

    ;  MPR movie output
    base = widget_base(wBase, FRAME=1, /COLUMN)
    void = widget_label(base, VALUE='Orthogonal View Movies')
    abase = widget_base(base, /ROW)
    void = widget_label(abase, VALUE='View: ', /ALIGN_LEFT)
    abase = widget_base(abase, /ALIGN_LEFT, /ROW, /EXCLUSIVE)
    void = widget_button(abase, VALUE='X', UNAME='MPR_X')
    widget_control, void, /SET_BUTTON
    void = widget_button(abase, VALUE='Y', UNAME='MPR_Y')
    void = widget_button(abase, VALUE='Z', UNAME='MPR_Z')
    abase = widget_base(base, /ALIGN_LEFT, /ROW)
    void = widget_label(abase, VALUE='Movie skip and average frames ')
    void = widget_slider(abase, VALUE=0, MIN=0, MAX=9, UVALUE='control_MPRSkipAndAverage', /SUPPRESS, XSIZE = 60)
    void = widget_label(abase, VALUE=' 0', /SUNKEN_FRAME, UNAME='MPR_SKIP_AND_AVERAGE')
    abase = widget_base(base, /ALIGN_LEFT, /NONEXCLUSIVE)
    void = widget_button(abase, VALUE='Do not delete frame files', UNAME='MPR_KEEP_FRAMES')
    abase = widget_base(base, /ALIGN_LEFT, /ROW)
    void = widget_droplist(abase, TITLE='FPS:  ', UNAME='MPR_FPS',  $
                           VALUE=['5','10','15','20','25','30'])
    void = widget_droplist(abase, TITLE='  Scale: ', UNAME='MPR_SCALE_FACTOR',  $
                           VALUE=['1x','2x','3x','4x','5x'])
    void = widget_label(base, VALUE='File extension determines movie type:   ', /ALIGN_LEFT)
    void = widget_label(base, VALUE='     .mj2   - Motion JPEG 2000', /ALIGN_LEFT)
    void = widget_label(base, VALUE='     .mpg   - MPEG-1', /ALIGN_LEFT)
    void = widget_label(base, VALUE='     .mp4   - MPEG-4', /ALIGN_LEFT)
    void = widget_label(base, VALUE='     .mjpeg - Motion JPEG', /ALIGN_LEFT)
    void = widget_label(base, VALUE='     .mov   - Quicktime', /ALIGN_LEFT)
    void = widget_label(base, VALUE='     .wmv   - Windows media', /ALIGN_LEFT)
    abase = widget_base(base, /ROW)

    aBase = widget_base(base, $
        /ROW)
    wButton = widget_button(aBase, $
        UVALUE='mpr_animate', $
        VALUE='Animate')
    wButton = widget_button(aBase, $
        UVALUE='mpr_movie', $
        UNAME='MAKE_MPR_MOVIE', $
        VALUE='Make Movie')
    wButton = widget_button(aBase, UVALUE='control_playback', $
        VALUE='Play Movie')
    aBase = widget_base(base, $
        /NONEXCLUSIVE, $
        /ROW)
    wButton = widget_button(aBase, $
        VALUE='Animate while creating movie', $
        UNAME='MPR_ANIMATION')

    ;  Multiple molecules
    abase = widget_base(wBase, FRAME=1, /COLUMN, UNAME='MM_BASE')
    void = widget_label(abase, VALUE='Multiple Molecules', /ALIGN_CENTER)
    void = widget_label(abase, VALUE=' ', YSIZE=15)
    base = widget_base(abase, /ROW)
    void = widget_label(base, VALUE='Show molecules:')
    values = ['ALL','1','2','3','4','1-2','1-3','1-4','1-2-3',  $
              '1-2-4','1-3-4','2-3','2-4','2-3-4','3-4']
    void = widget_droplist(base, VALUE=values, UNAME='MM_SHOW_MOL')
    void = widget_label(abase, VALUE=' ', YSIZE=15)
    void = widget_label(abase, VALUE='Labels & Hues:', /ALIGN_LEFT)

    base = widget_base(abase, /ROW, /BASE_ALIGN_CENTER)
    void = widget_label(base, VALUE='    ')
    void = widget_label(base, VALUE='1:')
    void = widget_text(base, /EDIT, VALUE='', XSIZE=10, UNAME='MM_LABEL_1')
    void = widget_label(base, VALUE=' ')
    void = widget_draw(base, XSIZE=40, YSIZE=30, RETAIN=2, UNAME='MM_DRAW1')
    void = widget_label(base, VALUE=' ')
    void = widget_slider(base, /DRAG, MIN=0, MAX=320, VALUE=0, UVALUE='mm_DRAW1', UNAME='MM_DRAW1_SLIDER')

    base = widget_base(abase, /ROW, /BASE_ALIGN_CENTER)
    void = widget_label(base, VALUE='    ')
    void = widget_label(base, VALUE='2:')
    void = widget_text(base, /EDIT, VALUE='', XSIZE=10, UNAME='MM_LABEL_2')
    void = widget_label(base, VALUE=' ')
    void = widget_draw(base, XSIZE=40, YSIZE=30, RETAIN=2, UNAME='MM_DRAW2')
    void = widget_label(base, VALUE=' ')
    void = widget_slider(base, /DRAG, MIN=0, MAX=320, VALUE=80, UVALUE='mm_DRAW2', UNAME='MM_DRAW2_SLIDER')

    base = widget_base(abase, /ROW, /BASE_ALIGN_CENTER)
    void = widget_label(base, VALUE='    ')
    void = widget_label(base, VALUE='3:')
    void = widget_text(base, /EDIT, VALUE='', XSIZE=10, UNAME='MM_LABEL_3')
    void = widget_label(base, VALUE=' ')
    void = widget_draw(base, XSIZE=40, YSIZE=30, RETAIN=2, UNAME='MM_DRAW3')
    void = widget_label(base, VALUE=' ')
    void = widget_slider(base, /DRAG, MIN=0, MAX=320, VALUE=160, UVALUE='mm_DRAW3', UNAME='MM_DRAW3_SLIDER')

    base = widget_base(abase, /ROW, /BASE_ALIGN_CENTER)
    void = widget_label(base, VALUE='    ')
    void = widget_label(base, VALUE='4:')
    void = widget_text(base, /EDIT, VALUE='', XSIZE=10, UNAME='MM_LABEL_4')
    void = widget_label(base, VALUE=' ')
    void = widget_draw(base, XSIZE=40, YSIZE=30, RETAIN=2, UNAME='MM_DRAW4')
    void = widget_label(base, VALUE=' ')
    void = widget_slider(base, /DRAG, MIN=0, MAX=320, VALUE=240, UVALUE='mm_DRAW4', UNAME='MM_DRAW4_SLIDER')

    base = widget_base(abase, /ROW, /NONEXCLUSIVE)
    void = widget_button(base, VALUE='Use hue for molecule types', UNAME='MM_HUE', UVALUE='mm')
end


;------------------------------------------------------------------------------
;+
; This method builds the information label widgets at the bottom
; of the GUI
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_MainGUI::ConstructInfoLabels, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
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

    wGeom = widget_info(self.tlb, /GEOMETRY)
    wBase = widget_base(self.tlb, $
        /ROW, $
        UNAME='info_label_base')
    wLabel = widget_label(wBase, $
        /SUNKEN_FRAME, $
        /ALIGN_LEFT, $
        UNAME='InfoLabel', $
        VALUE=' ', $
        XSIZE=wGeom.xSize)

    return, 1

end


;------------------------------------------------------------------------------
;+
; This method constructs the GUI's main menu
;
; @param
;   menu {in}{type=long}{required}
;     The widget ID of the top level base's menu
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_MainGUI::ConstructMenu, menu, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
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
;
; File menu
;
    wFile = widget_button(menu, $
        /MENU, $
        UNAME='file', $
        UVALUE='file', $
        VALUE='File')
    if (self.standalone) then begin
        void = widget_button(wFile, UVALUE='menu_file_open', VALUE='Open...')
    endif
    wButton = widget_button(wFile, $
        UVALUE='menu_file_exit', $
        VALUE='Exit')
;
; View menu
;
    wView = widget_button(menu, $
        /MENU, $
        UNAME='view', $
        UVALUE='view', $
        VALUE='View')
; Color table
    wColorTable = widget_button(wView, $
        /MENU, $
        VALUE='Color table')
    wButton = widget_button(wColorTable, $
        UVALUE='menu_view_ColorTable', $
        VALUE='Edit...')
    wButton = widget_button(wColorTable, $
        /CHECKED_MENU, $
        UNAME='menu_view_InvertColorTable', $
        UVALUE='menu_view_InvertColorTable', $
        VALUE='Invert')
; Background color
    wBackground = widget_button(wView, $
        /MENU, $
        VALUE='Background color')
    wButton = widget_button(wBackground, $
        /CHECKED_MENU, $
        UVALUE='menu_view_background_Black', $
        VALUE='Black')
    widget_control, wButton, /SET_BUTTON
    wButton = widget_button(wBackground, $
        /CHECKED_MENU, $
        UVALUE='menu_view_background_White', $
        VALUE='White')
; Drag quality
    wDrag = widget_button(wView, $
        /MENU, $
        VALUE='Drag Quality')
    wButton = widget_button(wDrag, $
        /CHECKED_MENU, $
        UVALUE='menu_view_drag_low', $
        VALUE='Low')
    wButton = widget_button(wDrag, $
        /CHECKED_MENU, $
        UVALUE='menu_view_drag_medium', $
        VALUE='Medium')
    widget_control, wButton, /SET_BUTTON
    wButton = widget_button(wDrag, $
        /CHECKED_MENU, $
        UVALUE='menu_view_drag_high', $
        VALUE='High')
; Reset
    wReset = widget_button(wView, $
        /MENU, $
        VALUE='Reset')
    wButton = widget_button(wReset, $
        UVALUE='menu_view_reset_all', $
        VALUE='All Displays')
    wButton = widget_button(wReset, $
        UVALUE='menu_view_reset_3d', $
        VALUE='3D Display')
    wXYZ = widget_button(wReset, $
        /MENU, $
        UVALUE='menu_view_reset', $
        VALUE='XYZ Display')
    wButton = widget_button(wXYZ, $
        UVALUE='menu_view_reset_XYZ_all', $
        VALUE='All')
    wButton = widget_button(wXYZ, $
        UVALUE='menu_view_reset_XYZ_scale/translate', $
        VALUE='Scale/Translation')
    wButton = widget_button(wXYZ, $
        UVALUE='menu_view_reset_XYZ_windowlevel', $
        VALUE='Window Level')
    wButton = widget_button(wReset, $
        /SEPARATOR, $
        UVALUE='menu_view_reset_grid', $
        VALUE='2X2 Grid')
    wButton = widget_button(wView, $
        /CHECKED_MENU, $
        UNAME='menu_view_control', $
        UVALUE='menu_view_control', $
        VALUE='Show Controls')
    widget_control, wButton, /SET_BUTTON
    wButton = widget_button(wView, $
        /CHECKED_MENU, $
        UNAME='menu_view_toolbar', $
        UVALUE='menu_view_toolbar', $
        VALUE='Show Toolbar')
    widget_control, wButton, /SET_BUTTON
;
; Help menu
;
    wHelp = widget_button(menu, $
        /MENU, $
        VALUE='Help')
    wButton = widget_button(wHelp, $
        UVALUE='menu_help_about', $
        VALUE='About...')

    return, 1

end


;------------------------------------------------------------------------------
;+
; This method constructs the toolbar for specifying the manipulation mode.
;
; @param wMainBase {in}{type=widget id}{required}
;   Reference to the main base
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_MainGUI::ConstructToolbar, wMainBase, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
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
    bitmapDir = self.ResourceDirectory + 'bitmaps' + path_sep()
    files = file_search(bitmapDir+'*.bmp', /TEST_REGULAR)
    wToolbarBase = widget_base(wMainBase, $
        /ROW)
; Add the mouse mode buttons
    wBase = widget_base(wToolbarBase, $
        /BASE_ALIGN_CENTER, $
        /EXCLUSIVE, $
        /ROW, $
        /TOOLBAR, $
        UNAME='toolbar_base')
    uNames = 'toolbar_' + strlowcase(*self.pWindowModes)
    for i = 0, n_elements(*self.pWindowModes)-1 do begin
        index = where(files EQ bitmapDir+(*self.pBitmapFiles)[i]+'.bmp', count)
        if count GT 0 then begin
            tooltip = strupcase(strmid((*self.pWindowModes)[i],0,1)) + $
                strmid((*self.pWindowModes)[i],1)
            wButton = widget_button(wBase, $
                /BITMAP, $
                /NO_RELEASE, $
                TOOLTIP=tooltip, $
                UNAME=uNames[i], $
                UVALUE=uNames[i], $
                VALUE=files[index[0]])
        endif else begin
            print, 'Unable to locate bitmap file for ' + (*self.pWindowModes)[i]
        endelse
    endfor

    wButton = self -> Get(uNames[0])
    if widget_info(wButton, /VALID) then $
        widget_control, wButton, /SET_BUTTON

; Other toolbar buttons
    wBase = widget_base(wToolbarBase, $
        /ROW)

    return, 1

end

;------------------------------------------------------------------------------
;+
; This method constructs the GUI
;
; @keyword COMPOSITE_FUNCTION {in}{type=number}
;   Volume composite value
;
; @keyword DATA_RANGE {in}{type=vector}
;   Data range (nm)
;
; @keyword ERROR_MESSAGE {out}{type=string}
;   Output error message, if any
;
; @keyword GROUP_LEADER {in}{type=widget id}
;   Widget id of PeakSelector base
;
; @keyword MAP {in}{type=number}
;   If not set, hide the widgets
;
; @keyword MAXIMUM_SCALE {in}{type=number}
;   Maximum scane value
;
; @keyword MAXIMUM_VOLUME_DIMENSION {in}{type=number}
;   Maxiumum volume dimension
;
; @keyword SUBVOLUME_WIDTH {in}{type=number}
;   Subvolume size (cube, nm)
;
; @keyword Z_SCALE_FACTOR {in}{type=number}
;   Z axs scale factor
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_MainGUI::ConstructWidgets, $
    COMPOSITE_FUNCTION=CompositeFunction, $
    DATA_RANGE=DataRange, $
    ERROR_MESSAGE=ErrMsg, $
    GROUP_LEADER=GroupLeader, $
    MAP=map, $
    MAXIMUM_SCALE=MaxScale, $
    MAXIMUM_VOLUME_DIMENSION=MaxVolumeDimension, $
    SUBVOLUME_WIDTH=SubVolumeWidth, $
    Z_SCALE_FACTOR=zScaleFactor

    compile_opt idl2
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

    scr = get_screen_size()<1300
    scr *= 0.95 ;make slightly smaller than screen
    scr[1] -= 100 ;subtract small amount from y dimension
    pad=2
    tlbSize = [scr[0], scr[1]]

    xyzXSize = 0.5*tlbSize[0] - 2*pad
    windowXSize = tlbSize[0] - xyzxSize - 2*pad

    self.tlb = widget_base(/COL, $
        EVENT_PRO = 'GenericClassEvent', $
        GROUP_LEADER=GroupLeader, $
        KILL_NOTIFY = 'GenericClassKillNotify', $
        NOTIFY_REALIZE = 'GenericClassNotifyRealize', $
        MAP=keyword_set(map), $
        MBAR=menu, $
        TITLE='3D Viewer, Howard Hughes Medical Institute ('+MOD_DATE()+')', $
        /TLB_KILL_REQUEST_EVENTS, $
        /TLB_SIZE_EVENTS, $
        UNAME='tlb', $
        UVALUE=self, $
        XSIZE=tlbSize[0], $
        YSIZE=tlbSize[1])
; Construct the menu
    if ~self->ConstructMenu(menu, ERROR_MESSAGE=errMsg) then $
        return, 0
    menuGeom = widget_info(menu, /GEOM)
; Main base.  Do not set COLUMN or ROW to get around control resizing issue
    wMainBase = widget_base(self.tlb)
; Draw widgets
    wDrawBase = widget_base(wMainBase, $
        UNAME='DrawBase')
    wTopDrawBase = widget_base(wDrawbase, $
        UNAME='TopBase')
    wMiddleBase = widget_base(wDrawBase, $
        UNAME='MiddleBase')
    wBottomDrawBase = widget_base(wDrawBase, $
        UNAME='BottomBase')
    wXDraw = widget_draw(wTopDrawBase, $
        CLASSNAME='PALM_XYZWindow', $
        /EXPOSE_EVENTS, $
        GRAPHICS_LEVEL=2, $
        RENDERER=self.renderer, $
        RETAIN = (self.renderer eq 0) ? 0 : 1, $
        UNAME='PALM_XWindow', $
        XSIZE = xyzXSize)
    wYDraw = widget_draw(wBottomDrawBase, $
        CLASSNAME='PALM_XYZWindow', $
        /EXPOSE_EVENTS, $
        GRAPHICS_LEVEL=2, $
        RENDERER=self.renderer, $
        RETAIN = (self.renderer eq 0) ? 0 : 1, $
        UNAME='PALM_YWindow', $
        XSIZE = xyzXSize)
    wZDraw = widget_draw(wTopDrawBase, $
        CLASSNAME='PALM_XYZWindow', $
        /EXPOSE_EVENTS, $
        GRAPHICS_LEVEL=2, $
        RENDERER=self.renderer, $
        RETAIN = (self.renderer eq 0) ? 0 : 1, $
        UNAME='PALM_ZWindow', $
        XSIZE = xyzXSize)
    w3DDraw = widget_draw(wBottomDrawBase, $
        CLASSNAME='PALM_3DWindow', $
        EXPOSE_EVENTS=(self.renderer eq 0), $
        GRAPHICS_LEVEL=2, $
        RENDERER=self.renderer, $
        RETAIN = (self.renderer eq 0) ? 0 : 2, $
        UNAME='PALM_3DWindow')
    wDraw = widget_draw(wMiddleBase, $
        /BUTTON_EVENTS, $
        /MOTION_EVENTS, $
        UNAME='MiddleDraw', $
        UVALUE='PALM_SizeWindow', $
        XSIZE=6, $
        YSIZE=6)
    if ~self->ConstructToolbar(wMainBase, ERROR_MESSAGE=ErrMsg) then $
        return, 0
    if ~self->ConstructContextMenus(ERROR_MESSAGE=ErrMsg) then $
        return, 0
    if ~self->ConstructControlBase(wMainBase, $
        COMPOSITE_FUNCTION=CompositeFunction, $
        DATA_RANGE=DataRange, $
        ERROR_MESSAGE=ErrMsg, $
        MAXIMUM_SCALE=MaxScale, $
        MAXIMUM_VOLUME_DIMENSION=MaxVolumeDimension, $
        SUBVOLUME_WIDTH=SubVolumeWidth, $
        Z_SCALE_FACTOR=zScaleFactor) then $
        return, 0

; Add the information base
    if ~self->ConstructInfoLabels(ERROR_MESSAGE=errMsg) then $
        return, 0

    infoGeom = widget_info(widget_info(self.tlb, FIND_BY_UNAME='info_label_base'), $
        /GEOMETRY)
    ySize = (tlbSize[1]-menuGeom.scr_ySize-infoGeom.scr_ySize-2*pad)/2
    widget_control, wXDraw, YSIZE=ySize
    widget_control, wYDraw, YSIZE=ySize
    widget_control, wZDraw, YSIZE=ySize
    widget_control, w3DDraw, YSIZE=ySize

    return, 1

end



;------------------------------------------------------------------------------
;+
; This method handles events from the control base
;
; @Param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from the PALM_MainGUI base widget.
;
; @Param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::ControlEvent, event, components, $
    ERROR_MESSAGE=ErrMsg

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    case components[0] of
        'animate': self.o3DDisplay -> Animate
        'mpeg': self.o3DDisplay -> Animate, /EXPORT_TO_MPEG, NO_DRAW=~widget_info(self->Get('animate'), /BUTTON_SET)
        'playback': self->Playback
        'FiducialCutoff': begin
            widget_control, widget_info(self.tlb, FIND_BY_UNAME='FiducialCutoffLabel'), $
                SET_VALUE=strtrim(event.value,2)
        end
        'filter': begin
; base_maxscale, base_datarange
            if event.select then begin
                if components[1] EQ 'auto' then begin
                    str = 'log'
                    sensitive = 0
                endif else begin
                    str = 'auto'
                    sensitive = 1
                endelse
                widget_control, self->Get('control_filter_'+str), SET_BUTTON=0
                widget_control, self->Get('base_datarange'), SENSITIVE=sensitive
                widget_control, self->Get('base_maxscale'), SENSITIVE=0
            endif else begin
                widget_control, self->Get('base_datarange'), /SENSITIVE
                widget_control, self->Get('base_maxscale'), /SENSITIVE
;                widget_control, self->Get('maxscale'), SENSITIVE=0
            endelse
        end
        'FlythroughBrowse': begin
            file = dialog_pickfile( $
                DIALOG_PARENT=self.tlb, $
                FILTER='*.csv', $
                /MUST_EXIST, $
                TITLE='Select a Flythrough File')
            if file NE '' then begin
                wText = widget_info(self.tlb, FIND_BY_UNAME='FlythroughFile')
                widget_control, wText, SET_VALUE=file
            endif
        end
        'increment': widget_control, self->Get('incrementLabel'), $
            SET_VALUE=string(event.value/10.0, FORMAT='(f4.1)')
        'maxscale': begin
            widget_control, self->Get('maxscale'), $
                SET_VALUE=string(float(event.value)/ $
                (widget_info(self->Get('control_maxscale'),/SLIDER_MIN_MAX))[1], $
                FORMAT='(f-4.2)')
        end
        'Recalculate': self -> RecalculateVolume
        'rotate': self -> RotateVolume, components[1:*]
        'RotationAngle': begin
            widget_control, widget_info(self.tlb, FIND_BY_UNAME='RotationAngle'), $
                SET_VALUE=strtrim(event.value,2)
        end
        'VolumeDimensions': begin
            widget_control, widget_info(self.tlb, FIND_BY_UNAME='VolumeDimensionsLabel'), $
                SET_VALUE=strtrim(event.value,2)
        end
        'zScaleFactor': begin
            widget_control, widget_info(self.tlb, FIND_BY_UNAME='zScaleFactorLabel'), $
                SET_VALUE=string(float(event.value)/10, FORMAT='(f4.1)')

        end
        'animationtype' : begin
            rotationbase = widget_info(self.tlb, FIND_BY_UNAME='AnimateRotationBase')
            shearbase = widget_info(self.tlb, FIND_BY_UNAME='AnimateShearBase')
            wFlythroughBase = widget_info(self.tlb, FIND_BY_UNAME='AnimateFlythroughBase')
            mode = widget_info(event.id, /combobox_gettext)
            widget_control, rotationbase, map = mode eq 'Rotation'
            widget_control, shearbase, map = mode eq 'Shear'
            widget_control, wFlythroughBase, MAP=(mode EQ 'Flythrough')
        end
        'shear': self -> ShearVolume, components[1:*]
        'ShearScale' : begin
            widget_control, widget_info(self.tlb, FIND_BY_UNAME='ShearScaleLabel'), $
                SET_VALUE=strtrim(string(event.value/10., format='(f3.1)'), 2)
            end
        'MPRSkipAndAverage' : begin
            widget_control, widget_info(self.tlb, FIND_BY_UNAME='MPR_SKIP_AND_AVERAGE'), $
                SET_VALUE = strtrim(event.value, 2)
            end
        'UseHueMPR':begin
            if event.select then begin
                self.oXYZDisplay->XYHueImage
                self.oXYZDisplay->XZHueImage
                self.oXYZDisplay->YZHueImage
                self.oXYZDisplay->RenderScene
            endif else begin
                self.oXYZDisplay->UpdateImages, /ALL
                self.oXYZDisplay->RenderScene
            endelse
        end
        else: message, /CONTINUE, 'Unknown component: ' + components[0]
    endcase

end


;------------------------------------------------------------------------------
;+
; This method has the primary responsibility for cleaning up the PALM_MainGUI
; object at the end of its lifecycle.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::Destruct, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    if obj_valid(self.o3DDisplay) then $
        obj_destroy, self.o3DDisplay
    if obj_valid(self.o3DWindow) then $
        obj_destroy, self.o3DWindow
    if obj_valid(self.o3DWindowObserver) then $
        obj_destroy, self.o3DWindowObserver
    if obj_valid(self.oXYZDisplay) then $
        obj_destroy, self.oXYZDisplay
    if obj_valid(self.oXYZWindow[0]) then $
        obj_destroy, self.oXYZWindow
    if obj_valid(self.oXYZWindowObserver) then $
        obj_destroy, self.oXYZWindowObserver
    ptr_free, self.pBitmapFiles, self.pCursors, self.pWindowModes

    if widget_info(self.tlb, /VALID) then $
        widget_control, self.tlb, /DESTROY

    if obj_valid(self) then $
        obj_destroy, self

end


;------------------------------------------------------------------------------
;+
; This method handles events from draw widgets who do not use an observer
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from a button on the PALM_MainGUI widget.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::DrawEvent, event, $
    ERROR_MESSAGE=errMsg

    case event.type of
        0: begin ; Press
            if ~array_equal(self.xy, [-1,-1]) then $
                return
            self.xy = [event.x, event.y]
            self.o3DDisplay->GetProperty, HIDE_SLICE_LINES=hlines, HIDE_VOLUME=hvol,  $
                                          HIDE_SLICE_PLANES=hplanes
            self.hide = [hlines,hplanes,hvol]
            self.o3DDisplay->SetProperty, $
                /EM_VOLUME_HIDE, $
                /HIDE_SLICE_LINES, $
                /HIDE_SLICE_PLANES, $
                /HIDE_VOLUME
            self.o3DDisplay->RenderScene
            self.oXYZDisplay->SetProperty, HIDE=1
            self.oXYZDisplay->RenderScene
        end
        1: begin ; Release
            self.xy = [-1,-1]
            self.o3DDisplay -> SetProperty, $
                EM_VOLUME_HIDE=~widget_info(self->Get('EM_SHOW_EM'), /BUTTON_SET), $
                HIDE_SLICE_LINES=self.hide[0], $
                HIDE_SLICE_PLANES=self.hide[1], $
                HIDE_VOLUME=self.hide[2]
            self -> BaseEvent, /NO_XYZ_DRAW
            self.oXYZDisplay->SetProperty, HIDE=0
            self.oXYZDisplay->SetViews, /ADJUST_SLIDERS
            self.o3DDisplay->RenderScene
        end
        2: begin ; Motion
            if array_equal(self.xy, [-1,-1]) then $
                return
            wGeom = widget_info(self.tlb, /GEOMETRY)
            wLabelBase = self -> Get('info_label_base')
            wLabelGeom = widget_info(wLabelBase, /GEOMETRY)
            menuGeom = widget_info(self -> Get('menu_base'), /GEOM)
            wControlBase = self -> Get('ControlBase')
            controlGeom = widget_info(wControlBase, /GEOM)
            wMiddleBase = self -> get('MiddleBase')
            wMiddleGeom = widget_info(wMiddleBase, /GEOMETRY)
            wDrawGeom = widget_info(self->Get('MiddleDraw'), /GEOM)
            xSize = wGeom.xSize - 4*wGeom.yPad

            if widget_info(self->Get('menu_view_control'), /BUTTON_SET) then $
                xSize = xSize - controlGeom.scr_xSize
            ySize = wGeom.ySize - $
                menuGeom.scr_ySize - $
                wMiddleGeom.scr_ySize - $
                wLabelGeom.scr_ySize
            x = ((float(wDrawGeom.xOffset+wDrawGeom.xSize/2+event.x-wDrawGeom.scr_xSize+1) / xSize) $
                >self.winPctRange[0])<self.winPctRange[1]
        ; I'm not sure why but multiplying by 1.02 yields better results...
            y = (1.02*(float(wMiddleGeom.yOffset+wMiddleGeom.ySize/2-event.y+wDrawGeom.ySize-1) / ySize) $
                >self.winPctRange[0])<self.winPctRange[1]
            self.winPct = [x,y]
            self -> BaseEvent, /NO_UPDATE
            self.oXYZDisplay->SetViews, /ADJUST_SLIDERS
        end
        4: begin
        ; Expose event
            widget_control, event.id, GET_VALUE=oWindow
            oWindow->Draw
        end
        else: help, event, /STR
    endcase

end


;------------------------------------------------------------------------------
;+
;  Change the color table on the EM volume
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::EMVolumeColorTable, $
    NO_DRAW=noDraw

    compile_opt idl2, logical_predicate

    oPalette = obj_new('IDLgrPalette')
    oPalette->LoadCT, widget_info(self->Get('EM_CT'), /DROPLIST_SELECT)
    oPalette->GetProperty, BLUE_VALUES=blue, $
        GREEN_VALUES=green, $
        RED_VALUES=red
    obj_destroy, oPalette
    ct = [[red],[green],[blue]]
    self.o3DDisplay->SetProperty, EM_COLOR_TABLE=ct
    self.oXYZDisplay->SetProperty, EM_COLOR_TABLE=ct

    if ~keyword_set(noDraw) then begin
        self.o3DDisplay->RenderScene
        self.oXYZDisplay->RenderScene
    endif

end


;------------------------------------------------------------------------------
;+
; This method handles events from droplist widgets
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from a button on the PALM_MainGUI widget.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::DroplistEvent, event, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    widget_control, event.id, GET_UVALUE=uval
    if n_elements(uval) EQ 0 then $
        return

    case uval of
        'em_ct': self->EMVolumeColorTable
        else: print, uval
    endcase

end


;------------------------------------------------------------------------------
;+
; Main event handler
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from the PALM_MainGUI widget.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::Event, event, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    if n_elements(event) EQ 0 then $
        return

    case strupcase(tag_names(event, /STRUCTURE)) of
        'WIDGET_BASE': self -> BaseEvent, event
        'WIDGET_BUTTON': self -> ButtonEvent, event
        'WIDGET_COMBOBOX' : self -> ButtonEvent, event
        'WIDGET_DRAW': self -> DrawEvent, event
        'WIDGET_KILL_REQUEST': self -> ExitEvent, event
        'WIDGET_DROPLIST': self -> DroplistEvent, event
        'WIDGET_SLIDER': self -> SliderEvent, event
        else:
    endcase

end


;------------------------------------------------------------------------------
;+
; This method is called when the user closes the application
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from the PALM_MainGUI widget.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::ExitEvent, event, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    widget_control, self.tlb, /DESTROY

end


;--------------------------------------------------------------
;+
; Playback of movie files
;
; @History
;   Mar 2009, AB - ITT, Initial version <br>
;   Jun 2009, RK - ITT, Mod for multiple players<br>
;-
;--------------------------------------------------------------
pro PALM_MainGui::Playback
    compile_opt idl2, logical_predicate

    ;  Ask for the file first, use the extension to decide what player to launch
    fname = dialog_pickfile(TITLE='Select a movie file...', DIALOG_PARENT=self.tlb,  $
                            GET_PATH=path)
    if (fname eq '') then return
    cd, path
    bname = file_basename(fname)

    ;  Get the extension
    ext = ''
    idx = strpos(bname, '.', /REVERSE_SEARCH)
    if (idx ne -1) then begin
        if ((idx+1) ne strlen(bname)) then  $
            ext = strmid(bname, idx+1)
    endif
    ext = strlowcase(ext)

    ;  Launch according to type
    switch ext of
        'mj2': begin
            ; Check if it is already running, if so show and return
            if xregistered('mj2player') then return

            ; Create player, make it non-blocking for now
            oPlayer = obj_new('PALM_Mj2Player', fname)
            oPlayer->CreateGui, NO_BLOCK=1, GROUP_LEADER=self.tlb
            break
        end
        'mov':
        'mpg':
        'mp4':
        'wmv':
        'mjpeg': begin
            ;  Launch mplayer or Windows media depending upon os family
            if (!version.os_family eq 'unix') then begin
                cmd = 'mplayer '+ fname + ' -loop 0'
                spawn, cmd, output, /STDERR, EXIT_STATUS=es
                if (es ne 0) then begin
                    void = dialog_message(/ERROR, TITLE='Error', DIALOG_PARENT=self.tlb,  $
                                          ['There was an error running mplayer:', output, $
                                           'exit status = ' + strtrim(es,2)])
                endif
            endif else begin
                if file_test('C:\MPlayer\mplayer.exe') then begin
                    ;  Look for MPlayer first
                    cmd = 'cd C:\MPlayer & mplayer ' + fname + ' -loop 0'
                    spawn, cmd, output, /STDERR, /HIDE, EXIT_STATUS=es
                    if (es ne 0) then begin
                        void = dialog_message(/ERROR, TITLE='Error', DIALOG_PARENT=self.tlb,  $
                                             ['There was an error running MPlayer:',   $
                                             output,'exit status = ' + strtrim(es,2)])
                    endif
                endif else begin
                    ;  Launch whatever player is associated with the file
                    cname = filepath('launch_movie.vbs', /TMP)
                    ccname = '"' + file_dirname(cname) + path_sep() + file_basename(cname, '.vbs') + '"'
                    openw, u, cname, /GET_LUN
                    writeu, u, 'Sub Run(ByVal sFile)'                              + string(byte([13,10]))
                    writeu, u, 'Dim shell'                                         + string(byte([13,10]))
                    writeu, u, '    Set shell = CreateObject("WScript.Shell")'     + string(byte([13,10]))
                    writeu, u, '    shell.Run Chr(34) & sFile & Chr(34), 1, false' + string(byte([13,10]))
                    writeu, u, '    Set shell = Nothing'                           + string(byte([13,10]))
                    writeu, u, 'End Sub'                                           + string(byte([13,10]))
                    writeu, u, 'Run Wscript.Arguments(0)'                          + string(byte([13,10]))
                    free_lun, u
                    cmd = ccname + ' ' + fname
                    setenv, 'CMD='+cmd
                    spawn, '%CMD%', output, /STDERR, /HIDE, EXIT_STATUS=es
                    if (es ne 0) then begin
                        void = dialog_message(/ERROR, TITLE='Error', DIALOG_PARENT=self.tlb,  $
                                              ['There was an error opening the movie file:',   $
                                               output,'exit status = ' + strtrim(es,2)])
                    endif
                endelse
            endelse
            break
        end
        else: begin
            ;  Unknown movie format
            void = dialog_message(/ERROR, TITLE='Error', DIALOG_PARENT=self.tlb,  $
                                  'Unknown movie file extension: "' + ext + '"')
            return
        end
    endswitch

end


;------------------------------------------------------------------------------
;+
; This method is called when a button under the File menu option is
; pressed
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from the PALM_MainGUI widget.
;
; @param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::FileEvent, event, components, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    case components[0] of
        'exit': self -> ExitEvent, event
        'output': self -> OutputEvent, components[1:*]
        else:  message, /CONTINUE, 'Unknown component: ' + components[0]
    endcase

end


;------------------------------------------------------------------------------
;+
; This method returns the widget ID of the widget with the specified UNAME
;
; @param
;    name {in}{type=string}{required}
;      The UNAME of the widget whose ID is to be returned.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_MainGUI::Get, name, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
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

    if n_elements(name) EQ 0 then $
        return, 0

    if size(name[0], /TYPE) NE 7 then $
        return, 0

    case name of
        'menu_base': begin
            wButton = widget_info(self.tlb, FIND_BY_UNAME='file')
            return, widget_info(wButton, /PARENT)
        end
        else: return, widget_info(self.tlb, FIND_BY_UNAME=name[0])
    endcase

end


;------------------------------------------------------------------------------
;+
; This method returns the widget ID of the widget with the specified UNAME
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @keyword ANIMATION_TYPE {out}{type=number}
;   Animation type
;
; @keyword ROTATION_ANGLE {out}{type=number}
;   Rotation angle (degrees)
;
; @keyword ROTATION_AXIS {out}{type=number}
;   Axis about which to rotate
;
; @keyword ROTATION_INCREMENT {out}{type=number}
;   Rotation increment
;
; @keyword SHEAR_AXIS {out}{type=number}
;   Shear axis label
;
; @keyword SHEAR_MODE {out}{type=number}
;   Shear mode (W-E, S-N, circular)
;
; @keyword SHEAR_SCALE {out}{type=number}
;   Maximum size of the shear
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::GetAnimationSettings, $
    ANIMATION_TYPE=animationType, $
    FLYTHROUGH_FILE=FlythroughFile, $
    ROTATION_ANGLE=angle, $
    ROTATION_AXIS=rotationAxis, $
    ROTATION_INCREMENT=rotInc, $
    SHEAR_AXIS=shearAxis, $
    SHEAR_MODE=shearMode, $
    SHEAR_SCALE=shearScale

    wSibling = widget_info(self.tlb, FIND_BY_UNAME='RotateX')
    repeat begin
        if widget_info(wSibling, /BUTTON_SET) then begin
            widget_control, wSibling, GET_UVALUE=uval
            rotationAxis = (strtok(uval, '_', /EXTRACT))[2]
        endif
        wSibling = widget_info(wSibling, /SIBLING)
    endrep until (~widget_info(wSibling, /VALID_ID))

    shearAxis = 'Z'

    widget_control, widget_info(self.tlb, FIND_BY_UNAME='incrementLabel'), GET_VALUE=sVal
    rotInc = round(float(sVal))

    widget_control, widget_info(self.tlb, FIND_BY_UNAME='RotationAngle'), $
        GET_VALUE=sVal
    angle = fix(sVal)
    widget_control, widget_info(self.tlb, FIND_BY_UNAME='ShearScaleLabel'), GET_VALUE=shearScale
    animationType = widget_info(widget_info(self.tlb, $
        FIND_BY_UNAME='AnimationType'), /COMBOBOX_GETTEXT)

    wShearMode = widget_info(self.tlb, FIND_BY_UNAME='ShearMode')
    shearMode = widget_info(wShearMode, /DROPLIST_SELECT)

    widget_control, self->Get('FlythroughFile'), GET_VALUE=FlythroughFile
    FlythroughFile = FlythroughFile[0]

end


;------------------------------------------------------------------------------
;+
; This method returns a reference to a specified object.  If no
; match is found a null object will be returned
;
; @param
;    name {in}{type=string}{required}
;      The name of the object reference to be retrieved.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_MainGUI::GetObjectByName, name, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return, obj_new()
    endif

    if n_elements(name) EQ 0 then $
        return, obj_new()

    if size(name[0], /TYPE) NE 7 then $
        return, obj_new()

    case strupcase(name) of
        'XYZDISPLAY': return, self.oXYZDisplay
        '3DDISPLAY': return, self.o3DDisplay
        '3DWINDOW': return, self.o3DWindow
        else: return, obj_new()
    endcase

end


;------------------------------------------------------------------------------
;+
; This method is for retrieving object properties
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @Keyword
;   TLB {out}{optional}
;     Set this keyword to a named variable to retrieve the widget ID for the
;     top level base
;
; @keyword
;    WINDOW_3D {out}{optional}
;      Set this keyword to a named variable to retrieve an object
;      reference to the 3D window.
;
; @keyword DISPLAY_3D {out}{optional}
;   The 3D display object reference.
;
; @keyword DISPLAY_XYZ {out}{optional}
;   The XYZ display object reference
;
; @keyword REF_VOXEL_VOLUME {out}{optional}
;   The reference voxel volume
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::GetProperty, $
    ERROR_MESSAGE=errMsg, $
    DISPLAY_3D=o3DDisplay, $
    DISPLAY_XYZ=oXYZDisplay, $
    TLB=tlb, $
    WINDOW_3D=o3DWindow,  $
    MEAN_VOXEL_BRIGHTNESS=meanVoxBrightness

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    if arg_present(meanVoxBrightness) then begin
        meanVoxBrightness = self.meanVoxBrightness
    endif

    if arg_present(o3DDisplay) then $
        o3DDisplay = self.o3DDisplay

    if arg_present(oXYZDisplay) then $
        oXYZDisplay = self.oXYZDisplay

    if arg_present(tlb) then $
        tlb = self.tlb

    if arg_present(o3DWindow) then $
        o3DWindow = self.o3DWindow

end

;------------------------------------------------------------------------------
;+
; This method initializes the object
;
; @Keyword
;   COLOR_TABLE {in}{type=bytarr}{optional}
;     Set this keyword to a [3,256] byte array specifying the color table to
;     be used in the displays
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
; @keyword
;    VERBOSE {in}{type=boolean}{optional}
;      Setting this keyword will result in the object displaying any
;      error messages to the IDL output log.
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Global Services Group
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_MainGUI::Init, $
    BACKGROUND_COLOR=backGroundColor, $
    COLOR_TABLE=ColorTable, $
    COMPOSITE_FUNCTION=CompositeFunction, $
    DATA_RANGE=DataRange, $
    ERROR_MESSAGE=errMsg, $
    GROUP_LEADER=GroupLeader, $
    HIDE=doHide, $
    MAXIMUM_SCALE=MaxScale, $
    MAXIMUM_VOLUME_DIMENSION=MaxVolumeDimension, $
    SUBVOLUME_WIDTH=SubVolumeWidth, $
    VERBOSE=verbose, $
    Z_SCALE_FACTOR=zScaleFactor,  $
    STANDALONE=standalone

    self.verbose = keyword_set(verbose)
    self.standalone = keyword_set(standalone)

    compile_opt idl2
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

; Initialize the objects state
    if ~self->InitializeState(ERROR_MESSAGE=errMsg) then begin
        catch, /CANCEL
        return, 0
    endif
; Construct the widgets
    void = self -> ConstructWidgets(GROUP_LEADER=GroupLeader, $
        COMPOSITE_FUNCTION=CompositeFunction, $
        DATA_RANGE=DataRange, $
        ERROR_MESSAGE=ErrMsg, $
        MAP=~keyword_set(doHide), $
        MAXIMUM_SCALE=MaxScale, $
        MAXIMUM_VOLUME_DIMENSION=MaxVolumeDimension, $
        SUBVOLUME_WIDTH=SubVolumeWidth, $
        Z_SCALE_FACTOR=zScaleFactor)
    if ~void then begin
        catch, /CANCEL
        return, 0
    endif

; Realize the widgets
    widget_control, self.tlb, /REALIZE

    ;  Color the draw widgets properly
    t = {WIDGET_SLIDER}
    t.value = 0
    self->MultMolSliderEvent, t, 'DRAW1'
    t.value = 80
    self->MultMolSliderEvent, t, 'DRAW2'
    t.value = 160
    self->MultMolSliderEvent, t, 'DRAW3'
    t.value = 240
    self->MultMolSliderEvent, t, 'DRAW4'

    if n_elements(BackgroundColor) EQ 3 then begin
        self.o3DDisplay -> SetProperty, BACKGROUND_COLOR=BackgroundColor
        self.oXYZDisplay -> SetProperty, BACKGROUND_COLOR=BackgroundColor
    endif

    catch, /CANCEL
    return, 1
end


;------------------------------------------------------------------------------
;+
; This method is responsible for initializing the standard graphics tree
; to be displayed in the draw widget.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::InitializeDisplays, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

; Initialize the displays
    wDraw = self -> Get('PALM_3DWindow')
    widget_control, wDraw, GET_VALUE=oWindow
    oWindow -> SetProperty, VERBOSE=self.verbose
    self.o3DWindow = oWindow
    self.o3DDisplay = obj_new('PALM_3DDisplay', self, VERBOSE=self.verbose)

    wDraw = self -> Get('PALM_XWindow')
    widget_control, wDraw, GET_VALUE=oWindow
    self.oXYZWindow[0] = oWindow
    oWindow -> SetProperty, NAME='X'
    wDraw = self -> Get('PALM_YWindow')
    widget_control, wDraw, GET_VALUE=oWindow
    self.oXYZWindow[1] = oWindow
    oWindow -> SetProperty, NAME='Y'
    wDraw = self -> Get('PALM_ZWindow')
    widget_control, wDraw, GET_VALUE=oWindow
    self.oXYZWindow[2] = oWindow
    oWindow -> SetProperty, NAME='Z'

    self.oXYZDisplay = obj_new('PALM_XYZDisplay', $
        MAIN_GUI=self, $
        X_WINDOW=self.oXYZWindow[0], $
        Y_WINDOW=self.oXYZWindow[1], $
        Z_WINDOW=self.oXYZWindow[2])

; Initialize window observer
    self.o3DWindowObserver = obj_new('PALM_3DObserver', self, $
        VERBOSE=self.verbose)
    self.o3DWindow -> AddWindowEventObserver, self.o3DWindowObserver
    self.o3DWindow -> SetEventMask, $
        /BUTTON_EVENTS, $
        /MOTION_EVENTS, $
        /WHEEL_EVENTS, $
        /KEYBOARD_EVENTS
    self.oXYZWindowObserver = obj_new('PALM_XYZObserver', self, $
        DRAW_WIDGET=wDraw)
    self.oXYZWindowObserver -> SetProperty, CURSOR_MODE='SliceScroll'
    for i = 0, 2 do begin
        self.oXYZWIndow[i] -> AddWindowEventObserver, self.oXYZWindowObserver
        self.oXYZWindow[i] -> SetEventMask, $
            /BUTTON_EVENTS, $
            /MOTION_EVENTS, $
            /WHEEL_EVENTS, $
            /KEYBOARD_EVENTS
    endfor

end


;------------------------------------------------------------------------------
;+
; This method is responsible for initializing the member variables of the class.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_MainGUI::InitializeState, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
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

    self.renderer = 0; hardware
    self.ResourceDirectory = !dir + path_sep() + $
        'resource' + path_sep()
    WindowModes = ['select', $
                   'rotate', $
                   'pan', $
                   'zoom']
    bitmapNames = ['arrow', $
                   'rotate', $
                   'hand', $
                   'zoom']
    Cursors = ['Arrow', $
               'Rotate', $
               'Pan', $
               'Zoom']
    self.pWindowModes = ptr_new(WindowModes, /NO_COPY)
    self.pBitmapFiles = ptr_new(bitmapNames, /NO_COPY)
    self.pCursors = ptr_new(Cursors, /NO_COPY)
    self.OpacityFunction = 'Linear (Increasing)'
    self.winPct = [0.5,0.5]
    self.winPctRange = [0.25,0.75]
    self.xy = [-1,-1]
    self -> RegisterCursors
    return, 1

end


;------------------------------------------------------------------------------
;+
; This method inverts the current color table being used in the displays
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::InvertColorTableEvent

    self.o3DDisplay -> GetProperty, COLOR_TABLE=ColorTable
    ColorTable = reverse(ColorTable,1)
    self.o3DDisplay -> SetProperty, COLOR_TABLE=ColorTable
    self.o3DDisplay -> RenderScene
    self.oXYZDisplay->GetProperty, COLOR_TABLE=ColorTable
    ColorTable = reverse(ColorTable,1)
    self.oXYZDisplay -> SetProperty, COLOR_TABLE=ColorTable
    self.oXYZDisplay -> RenderScene
    wButton = widget_info(self.tlb, FIND_BY_UNAME='menu_view_InvertColorTable')
    widget_control, wButton, SET_BUTTON=1-widget_info(wButton, /BUTTON_SET)

end


;------------------------------------------------------------------------------
;+
; Handle a help menu event.
;
; @param event {in}{type=menu event}{required}
;   The menu event
;
; @param parts {in}{type=string array}{required}
;   Menu item selected
;
; @history
;   June 2009 : RK ITT PSG : initial
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::HelpEvent, event, parts
    compile_opt idl2, logical_predicate

    case parts[0] of
        'about': void = dialog_message(TITLE='About 3D Viewer (' + MOD_DATE() + ')',  $
                          ['View and manipulate 3D molecule density volumes', '',     $
                           'Copyright 2007-2009, Howard Hughes Medical Institute'],   $
                          DIALOG_PARENT=self.tlb)
        else:  message, /CONTINUE, 'Unknown component: ' + parts[0]
    endcase

end


;------------------------------------------------------------------------------
;+
; This method handles events from button widgets in the main menu
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from the PALM_MainGUI widget.
;
; @param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::MenuEvent, event, components, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    wait, 0.3 ; Wait for the button to disappear
    case components[0] of
        'file': self -> FileEvent, event, components[1:*]
        'view': self -> ViewEvent, event, components[1:*]
        'help': self->HelpEvent, event, components[1:*]
        else: message, /CONTINUE, 'Unknown component: ' + components[0]
    endcase

end


;------------------------------------------------------------------------------
;+
; This method is called when the top level base is realized
;
; @param
;    tlb {in}{type=long}{required}
;      The widget ID of the top level base.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::Notify_Realize, tlb, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    self -> InitializeDisplays

; Show the widgets
    ss = get_screen_size()
    wGeom = widget_info(self.tlb, /GEOMETRY)

    widget_control, self.tlb, $
        XOFFSET=(ss[0]-wGeom.xSize)/2, $
        YOFFSET=(ss[1]-wGeom.ySize)/2

    if obj_valid(self.oModel) then $
        self -> UpdateModel

    self.o3DWindow -> SetCurrentCursor, (*self.pCursors)[0]
    XManager, 'PALM_MainGUI', self.TLB, $
        EVENT_HANDLER= 'GenericClassEvent', $
        /NO_BLOCK

end


;------------------------------------------------------------------------------
;+
; This method sends File->Output events to the proper method
;
; @param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::OutputEvent, components

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    case components[0] of
        'image': self -> OutputImageFile, components[1:*]
        'motion': self -> OutputMotionFile, components[1:*]
        else: message, /CONTINUE, 'Unknown component: ' + components[0]
    endcase

end


;------------------------------------------------------------------------------
;+
; This method handles image output events
;
; @param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::OutputImageFile, components
end


;------------------------------------------------------------------------------
;+
; This method handles image output events
;
; @param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::OutputMotionFile, components
end


;------------------------------------------------------------------------------
;+
; Get the parameters to use when generating the volume.
;-
;------------------------------------------------------------------------------
function PALM_MainGUI::GetZParameters
    compile_opt idl2, logical_predicate

    ;  CGroupParams indices
    params = [[19,20,40,21,22,41],  $  ;  Group Z
              [ 2, 3,34,16,17,35],  $  ;  Z
              [19,20,47,21,22,41],  $  ;  Unwrapped Group Z
              [ 2, 3,44,16,17,35]]     ;  Unwrapped Z
    idx = widget_info(self->Get('Z_PARAM'), /DROPLIST_SELECT)
    return, reform(params[*,idx])
end


;------------------------------------------------------------------------------
;+
; This method recalculates the volume using the current zoom level and
; volume control settings
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::RecalculateVolume

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if n_elements(tlb) GT 0 then $
            if widget_info(tlb, /VALID) then $
                widget_control, tlb, /DESTROY
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif
;
; Get the properties for generating the volume...
;
; Accumulation
    wDroplist = self->Get('Accumulation')
    widget_control, wDroplist, GET_VALUE=dlValues
    Accumulation = dlValues[widget_info(wDroplist, /DROPLIST_SELECT)]
; Compsite function
    CompositeFunction = widget_info(self->Get('CompositeFunction'), /DROPLIST_SELECT)
; Hue volume?
    UseHue = widget_info(self->Get('UseHue'), /BUTTON_SET)
; Use EDM
    UseEDM = widget_info(self->Get('UseEDM'), /BUTTON_SET)
; Use constant scaling
    UseConstant = widget_info(self->Get('UseConstant'), /BUTTON_SET)
; Use shear
    useShear = widget_info(self->Get('AnimateShearBase'), /MAP)
    shearAxis = 2
    ; Shear amount is a factor to be applied to the shear IDLgrModel's transformation axis.
    shearAmount = 0.
; Volume gamma
    widget_control, self->Get('GammaRender'), GET_VALUE=s
    gammaRender = fix(s[0])/100.0
; Brightness
    widget_control, self->Get('Brightness'), GET_VALUE=s
    brightness = double(s[0])/10.0
; Sub-volume width
    wDroplist = self->Get('SubvolumeWidth')
    widget_control, wDroplist, GET_VALUE=dlValues
    SubvolumeWidth = dlValues[widget_info(wDroplist, /DROPLIST_SELECT)]
; Volume dimensions
    widget_control, self->Get('VolumeDimensionsLabel'), GET_VALUE=MaxVolumeDimension
    MaxVolumeDimension = fix(MaxVolumeDimension)
; Z-Scale factor
    widget_control, self->Get('zScaleFactorLabel'), GET_VALUE=zScaleFactor
    zScaleFactor = float(zScaleFactor)
; Filters
    doAuto = 0
    doLog = 0
; Data range
    dataRange = [0,100.0d] ;dataRange[sort(dataRange)]
; Maximum range
    maxScale = 1.0
; Params and group params
    vParams = self->GetZParameters()
; Scale the data range
    if ~doLog then $
        dataRange[1] = dataRange[1]*maxScale
    doReset = 0
    self.o3DWindow -> GetProperty, GRAPHICS_TREE=oView
    oView -> GetProperty, VIEWPLANE_RECT=vpr
    names = ['PALM','EMVolume']
    dialogStr = ['Molecule','EM Volume']
    for i = 0, n_elements(names)-1 do begin

        obj = self.o3DDisplay->GetObjectByName(names[i])
        if names[i] EQ 'PALM' then begin
            obj -> GetProperty, COLOR_TABLE=ColorTable, $
                FUNCTION_INDEX=FunctionIndex, $
                HUE=hue, $
                EDM=edm, $
                NANOS_PER_CCD_PIXEL=Nanos, $
                PARAMETER_LIMITS=ParameterLimits, $
                VOLUME_OBJECT=oVolume, $
                VOLUME_XRANGE=VolxRange, $
                VOLUME_YRANGE=VolyRange, $
                VOLUME_ZRANGE=Volzrange, $
                XCOORD_CONV=xs, $
                X_RANGE=xRange, $
                YCOORD_CONV=ys, $
                Y_RANGE=yRange, $
                ZCOORD_CONV=zs, $
                Z_RANGE=zRange
            tm = oVolume->GetCTM()
            ;  If hue, fall back to the red temperature color table
            ;  This won't return to the last used but for now this is okay
            if (hue) then begin
                oPalette = obj_new('IDLgrPalette')
                oPalette->LoadCt, 3
                oPalette->GetProperty, RED=r, GREEN=g, BLUE=b
                obj_destroy, oPalette
                ColorTable = [[r],[g],[b]]
            endif
        endif else begin
            if ~obj_valid(obj) then $
                continue
            tm = obj->GetCTM()
            obj->GetProperty, $
                XRANGE=VolxRange, $
                YRANGE=VolyRange, $
                ZRANGE=VolzRange
        endelse
        volDims = long([VolxRange[1],VolyRange[1],VolzRange[1]])
    ;
    ; Check the outer corners first
    ;
        temp = [ $
                [0,0,0,1], $
                [0,volDims[1],0,1], $
                [0,volDims[1],volDims[2],1], $
                [0,0,volDims[2],1], $
                [volDims[0],0,0,1], $
                [volDims[0],volDims[1],0,1], $
                [volDims[0],volDims[1],volDims[2],1], $
                [volDims[0],0,volDims[2],1] $
               ]
        check = matrix_multiply(tm, temporary(temp), /ATRANSPOSE)
        indexGood = where( $
            (check[0,*] GT vpr[0]) AND $
            (check[0,*] LT (vpr[0]+vpr[2])) AND $
            (check[1,*] GT vpr[1]) AND $
            (check[1,*] LT (vpr[1]+vpr[3])), count, $
            NCOMPLEMENT=nComp)
        if nComp GT 3 then begin
        ;
        ; At least one of the corners lies outside the view...
        ; Need to check the limits
        ;
            tlb = PALM_Dialog( $
                TEXT='  *** Checking '+ dialogStr[i] +' Limits ***  ', $
                TITLE='Recalculating...')
            nVoxels = product(volDims, /PRESERVE_TYPE)
            nYZ = product(volDims[1:2], /PRESERVE_TYPE)
            temp = transpose([ $
                [lindgen(nVoxels) / nYZ], $
                [lindgen(nVoxels) mod nYZ / volDims[2]], $
                [lindgen(nVoxels) mod volDims[2]], $
                [replicate(1L, nVoxels)] $
                ])
            check = matrix_multiply(tm, temp, /ATRANSPOSE)
            indexGood = where( $
                (check[0,*] GT vpr[0]) AND $
                (check[0,*] LT (vpr[0]+vpr[2])) AND $
                (check[1,*] GT vpr[1]) AND $
                ((temporary(check))[1,*] LT (vpr[1]+vpr[3])), count, $
                NCOMPLEMENT=nComp)

            if count EQ 0 then begin
                widget_control, tlb, /DESTROY
                if names[i] EQ 'PALM' then begin
                    message, /CONTINUE, $
                        'No molecule voxels are within the view'
                    return
                endif else begin
                    obj_destroy, obj
                    continue
                endelse
            endif

            temp = temp[*,temporary(indexGood)]
        ; X-Range and Y-range in the volume
            minPoint = [ $
                        min(temp[0,*], MAX=maxX), $
                        min(temp[1,*], MAX=maxY), $
                        min((temporary(temp))[2,*], MAX=maxZ), $
                        1 $
                       ]
            maxPoint = [maxX,maxY,maxZ,1]
            if names[i] EQ 'PALM' then begin
            ; Normalize
                minPoint = minPoint / ([VolxRange[1],VolyRange[1]]-1)
                maxPoint = maxPoint / ([VolxRange[1],VolyRange[1]]-1)
            ; Map to data space
                dx = xRange[1]-xRange[0]
                dy = yRange[1]-yRange[0]
                minPoint = (minPoint * [dx,dy] + [xRange[0],yRange[0]]) / Nanos
                maxPoint = (maxPoint * [dx,dy] + [xRange[0],yRange[0]]) / Nanos
            ; Modify ParameterLimits
                ParameterLimits[[2,3],0] = minPoint
                ParameterLimits[[2,3],1] = maxPoint
                ParameterLimits[[2,3],3] = maxPoint-minPoint
                ParameterLimits[[2,3],2] = ParameterLimits[[2,3],3]/2.
                ParameterLimits[19,*] = ParameterLimits[2,*]
                ParameterLimits[20,*] = ParameterLimits[3,*]
            endif
            widget_control, tlb, /DESTROY
            if names[i] EQ 'EMVolume' then begin
                obj->GetProperty, DATA0=volData, $
                    OPACITY_TABLE0=OpacityTable
                volData = volData[ $
                    minPoint[0]:maxPoint[0], $
                    minPoint[1]:maxPoint[1], $
                    minPoint[2]:maxPoint[2] $
                    ]
                obj_destroy, obj
                self->LoadEMVolume, volData, /NO_DRAW, $
                    OPACITY_TABLE=OpacityTable
            endif
        endif

        if names[i] EQ 'PALM' then begin
            oModel = obj_new('PALMgr3DModel', $
                ACCUMULATION=Accumulation, $
                AUTO_FILTER=doAuto, $
                COLOR_TABLE=ColorTable, $
                COMPOSITE_FUNCTION=CompositeFunction, $
                DATA_RANGE=dataRange, $
                FILTER_INDEX=1, $
                FUNCTION_INDEX=FunctionIndex, $
                GAMMA_RENDER=gammaRender, $
                LOG_FILTER=doLog, $
                MAXIMUM_VOLUME_DIMENSION=MaxVolumeDimension, $
                NANOS_PER_CCD_PIXEL=Nanos, $
                PARAMETER_LIMITS=ParameterLimits, $
                SUBVOLUME_WIDTH=SubvolumeWidth, $
                USE_HUE=UseHue, $
                USE_EDM=UseEDM, $
                CONSTANT=UseConstant, $
                VERBOSE=self.verbose, $
                Z_SCALE_FACTOR=zScaleFactor,  $
                BRIGHTNESS=brightness,  $
                PARAMS=vParams, $
                GROUP_PARAMS=vGroupParams, $
                MAIN_GUI=self)
            if obj_valid(oModel) then begin
                self->UpdateModel, oModel
                doReset = 1
            endif
        endif else begin
        endelse
    endfor

    void = check_math()
    if doReset then $
        self->Reset, 'all'

end


;------------------------------------------------------------------------------
;+
; This method registers the cursors used
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::RegisterCursors, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

; Rotate
    strArray = [ $
        '       .        ', $
        '      .#.       ', $
        '     .##..      ', $
        '    .$####.     ', $
        '     .##..#.    ', $
        '      .#. .#.   ', $
        '       .   .#.  ', $
        '  .        .#.  ', $
        ' .#.       .#.  ', $
        ' .#.       .#.  ', $
        ' .#.       .#.  ', $
        '  .#.     .#.   ', $
        '   .#.....#.    ', $
        '    .#####.     ', $
        '     .....      ', $
        '                ']
    image = create_cursor(strArray, HOTSPOT=hotspot, MASK=mask)
    register_cursor, (*self.pCursors)[1], image, HOTSPOT=hotspot, MASK=mask
; Pan
    strArray = [ $
        '   .. .$#...    ', $
        '  .##.#..###.   ', $
        ' .#..##..#..#.. ', $
        ' .#..##..#..#.#.', $
        '  .#..#..#..##.#', $
        ' ..#..#..#..#..#', $
        '.##.#.......#..#', $
        '#..##..........#', $
        '#...#.........#.', $
        '.#............#.', $
        ' .#...........#.', $
        ' .#..........#. ', $
        '  .#.........#. ', $
        '   .#.......#.  ', $
        '    .#......#.  ', $
        '                ']
    image = create_cursor(strArray, HOTSPOT=hotspot, MASK=mask)
    register_cursor, (*self.pCursors)[2], image, HOTSPOT=hotspot, MASK=mask
; Zoom
    strArray = [ $
        '     .....      ', $
        '    .#####.     ', $
        '   .#.....#.    ', $
        '  .#.     .#.   ', $
        ' .#.       .#.  ', $
        ' .#.       .#.  ', $
        ' .#.   $   .#.  ', $
        ' .#.       .#.  ', $
        ' .#.       .#.  ', $
        '  .#..... .##.  ', $
        '   .#.....####. ', $
        '    .######..##.', $
        '     .... .#..#.', $
        '           .##. ', $
        '            ..  ', $
        '                ']
    image = create_cursor(strArray, HOTSPOT=hotspot, MASK=mask)
    register_cursor, (*self.pCursors)[3], image, HOTSPOT=hotspot, MASK=mask
end


;------------------------------------------------------------------------------
;+
; This method sends File->Output events to the proper method
;
; @param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::Reset, components

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    case strupcase(components[0]) of
        '3D': self.o3DDisplay -> Reset
        'ALL': begin
            self.oXYZDisplay -> Reset
            self.o3DDisplay -> Reset
        end
        'GRID': begin
            self -> SetProperty, GRID_PERCENT=[0.5,0.5]
            self -> BaseEvent
        end
        'XYZ': begin

            case components[1] of
                'scale/translate': noWindowLevel = 1
                'windowlevel': noX = (noY = (noZ = 1))
;
; Now that the windows zoom and pan together we do not want
; to only reset one view.
;
;                'X': noWindowLevel = (noY = (noZ = 1))
;                'Y': noWindowLevel = (noX = (noZ = 1))
;                'Z': noWindowLevel = (noX = (noY = 1))
                else:
            endcase

            self.oXYZDisplay -> Reset, $
                NO_WINDOWLEVEL=noWindowLevel, $
                NO_X=noX, $
                NO_Y=noY, $
                NO_Z=noZ
        end
        else: message, /CONTINUE, 'Unknown component: ' + components[0]
    endcase

end


;------------------------------------------------------------------------------
;+
; Rotates the volume in the 3-dimensional display
;
; @Param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::RotateVolume, components, $
    ERROR_MESSAGE=ErrMsg

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    widget_control, widget_info(self.tlb, FIND_BY_UNAME='incrementLabel'), $
        GET_VALUE=increment
    increment = float(increment)
    case components[0] of
        'X':
        'x-': self.o3DDisplay -> RotateVolume, -increment, /X
        'x+': self.o3DDisplay -> RotateVolume, increment, /X
        'x0': self.o3DDisplay -> RotateVolume, 0, /X, /ABSOLUTE
        'Y':
        'y-': self.o3DDisplay -> RotateVolume, -increment, /Y
        'y+': self.o3DDisplay -> RotateVolume, increment, /Y
        'y0': self.o3DDisplay -> RotateVolume, 0, /Y, /ABSOLUTE
        'Z':
        'z-': self.o3DDisplay -> RotateVolume, -increment, /Z
        'z+': self.o3DDisplay -> RotateVolume, increment, /Z
        'z0': self.o3DDisplay -> RotateVolume, 0, /Z, /ABSOLUTE
        else: message, /CONTINUE, 'Unknown component: ' + components[0]
    endcase

end


;------------------------------------------------------------------------------
;+
; For setting the properties of the object class
;
; @Keyword
;   BACKGROUND_COLOR {in}{optional}{type=bytarr}
;     Set this keyword to a 3-element byte array specifying the background
;     color for the displays
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
; @Keyword
;   OPACITY_FUNCTION {in}{optional}{type=string}
;     Set this keyword to a string specifying the name of the opacity
;     function used.  Acceptable values are: "Free hand", "Gaussian",
;     "Linear (Decreasing)", "Linear (Increasing)"
;
; @keyword REF_VOXEL_VOLUME {in}{optional}{double}
;     Reference voxel volume in nm^3.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::SetProperty, $
    BACKGROUND_COLOR=BackgroundColor, $
    ERROR_MESSAGE=errMsg, $
    GRID_PERCENT=GridPercent, $
    HIDE=doHide, $
    MODEL=oModel, $
    OPACITY_FUNCTION=OpacityFunction,  $
    MEAN_VOXEL_BRIGHTNESS=meanVoxBrightness

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    if (n_elements(meanVoxBrightness) ne 0) then begin
        self.meanvoxBrightness = meanVoxBrightness
    endif

    if n_elements(BackgroundColor) GT 2 then begin
        self.o3DDisplay -> SetProperty, BACKGROUND_COLOR=BackgroundColor[0:2]
        self.o3DDisplay -> RenderScene
        self.oXYZDisplay -> SetProperty, BACKGROUND_COLOR=BackgroundColor[0:2]
        self.oXYZDisplay -> RenderScene
    endif

    if n_elements(doHide) GT 0 then $
        widget_control, self.tlb, MAP=~keyword_set(doHide)

    nGrid = n_elements(gridPercent)
    case nGrid of
        0:
        1: self.winPct = replicate((gridPercent>self.winPctRange[0])<self.winPctRange[1],2)
        else: self.winPct = (gridPercent[0:1]>self.winPctRange[0])<self.winPctRange[1]
    endcase

    if n_elements(OpacityFunction) GT 0 then begin
        if size(OpacityFunction, /TYPE) NE 7 then $
            return
        self.OpacityFunction = OpacityFunction[0]
    endif

end


;------------------------------------------------------------------------------
;+
; Shears the volume in the 3-dimensional display
;
; @Param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   December, 2008 : Jim P., ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::ShearVolume, components, $
    ERROR_MESSAGE=ErrMsg

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    increment = .02
    increment = float(increment)
    case components[0] of
        'X':
        'x-': self.o3DDisplay -> ShearVolume, -increment, /X
        'x+': self.o3DDisplay -> ShearVolume, increment, /X
        'x0': self.o3DDisplay -> ShearVolume, 0, /X, /ABSOLUTE
        'Y':
        'y-': self.o3DDisplay -> ShearVolume, -increment, /Y
        'y+': self.o3DDisplay -> ShearVolume, increment, /Y
        'y0': self.o3DDisplay -> ShearVolume, 0, /Y, /ABSOLUTE
        'Z':
        'z-': self.o3DDisplay -> ShearVolume, -increment, /Z
        'z+': self.o3DDisplay -> ShearVolume, increment, /Z
        'z0': self.o3DDisplay -> ShearVolume, 0, /Z, /ABSOLUTE
        else: message, /CONTINUE, 'Unknown component: ' + components[0]
    endcase

end


;------------------------------------------------------------------------------
;+
; This method hides/shows the control panel
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::ShowControlPanel, $
    ERROR_MESSAGE=ErrMsg

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    wButton = self -> Get('menu_view_control')
    widget_control, wButton, SET_BUTTON=~widget_info(wButton, /BUTTON_SET)
    self -> BaseEvent

end


;--------------------------------------------------------------
;+
; This routine overwrites an MPR display with the given RGB image, used during
; the creation of a movie file.
;
; @param
;   xyz {in}{type=long}{required}
;       Flag indicating the plane represented by the RGB image, where
;       0 = x, 1 = y, 2 = z.
; @param
;   index {in}{type=long}{required}
;       Index into the volume represented by the slice, along the orthogonal
;       to xyz.
; @param
;   rgb {in}{type=bytarr(3,nx,ny)}{required}
;       The RGB image to be displayed in the appropriate MPR display window.
;
; @keyword UPDATE_3D {in}{type=boolean}{optional}
;   If set, update the 3D view
;-
;--------------------------------------------------------------
pro PALM_MainGUI::ShowPlane, xyz, index, rgb, UPDATE_3D=update
    compile_opt idl2

    case xyz of
        0 : x_location = index
        1 : y_location = index
        2 : z_location = index
    endcase

    if keyword_set(update) then begin
        self.o3DDisplay->UpdateSlicePlanes, X_LOCATION = x_location, $
            Y_LOCATION = y_location, Z_LOCATION = z_location
    endif

    self.oXYZDisplay->UpdateImages, X = xyz eq 0, Y = xyz eq 1, Z = xyz eq 2, $
        INPUT_INDEX = index
    self.oXYZDisplay->RenderScene
end


;------------------------------------------------------------------------------
;+
; Change the color of the hue indicator
;
; @param event {in}{type=structure}
;   Slider event structure
;
; @param draw_widget {in}{type=string}
;   Name of the draw widget to update
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::MultMolSliderEvent, event, draw_widget
    compile_opt idl2, logical_predicate

    ;  Set up the HSV values and map to RGB
    color_convert, event.value, 1.0, 1.0, r,g,b, /HSV_RGB
    color = ((long(b) > 0) < 255)*65536L + ((long(g) > 0) < 255)*256 + ((long(r) > 0) < 255)

    ;  Color the draw widget
    widget_control, self->Get('MM_' + draw_widget), GET_VALUE=id
    wset, id
    device, /DECOMPOSE
    plots, [0,100],[0,100], /DEVICE, COLOR=color, THICK=100
end


;------------------------------------------------------------------------------
;+
;  Process an EM slider event
;
;  @param event {in}{type=slider event}{required}
;    The slider event structure.
;
;  @param components {in}{type=string vector}{required}
;    Additional information from the UNAME.
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::EMSliderEvent, event, components
    compile_opt idl2, logical_predicate

    case components[0] of
        'alpha': begin
            wSlider = self->Get('EM_ALPHA')
            widget_control, wSlider, GET_VALUE=alpha
            mm = widget_info(wSlider, /SLIDER_MIN_MAX)
            alpha = (float(alpha)-mm[0])/(mm[1]-mm[0])
            self.o3DDisplay->SetProperty, EM_ALPHA_CHANNEL=alpha
            self.o3DDisplay->RenderScene
        end
        'imageOpacity': begin
            widget_control, self->Get('emOpacityLabel'), $
                SET_VALUE=strtrim(event.value,2)
            self.oXYZDisplay->SetProperty, EM_OPACITY=float(event.value)/100
            self.oXYZDisplay->RenderScene
        end
        else: message, /CONTINUE, 'Unknown component: ' + components[0]
    endcase
end


;------------------------------------------------------------------------------
;+
; This method handles events from slider widgets
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from a button on the PALM_MainGUI widget.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::SliderEvent, event, $
    ERROR_MESSAGE=ErrMsg

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    widget_control, event.id, GET_UVALUE=components
    components = strtok(components, '_', /EXTRACT)
    case components[0] of
        'control': self -> ControlEvent, event, components[1:*]
        'gamma' : self->GammaEvent, event
        'brightness': self->BrightnessEvent, event
        'mm': self->MultMolSliderEvent, event, components[1:*]
        'em': self->EMSliderEvent, event, components[1:*]
        else: message, /CONTINUE, 'Unknown component: ' + components[0]
    endcase

end


;------------------------------------------------------------------------------
;+
;  Update the brightness value
;
; @param event {in}{type=structure}
;   Slider event structure
;-
pro PALM_MainGUI::BrightnessEvent, event
    compile_opt idl2

    widget_control, event.id, GET_VALUE=v
    widget_control, self->Get('BrightnessValue'), SET_VALUE=string(v/10.0,FORMAT='(F4.1)')
end


;------------------------------------------------------------------------------
;+
;  Update the gamma value
;
; @param event {in}{type=structure}
;   Slider event structure
;-
pro PALM_MainGUI::GammaEvent, event
    compile_opt idl2

    widget_control, event.id, GET_VALUE=v
    widget_control, self->Get('GammaRenderValue'), SET_VALUE=string(v/100.0,FORMAT='(F4.1)')
end


;------------------------------------------------------------------------------
;+
; This method is called when a toolbar button is pressed
;
; @param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::ToolbarEvent, components, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    self.o3DWindowObserver -> SetWindowMode, strlowcase(components[0:*])
    cursorIndex = where(strupcase(*self.pWindowModes) EQ strupcase(components[0]), count)
    if count GT 0 then $
        self.o3DWindow -> SetCurrentCursor, (*self.pCursors)[cursorIndex] $
    else $
        print, 'Cursor not registered: ' + strupcase(components[0])

    self -> UpdateMenus, /DO_3D

end


;------------------------------------------------------------------------------
;+
; This method sets the text in the information label
;
; @Param
;    str {in}{optional}{type=string}
;      Set this parameter to a string to display on the information label
;
; @Keyword
;    APPEND {in}{optional}{type=boolean}
;      Set this keyword to have the string appended to the contents of the
;      label
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::UpdateInformationLabel, str, $
    APPEND=append, $
    REMOVE_PERCENT=RemovePercent

    newStr = n_elements(str) GT 0 ? str[0] : ''
    wLabel = widget_info(self.tlb, FIND_BY_UNAME='InfoLabel')
    if keyword_set(append) then begin
        widget_control, wLabel, GET_VALUE=tempStr
        if keyword_set(RemovePercent) then begin
            index = strpos(tempStr, '.', /REVERSE_SEARCH)
            if index GE 0 then $
                tempStr = strmid(tempStr, 0, index+1)
            newStr = tempStr+newStr
        endif
    endif

    widget_control, wLabel, SET_VALUE=newStr

end


;------------------------------------------------------------------------------
;+
; This method updates the XYZ context menus accoring to the current cursor mode
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @keyword DO_3D
;   Update 3D
;
; @keyword DO_TOOLBAR
;   Update toolbar
;
; @keyword DO_XYZ
;   Update XYZ
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::UpdateMenus, $
    DO_3D=do3D, $
    DO_TOOLBAR=doToolbar, $
    DO_XYZ=doXYZ

    if keyword_set(do3D) then begin
        wMode = self.o3DWindowObserver->GetWindowMode()
        wBase = self -> Get('PALM_3DWindowContextBase')
        wSibling = widget_info(wBase, /CHILD)
        repeat begin
            widget_control, wSibling, GET_UVALUE=uVal
            tok = strtok(uVal, '_', /EXTRACT)
            widget_control, wSibling, SET_BUTTON=(wMode EQ tok[1])
            wSibling = widget_info(wSibling, /SIBLING)
        endrep until (~widget_info(wSibling, /VALID_ID))
    endif

    if keyword_set(doToolbar) then begin
        wMode = self.o3DWindowObserver->GetWindowMode()
        wBase = self -> Get('toolbar_base')
        wSibling = widget_info(wBase, /CHILD)
        repeat begin
            widget_control, wSibling, GET_UVALUE=uVal
            tok = strtok(uVal, '_', /EXTRACT)
            widget_control, wSibling, SET_BUTTON=(wMode EQ tok[1])
            wSibling = widget_info(wSibling, /SIBLING)
        endrep until (~widget_info(wSibling, /VALID_ID))
    endif

    if keyword_set(doXYZ) then begin
        self.oXYZWindowObserver->GetProperty, CURSOR_MODE=cMode
        uNames = 'PALM_'+['X','Y','Z']+'WindowContextBase'
        for i = 0, n_elements(uNames)-1 do begin
            wBase = self -> Get(uNames[i])
            wSibling = widget_info(wBase, /CHILD)
            repeat begin
                widget_control, wSibling, GET_UVALUE=uVal
                tok = strtok(uVal, '_', /EXTRACT)
                widget_control, wSibling, SET_BUTTON=(cMode EQ tok[1])
                wSibling = widget_info(wSibling, /SIBLING)
            endrep until (~widget_info(wSibling, /VALID_ID))
        endfor
    endif

end


;------------------------------------------------------------------------------
;+
; This method adds a display model to the display destroying the
; existing model if present.
;
; @param
;    oModel {in}{type=object reference}{required}
;      A reference to a PALMgr3DModel object.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::UpdateModel, oModel, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    if n_elements(oModel) EQ 0 then $
        return
    if ~obj_valid(oModel) then $
        return
    if ~obj_isa(oModel, 'PALMgr3DModel') then $
        return

    oModelOld = self -> GetObjectByName('PALM')
    if obj_valid(oModelOld) then $
        obj_destroy, oModelOld

    self.o3DDisplay -> UpdateModel, oModel, /NO_DRAW
    oModel -> GetProperty, ACCUMULATION=Accumulation, $
        COLOR_TABLE=ColorTable, $
        HUE=hue, $
        FILTER_INDEX=FilterIndex
    if (hue NE 1) then begin
        self.oXYZDisplay -> SetProperty, COLOR_TABLE=ColorTable
        for i = 0, n_elements(uNames)-1 do begin
            widget_control, self->Get(uNames[i]), /SENSITIVE
        endfor
    endif
    widget_control, self->Get('UseHueMPR'), SENSITIVE=hue
    self.oXYZDisplay -> UpdateModel, oModel

    wDroplist = self -> Get('Accumulation')
    widget_control, wDropList, GET_VALUE=dlValues
    index = where(strupcase(dlValues) EQ strupcase(Accumulation), count)
    if count LT 0 then begin
        void = dialog_message(['Unknown accumulation setting', $
            'Setting to ' + dlValues[0]])
        index = 0
    endif

    widget_control, wDroplist, SET_DROPLIST_SELECT=index

;  Update the dimensions on the GUI
    oModel->GetProperty, VOLUME_XRANGE=x, VOLUME_YRANGE=y, VOLUME_ZRANGE=z
    s = strtrim(fix(x[1]),2)+'x'+strtrim(fix(y[1]),2)+'x'+strtrim(fix(z[1]),2)
    widget_control, self->Get('volume_dims'), SET_VALUE=s
end


;------------------------------------------------------------------------------
;+
; This method sets/unsets buttons on the same menu base.
;
; @Param
;   wID {in}{required}{type=long}
;     The widget ID of the button widget that is to be set.  All siblings
;     will be unset.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::UpdateSelection, wID

    wParent = widget_info(wID, /PARENT)
    wSiblings = widget_info(wParent, /ALL_CHILDREN)
    for i = 0, n_elements(wSiblings)-1 do $
        widget_control, wSiblings[i], SET_BUTTON=0
    widget_control, wID, /SET_BUTTON

end



;------------------------------------------------------------------------------
;+
; This method updates the slice planes in the 3D display
;
; @param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::UpdateSlicePlanes, components, $
    ERROR_MESSAGE=ErrMsg

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    case components[0] of
        'full': begin
            HideSliceLines = 0
            HideSlicePlanes = 0
        end
        'none': begin
            HideSliceLines = 1
            HideSlicePlanes = 1
        end
        'wire': begin
            HideSliceLines = 0
            HideSlicePlanes = 1
        end
        else: message, /CONTINUE, 'Unknown component: ' + components[0]
    endcase
    self.o3DDisplay -> SetProperty, HIDE_SLICE_LINES=HideSliceLines, $
        HIDE_SLICE_PLANES=HideSlicePlanes
    self.o3DDisplay -> RenderScene

end


;------------------------------------------------------------------------------
;+
; This method is called when a button under the View menu option is
; pressed
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from the PALM_MainGUI widget.
;
; @param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::ViewEvent, event, components, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    case components[0] of
        'background': begin
            case components[1] of
                'Black': self -> SetProperty, BACKGROUND_COLOR=[0,0,0]
                'White': self -> SetProperty, BACKGROUND_COLOR=[255,255,255]
                else:
            endcase
            self -> UpdateSelection, event.id
        end
        'ColorTable': self -> ColorTableEvent
        'control': begin
            widget_control, event.id, SET_BUTTON=~widget_info(event.id, /BUTTON_SET)
            self -> BaseEvent
        end
        'drag': begin
            self.o3DDisplay -> SetProperty, DRAG_QUALITY=components[1]
            self -> UpdateSelection, event.id
        end
        'InvertColorTable': self -> InvertColorTableEvent
        'reset': self -> Reset, components[1:*]
        'toolbar': begin
            widget_control, event.id, SET_BUTTON=~widget_info(event.id, /BUTTON_SET)
            self -> BaseEvent
        end
        else: message, /CONTINUE, 'Unknown component: ' + components[0]
    endcase

end


;------------------------------------------------------------------------------
;+
; This method is called when a button under the "View->Volume opacity..."
; menu option is pressed
;
; @param
;   tlb {in}{type=long}{required}
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::VolumeOpacityEvent, $
    EM_VOLUME=useEM, $
    ERROR_MESSAGE=errMsg

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    oVolumeOpacity = obj_new('PALM_VolumeOpacityDialog', self.tlb, self.o3DDisplay, $
        EM_VOLUME=useEM, $
        VERBOSE=self.verbose)
    obj_destroy, oVolumeOpacity

end


;------------------------------------------------------------------------------
;+
; This method handles events from the 3D-Display's conteext menu
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from a button on the PALM_MainGUI widget.
;
; @param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::VolMouseMenuEvent, event, components, $
    ERROR_MESSAGE=ErrMsg

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=errMsg
        return
    endif

    wait, 0.3 ; Wait for the button to disappear
    case components[0] of
        'Image': begin
            wParent = widget_info(widget_info(widget_info(widget_info(event.id, /PARENT), /PARENT), /PARENT), /PARENT)
            self.o3DDisplay -> ExportImage, $
                BMP=(components[1] EQ 'BMP'), $
                TIFF=(components[1] EQ 'TIFF')
        end
        'Motion': self.o3DDisplay -> Animate, /EXPORT_TO_MPEG
        'recalculate': self -> RecalculateVolume
        'Reset': self.o3DDisplay -> Reset
        'SlicePlanes': begin
            self -> UpdateSlicePlanes, components[1]
            self -> UpdateSelection, event.id
        end
        'VolumeOpacity': self -> VolumeOpacityEvent
        else: begin
            self.o3DWindowObserver -> SetWindowMode, components[0:*]
            cursorIndex = where(strupcase(*self.pWindowModes) EQ strupcase(components[0]), count)
            if count GT 0 then $
                self.o3DWindow -> SetCurrentCursor, (*self.pCursors)[cursorIndex] $
            else $
                print, 'Cursor not registered: ' + strupcase(components[0])
            self -> UpdateMenus, /DO_3D, /DO_TOOLBAR
        end
    endcase

end


;------------------------------------------------------------------------------
;+
; This method handles events from the XYZ Window's context menu
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from the PALM_MainGUI widget.
;
; @param
;   components {in}{type=strarr}{required}
;     Set this parameter to the decomposed UVALUE string
;     associated with the button, minus any prefix.  UVALUE
;     strings are decomposed on underscore delimiters.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI::XYZMouseMenuEvent, event, components, $
    ERROR_MESSAGE=ErrMsg

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=ErrMsg
        return
    endif

    if (widget_info(event.ID, /BUTTON_SET)) then $
        return

    wait, 0.3 ; wait for the button to disappear
    case components[0] of
        'Image': begin
            wParent = widget_info(widget_info(widget_info(widget_info(event.id, /PARENT), /PARENT), /PARENT), /PARENT)
            self.oXYZDisplay -> ExportImage, widget_info(wParent, /UNAME), $
                BMP=(components[1] EQ 'BMP'), $
                TIFF=(components[1] EQ 'TIFF')
        end
        'Motion': begin
            wParent = widget_info(widget_info(widget_info(widget_info(event.id, /PARENT), /PARENT), /PARENT), /PARENT)
            self.oXYZDisplay -> ExportMotion, widget_info(wParent, /UNAME), $
                MPEG=(components[1] EQ 'MPEG')
        end
        'Reset': self -> reset, components[1:*]
        else: begin
            self.oXYZWindowObserver->SetProperty, CURSOR_MODE=components[0]
            self -> UpdateMenus, /DO_XYZ
        end
    endcase

end


;------------------------------------------------------------------------------
;+
; Define the member variables of the PALM_MainGUI class, along with various
; internal structures.
;
; @File_comments
; This file defines the PALM_MainGUI class for the framework of the application.
;
; @field
;   o3DDisplay
;      A reference to an instance of the PALM_3DDisplay object
; @field
;   o3DWindow
;      A reference to an instance of the PALM_3DWindow object
; @field
;   o3DWindowObserver
;      A reference to an instance of the PALM_3DObserver object
; @field
;   pCursors
;      A pointer to a string array containing the cursor names.  Each entry
;      corresponds to the same entry in pWindowModes.
; @field
;   pWindowModes
;      A pointer to a string array containing the window manipulator modes.
;      Each entry corresponds to the same entry in pCursors.
; @field
;   renderer
;      An integer value indicating which graphics renderer to use when drawing
;      objects.  Valid values are:
;            0 : OpenGL
;            1 : Software
; @field
;   ResourceDirectory
;      A string indicating the location of the resource directory.
; @field
;   LastFileDirectory
;      The last directory in which a dialog_pickfile() was successful.
; @field
;   tlb
;      The widget ID of the top level base.
; @field
;   verbose
;      A boolean value that if set, will result in error messages being displayed
;      in the IDL ouput log.
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Global Services Group
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_MainGUI__Define

    void = {PALM_MainGUI, $
            o3DDisplay         : obj_new(), $
            o3DWindow          : obj_new(), $
            o3DWindowObserver  : obj_new(), $
            oModel             : obj_new(), $
            oXYZDisplay        : obj_new(), $
            oXYZWindow         : objarr(3), $
            oXYZWindowObserver : obj_new(), $
            OpacityFunction    : '',        $
            pBitmapFiles       : ptr_new(), $
            pCursors           : ptr_new(), $
            pWindowModes       : ptr_new(), $
            renderer           : 0,         $
            ResourceDirectory  : '',        $
            LastFileDirectory  : '',        $
            tlb                : 0L,        $
            verbose            : 0B,        $
            winPct             : fltarr(2), $
            winPctRange        : fltarr(2), $
            xy                 : intarr(2), $
            standalone         : 0b,        $
            haveEMvolume       : 0b,        $
            meanVoxBrightness  : 0.0d,      $
            hide               : bytarr(3)  $
           }

end
