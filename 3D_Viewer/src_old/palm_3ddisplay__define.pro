;$File: //depot/gsg/HHMI/Phase2/src/palm_3ddisplay__define.pro $
;$Revision: #37 $
;$Change: 145965 $
;$DateTime: 2009/11/12 14:20:28 $
;$Author: datencio $
;
;-------------------------------------------------------------------------------------
;-------------------------------------------------------------------------------------
function PALM_3DDisplay::AddEMVolume, volData, $
    ALPHA_CHANNEL=AlphaChannel, $
    COLOR_TABLE=ColorTable, $
    HIDE=hide, $
    NO_DRAW=noDraw, $
    OPACITY_TABLE=OpacityTable

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return, 0
    endif

    dims = size(volData)
    if dims[0] NE 3 then begin
        return, 0
    endif

    (self->GetObjectByName('Volume'))->GetProperty, $
        XCOORD_CONV=xs, $
        XRANGE=xRangeMolecule, $
        YCOORD_CONV=ys, $
        YRANGE=yRangeMolecule, $
        ZCOORD_CONV=zs, $
        ZRANGE=zRangeMolecule

    oVolumeEM = self->GetObjectByName('EMVolume')
    if ~obj_valid(oVolumeEM) then begin
        oVolumeEM = obj_new('IDLgrVolume', NAME='EMVolume', /INTERPOLATE, HINTS=2)
        (self->GetObjectByName('VolumeShearModel'))->Add, oVolumeEM, POSITION=0
    endif
    oVolumeEM->SetProperty, DATA0=bytscl(volData)
;
; Set the volume properties
;
    self->SetProperty, $
        EM_ALPHA_CHANNEL=AlphaChannel, $
        EM_OPACITY_TABLE=OpacityTable, $
        EM_VOLUME_HIDE=hide
;
; Scale the volume
;
    oVolumeEM -> GetProperty, XRANGE=xRange, $
                              YRANGE=yRange, $
                              ZRANGE=zRange
    xs = [-xRange[0],1.0]/(xRange[1]-xRange[0]) * $
        xs[1]*xRangeMolecule[1]+[xs[0],0]
    ys = [-yRange[0],1.0]/(yRange[1]-yRange[0]) * $
        ys[1]*yRangeMolecule[1]+[ys[0],0]
    zs = [-zRange[0],1.0]/(zRange[1]-zRange[0]) * $
        zs[1]*zRangeMolecule[1]+[zs[0],0]
    oVolumeEM->SetProperty, XCOORD_CONV=xs, YCOORD_CONV=ys, ZCOORD_CONV=zs

    if ~keyword_set(noDraw) then $
        self->RenderScene

    return, 1


end

;-------------------------------------------------------------------------------------
;+
; This method animates the contents of the 3D display.
;
; @history
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @keyword EXPORT_TO_MPEG {in}{type=boolean}{optional}
;   If set, export the animation to a movie file.
;
; @keyword NO_DRAW {in}{type=boolean}{optional}
;   If set, do not display updates on the screen.
;
;-
;-------------------------------------------------------------------------------------
pro PALM_3DDisplay::Animate, $
    EXPORT_TO_MPEG=mpeg, $
    NO_DRAW=noDraw

    compile_opt StrictArr
    on_error, 2
    cd, CURRENT=savedPath
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if obj_valid(oMPEG) then $
            obj_destroy, oMPEG
        if n_elements(viewProjection) GT 0 then begin
            self->IDLgrView::SetProperty, PROJECTION=viewProjection
            self->RenderScene
        endif
        help, /LAST
        return
    endif

    doMPEG = keyword_set(mpeg)
    if doMPEG then begin
        ;  Use FFMPEG for movie output
        if (widget_info(self.oMainGUI->Get('UseFFMPEG'), /BUTTON_SET)) then begin
            file = dialog_pickfile(TITLE='Select an output file...', GET_PATH=path, $
                /OVERWRITE, PATH=self.LastFileDirectory)
            if (file eq '') then return
            self.LastFileDirectory = path
            cd, path
            self->ExportUsingFFMPEG, file, noDraw
            cd, savedPath
            return
        endif

        ;  Old IDL MPEG output
        if ~(self->GetExportFile(EXTENSION='mpg', OUTPUT_FILE=file)) then begin
            cd, savedPath
            return
        endif
        oMPEG = obj_new('IDLgrMPEG', $
            FILE=file, $
            IFRAME_GAP=1, $
            QUALITY=100)
        self.oMainGUI -> UpdateInformationLabel, 'Capturing MPEG frames'
    endif else $
        self.oMainGUI -> UpdateInformationLabel, 'Animating'

    widget_control, /HOURGLASS
    self.oMainGUI -> GetAnimationSettings, $
        ROTATION_ANGLE=angle, ROTATION_AXIS=rotationAxis, $
        SHEAR_SCALE=shearScale, SHEAR_AXIS=shearAxis, SHEAR_MODE=shearMode, $
        ANIMATION_TYPE=animationType, $
        ROTATION_INCREMENT=rotInc
    rotInc = float(rotInc)
    self.oMainGUI -> GetProperty, DISPLAY_XYZ=oXYZDisplay

    maxDots = 50
    case animationType of
        'Rotation' : begin
            n = round(angle/rotInc)
            rotVal = [ $
                replicate(-rotInc, n),   $
                replicate( rotInc, n*2), $
                replicate(-rotInc, n)    $
                ]
            nIter = n_elements(rotVal)
        end
        'Shear' : begin
            nIter = round(shearScale/.05*8)
            shearValue = sin(!pi*4/(nIter - 1)*findgen(nIter))*shearScale
            if (shearMode eq 2) then begin
                ; make exception for circular pattern
                shearValue = replicate(shearScale, nIter)
                shearPhase = findgen(nIter) * (4 * !pi / (nIter - 1))
            endif else shearPhase = fltarr(nIter)
        end
        'Flythrough': begin
        ;
        ; Validate the flythrough data
        ;
            void = self->ReadFlythroughFile( $
                ANGLE=angle, $
                NUM_ITERATIONS=nIter, $
                POSITION=position)
            if ~void then $
                return
        ;
        ; Change the view projection
        ; Record the current transformation matrices
        ; Reset the transformation matrices
        ;
            self->IDLgrView::GetProperty, $
                EYE=viewEye, $
                PROJECTION=ViewProjection, $
                VIEWPLANE_RECT=vpr, $
                ZCLIP=ViewZClip
            self->IDLgrView::SetProperty, $
                EYE=1.1, $
                PROJECTION=2, $
                ZCLIP=[1,-1]
            oModelAxes = self->GetObjectByName('AxesModel')
            oModelAxes->GetProperty, TRANSFORM=tmAxes
            oModelRotate = self->GetObjectByName('VolumeRotate')
            oModelRotate->GetProperty, TRANSFORM=tmRotate
            oModelTranslate = self->GetObjectByName('VolumeTranslate')
            oModelTranslate->GetProperty, TRANSFORM=tmTranslate
            oModelZoom = self->GetObjectByName('VolumeModel')
            oModelZoom->GetProperty, TRANSFORM=tmZoom
            self->Reset, /NO_DRAW
        end
        else :
    endcase
    addVal = nIter/maxDots

    oStatusBar = obj_new('PALM_StatusBar', $
        COLOR=[0,111,0], $
        TEXT_COLOR=[255,255,255], $
        TITLE='Generating frames...', $
        TOP_LEVEL_BASE=tlb)
    for i = 0LL, nIter-1 do begin
        event = widget_event(tlb, /NOWAIT)
        if tag_names(event, /STRUCT) NE 'WIDGET_NOEVENT' then begin
        ;
        ; The cancel button was pressed.  Set back to the original
        ; settings if necessary
        ;
            self.oMainGUI->UpdateInformationLabel, ' '
            obj_destroy, oStatusBar
            cd, savedPath
            case animationType of
                'Shear': begin
                    self->ShearVolume, 0, $
                        NO_DRAW=noDraw, MODE=shearMode, $
                        SHEAR_PHASE=0, $
                        X=(shearAxis EQ 'X'), $
                        Y=(shearAxis EQ 'Y'), $
                        Z=(shearAxis EQ 'Z'), /ABSOLUTE
                end
                'Flythrough': begin
                    oModelAxes->SetProperty, TRANSFORM=tmAxes
                    oModelRotate->SetProperty, TRANSFORM=tmRotate
                    oModelTranslate->SetProperty, TRANSFORM=tmTranslate
                    oModelZoom->SetProperty, TRANSFORM=tmZoom
                    self->IDLgrView::SetProperty, $
                        EYE=viewEye, $
                        PROJECTION=ViewProjection, $
                        VIEWPLANE_RECT=vpr, $
                        ZCLIP=ViewZClip
                    self->RenderScene
                end
                else:
            endcase
            return
        endif

        case animationType of
            'Rotation' : begin
                self -> RotateVolume, rotVal[i], $
                    NO_DRAW=noDraw, $
                    X=(rotationAxis EQ 'X'), $
                    Y=(rotationAxis EQ 'Y'), $
                    Z=(rotationAxis EQ 'Z')
            end
            'Shear' : begin
                self -> ShearVolume, shearValue[i], $
                    NO_DRAW=noDraw, MODE=shearMode, $
                    SHEAR_PHASE=shearPhase[i], $
                    X=(shearAxis EQ 'X'), $
                    Y=(shearAxis EQ 'Y'), $
                    Z=(shearAxis EQ 'Z'), /ABSOLUTE
            end
            'Flythrough': begin
                self->Flythrough, position[*,i], angle[*,i], $
                    NO_DRAW=noDraw
            end
            else:
        endcase
        if doMPEG then $
            oMPEG -> Put, Reverse(self->GetImage(), 3)
        if ~(i MOD addVal) then begin
            self.oMainGUI -> UpdateInformationLabel, /APPEND, /REMOVE_PERCENT, $
                '. <'+strtrim(100*i/nIter,2)+'%>'
        endif
        oXYZDisplay -> RenderScene
        oStatusBar->UpdateStatus, float(i)/nIter
    endfor

    if (animationType EQ 'Flythrough') then begin
        oModelAxes->SetProperty, TRANSFORM=tmAxes
        oModelRotate->SetProperty, TRANSFORM=tmRotate
        oModelTranslate->SetProperty, TRANSFORM=tmTranslate
        oModelZoom->SetProperty, TRANSFORM=tmZoom
        self->IDLgrView::SetProperty, $
            EYE=viewEye, $
            PROJECTION=ViewProjection, $
            VIEWPLANE_RECT=vpr, $
            ZCLIP=ViewZClip
        self->RenderScene
    endif

    if (obj_valid(oStatusBar)) then  $
        obj_destroy, oStatusBar

    if doMPEG then begin
        self.oMainGUI -> UpdateInformationLabel, 'Saving', /APPEND, /REMOVE_PERCENT
        oMPEG -> Save
        obj_destroy, oMPEG
    endif else $
        self.oMainGUI -> UpdateInformationLabel, 'Finished', /APPEND, /REMOVE_PERCENT

    self.oMainGUI -> UpdateInformationLabel, ' '
    cd, savedPath
    if (animationType eq 'Shear' && shearMode eq 2) then begin
        ; Set back to original settings
        self->ShearVolume, 0, $
            NO_DRAW=noDraw, MODE=shearMode, $
            SHEAR_PHASE=0, $
            X=(shearAxis EQ 'X'), $
            Y=(shearAxis EQ 'Y'), $
            Z=(shearAxis EQ 'Z'), /ABSOLUTE
    endif

end


;------------------------------------------------------------------------------
;+
;  Export using Motion JPEG 2k.
;
;  @param file {in}{type=string}{required}
;    Pathname for the output file.
;
;  @param noDraw {in}{type=boolean}{required}
;    If set, do not animate while creating the output.
;
;-
pro PALM_3DDisplay::ExportUsingMotionJPEG2k, file, noDraw
    compile_opt idl2

    widget_control, /HOURGLASS
    self.oMainGUI -> GetAnimationSettings, ANGLE=angle, AXIS=axis,  $
        ROTATION_INCREMENT=rotInc
    rotInc = float(rotInc)
    self.oMainGUI -> GetProperty, DISPLAY_XYZ=oXYZDisplay

    maxDots = 50

    n = round(angle/rotInc)
    rotVal = [ $
        replicate(-rotInc, n),   $
        replicate( rotInc, n*2), $
        replicate(-rotInc, n)    $
        ]
    nIter = n_elements(rotVal)
    addVal = nIter/maxDots

    for i = 0LL, nIter-1 do begin

        self -> RotateVolume, rotVal[i], $
            NO_DRAW=noDraw, $
            X=(axis EQ 'X'), $
            Y=(axis EQ 'Y'), $
            Z=(axis EQ 'Z')

        ;  Write the frame to disk
        write_png, 'frame_'+string(i,FORMAT='(I04)')+'.png', self->GetImage()

        ;  Update the status bar
        if ~(i MOD addVal) then begin
            self.oMainGUI -> UpdateInformationLabel, /APPEND, /REMOVE_PERCENT, $
                '. <'+strtrim(100*i/nIter,2)+'%>'
        endif
        oXYZDisplay -> RenderScene
    endfor

    ;  Form the frames into the desired movie...

    self.oMainGUI -> UpdateInformationLabel, ' '
end


;------------------------------------------------------------------------------
;+
;  Export using FFMPEG.
;
;  @param file {in}{type=string}{required}
;    Pathname of the output movie.
;
;  @param noDraw {in}{type=boolean}{required}
;    If set, do not animate while creating the movie output.
;-
pro PALM_3DDisplay::ExportUsingFFMPEG, file, noDraw
    compile_opt idl2
    cd, CURRENT = savedPath
    ; Don't leak heap if we hit an unexpected error.
    catch, errorNumber
    if (errorNumber ne 0) then begin
        catch, /cancel
        if (n_elements(oMJ2K) ne 0) then begin
            obj_destroy, oMJ2K
        endif
        cd, savedPath
        message, /reissue_last
    endif
    ;  If file ends in ".mj2" then make a motion JPEG 2k output file
    ;  with IDL, do not use ffmpeg
    fileType = strtok(file, '.', /EXTRACT, COUNT=nComponents)
    if (nComponents lt 2) then Begin
        void = dialog_message('Need a file name extension')
        cd, savedPath
        return
    endif
    fileType = strlowcase(fileType[nComponents - 1])

    widget_control, /HOURGLASS
    self.oMainGUI -> GetAnimationSettings, ROTATION_ANGLE=angle, ROTATION_AXIS=rotationAxis, $
        SHEAR_SCALE=shearScale, SHEAR_AXIS=shearAxis, SHEAR_MODE=shearMode, $
        ANIMATION_TYPE=animationType,          $
        ROTATION_INCREMENT=rotInc
    rotInc = float(rotInc)
    self.oMainGUI -> GetProperty, DISPLAY_XYZ=oXYZDisplay

    maxDots = 50
    case animationType of
        'Rotation' : begin
            n = round(angle/rotInc)
            rotVal = [ $
                replicate(-rotInc, n),   $
                replicate( rotInc, n*2), $
                replicate(-rotInc, n)    $
                ]
            nIter = n_elements(rotVal)
        end
        'Shear' : begin
            nIter = round(shearScale/.05*2)
            shearValue = sin(!pi*2/(nIter - 1)*findgen(nIter))*shearScale
            if (shearMode eq 2) then begin
                ; make exception for circular pattern
                shearValue = replicate(shearScale, nIter)
                shearPhase = findgen(nIter) * (2 * !pi / (nIter - 1))
            endif else shearPhase = fltarr(nIter)
        end
        'Flythrough': begin
        ;
        ; Validate the flythrough data
        ;
            void = self->ReadFlythroughFile( $
                ANGLE=angle, $
                NUM_ITERATIONS=nIter, $
                POSITION=position)
            if ~void then $
                return
        ;
        ; Change the view projection
        ; Record the current transformation matrices
        ; Reset the transformation matrices
        ;
            self->IDLgrView::GetProperty, $
                EYE=viewEye, $
                PROJECTION=ViewProjection, $
                VIEWPLANE_RECT=vpr, $
                ZCLIP=ViewZClip
            self->IDLgrView::SetProperty, $
                EYE=1.1, $
                PROJECTION=2, $
                ZCLIP=[1,-1]
            oModelRotate = self->GetObjectByname('VolumeRotate')
            oModelRotate->GetProperty, TRANSFORM=tmRotate
            oModelTranslate = self->GetObjectByName('VolumeTranslate')
            oModelTranslate->GetProperty, TRANSFORM=tmTranslate
            oModelZoom = self->GetObjectByName('VolumeModel')
            oModelZoom->GetProperty, TRANSFORM=tmZoom
            self->Reset, /NO_DRAW
        end

        else :
    endcase
    addVal = nIter/maxDots

    ;  Form the frames into the desired movie
    self.oMainGUI->UpdateInformationLabel, ' creating movie', /APPEND, /REMOVE_PERCENT
    n = widget_info(self.oMainGUI->Get('FFMPEG_FPS'), /DROPLIST_SELECT)
    fps = strtrim(([5,10,15,20,25,30])[n], 2)

    oStatusBar = obj_new('PALM_StatusBar', $
        COLOR=[0,111,0], $
        TEXT_COLOR=[255,255,255], $
        TITLE='Generating frames...', $
        TOP_LEVEL_BASE=tlb)

    for i = 0LL, nIter-1 do begin

        event = widget_event(tlb, /NOWAIT)
        if tag_names(event, /STRUCT) NE 'WIDGET_NOEVENT' then begin
            self.oMainGUI->UpdateInformationLabel, ' '
            if ~widget_info(self.oMainGUI->Get('FFMPEG_KEEP_FRAMES'), /BUTTON_SET) then begin
                for i=0LL, nIter-1 do begin
                    file_delete, 'frame_'+string(i,FORMAT='(I04)')+'.png', /ALLOW_NONEXISTENT
                endfor
            endif
            obj_destroy, oStatusBar
            case AnimationType of
                'Shear': begin
                    self->ShearVolume, 0, $
                        NO_DRAW=noDraw, MODE=shearMode, $
                        SHEAR_PHASE=0, $
                        X=(shearAxis EQ 'X'), $
                        Y=(shearAxis EQ 'Y'), $
                        Z=(shearAxis EQ 'Z'), /ABSOLUTE
                end
                'Flythrough': begin
                    oModelRotate->SetProperty, TRANSFORM=tmRotate
                    oModelTranslate->SetProperty, TRANSFORM=tmTranslate
                    oModelZoom->SetProperty, TRANSFORM=tmZoom
                    self->IDLgrView::SetProperty, $
                        EYE=viewEye, $
                        PROJECTION=ViewProjection, $
                        VIEWPLANE_RECT=vpr, $
                        ZCLIP=ViewZClip
                    self->RenderScene
                end
                else:
            endcase

            if obj_valid(oMJ2K) then obj_destroy, oMJ2K
            cd, savedPath
            return
        endif

       case animationType of
            'Rotation' : begin
                self -> RotateVolume, rotVal[i], $
                NO_DRAW=noDraw, $
                X=(rotationAxis EQ 'X'), $
                Y=(rotationAxis EQ 'Y'), $
                Z=(rotationAxis EQ 'Z')
            end
            'Shear' : begin
                self -> ShearVolume, shearValue[i], $
                    NO_DRAW=noDraw, MODE=shearMode, $
                    SHEAR_PHASE=shearPhase[i], $
                    X=(shearAxis EQ 'X'), $
                    Y=(shearAxis EQ 'Y'), $
                    Z=(shearAxis EQ 'Z'), /ABSOLUTE
            end
            'Flythrough': begin
                self->Flythrough, position[*,i], angle[*,i], $
                    /DEGREES, NO_DRAW=noDraw
            end
            else :
        endcase

        switch fileType of
            'mj2' : begin
                if (i eq 0) then begin
                    ; Create the MJ2K file object on the first frame.
                    tickspersecond = 30000.
                    oMJ2K = obj_new('idlffmjpeg2000', file, /WRITE, /REVERSIBLE, $
                        FRAME_PERIOD = tickspersecond/fps, TIMESCALE = tickspersecond)
                endif
                R = oMJ2K->SetData(self->GetImage())
                break
            end
            else : begin
                ;  Write the frame to disk
                write_png, 'frame_'+string(i,FORMAT='(I04)')+'.png', self->GetImage()
            end
        endswitch

        ;  Update the status bar
        if ~(i MOD (addVal>1)) then begin
            self.oMainGUI -> UpdateInformationLabel, /APPEND, /REMOVE_PERCENT, $
                '. <'+strtrim(100*i/nIter,2)+'%>'
        endif
        oXYZDisplay -> RenderScene
        oStatusBar->UpdateStatus, float(i)/nIter
    endfor

    if (obj_valid(oStatusBar)) then  $
        obj_destroy, oStatusBar

    switch fileType of
        'mj2' : begin
            status = oMJ2K->Commit(10000L)
            if (status eq 0) then begin
                message, 'Error writing MJPEG2000 file to disk.', /TRACEBACK
            endif
            obj_destroy, oMJ2K
            cd, savedPath
            self.oMainGUI -> UpdateInformationLabel, ' '
            return
            break
         end
         else : begin
            break
         end
    endswitch

    ; For the purposes of testing, I look in the source path directory for the ffmpeg executable first.
    ffmpegexecutable = filepath('ffmpeg*', ROOT = sourcepath())
    mpegexecutable = file_search(ffmpegexecutable, COUNT=count)

    ; If I don't see the executable here, then I assume it's in PATH.
    ffmpeg = count eq 1 ? $
        '"' + mpegexecutable[0] + '"': $
        'ffmpeg'

    ; Write to a temp file, then rename it to override problems ffmpeg has with network drive paths.
    filebase = file_basename(file)
    tempfile = filepath(filebase, /TMP)
    if (file_test(tempfile)) then begin
        file_delete, tempfile
    endif

    ;  Get the file extension
    t = strsplit(file, '.', /EXTRACT)
    ext = t[n_elements(t)-1]
    if (ext eq 'mpg') then begin
        n = fix(fps)
        if (n lt 20) then begin
            fps = '20'
        endif else begin
            fps = '30'
        endelse

        command = ffmpeg + ' -qscale 5 -y -r ' + fps + ' -i frame_%04d.png ' + '"' + tempfile + '"'
    endif else begin
        command = ffmpeg + ' -qscale 5 -y -r ' + fps + ' -b 3600 -i frame_%04d.png ' + '"' + tempfile + '"'
    endelse
    if (!version.os_family eq 'Windows') then Begin
        spawn, command, output, error, /HIDE, /NOSHELL
    endif else begin
        spawn, command, output, error
    endelse
    ;  Clean up the frames
    if ~widget_info(self.oMainGUI->Get('FFMPEG_KEEP_FRAMES'), /BUTTON_SET) then begin
        self.oMainGUI->UpdateInformationLabel, 'Cleaning up frames', /APPEND, /REMOVE_PERCENT
        for i=0LL, nIter-1 do begin
            file_delete, 'frame_'+string(i,FORMAT='(I04)')+'.png', /ALLOW_NONEXISTENT
        endfor
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
    cd, savedPath
    self.oMainGUI -> UpdateInformationLabel, ' '
end


;------------------------------------------------------------------------------
;+
; This lifecycle method is called during the destruction of the PALM_3DDisplay
; object via OBJ_DESTROY.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::Cleanup
    obj_destroy, [self.oAxes, self.oBuffer, self.oText, self.oTrack]
    self -> IDLgrView::Cleanup
end

;-------------------------------------------------------------------------------------
;+
;-
;-------------------------------------------------------------------------------------
pro PALM_3DDisplay::ClearEMVolume

    compile_opt idl2
    on_error, 2
;
; Show the molecules
;
    widget_control, self.oMainGUI->Get('EM_SHOW_MOL'), /SET_BUTTON
    self->SetProperty, HIDE_VOLUME=0
;
; Remove the EM volume
;    
    oVolume = self->GetObjectByName('EMVolume')
    if ~obj_valid(oVolume) then $
        return
    obj_destroy, oVolume
    self->RenderScene

end

;-------------------------------------------------------------------------------------
;+
; This method exports the contents of the 3D window to an image file.
;
; @Keyword
;    BMP {in}{optional}{type=boolean}
;      Set this keyword to have the image written to a BMP file
;
; @Keyword
;    TIFF {in}{optional}{type=boolean}
;      Set this keyword to have the image written to a TIFF file
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_3DDisplay::ExportImage, $
    BMP=BMP, $
    TIFF=TIFF

    compile_opt StrictArr
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

; Get the image
    wait, 0.5 ; Pause to let the window clear
    oWindow = self.oMainGUI -> GetObjectByName('3DWindow')
    image = self -> GetImage()

    if keyword_set(BMP) then begin
        if self -> GetExportFile(EXTENSION='bmp', OUTPUT_FILE=file) then begin
            write_bmp, file, image, /RGB
        endif
    endif

    if keyword_set(TIFF) then begin
        if self -> GetExportFile(EXTENSION='tiff', OUTPUT_FILE=file) then begin
            write_tiff, file, reverse(image, 3)
        endif
    endif

end

;-------------------------------------------------------------------------------------
;+
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Global Services Group
;
; @History
;   June, 2009 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_3dDisplay::FlyThrough, position, angle, $
    NO_DRAW=noDraw

    compile_opt idl2
    on_error, 2
    if n_params() LT 2 then $
        return

    self->IDLgrView::GetProperty, EYE=eye, $
        VIEWPLANE_RECT=vpr
;
; Rotation
;
    tX = identity(4)
    tx[[1,2],[1,2]] = cos(-angle[0])
    tX[1,2] = sin(-angle[0])
    tX[2,1] = -tX[1,2]
    tY = identity(4)
    tY[[0,2],[0,2]] = cos(angle[1])
    tY[2,0] = sin(angle[1])
    tY[0,2] = -tY[2,0]
    tZ = identity(4)
    tZ[[0,1],[0,1]] = cos(angle[2])
    tZ[0,1] = sin(angle[2])
    tZ[1,0] = -tZ[0,1]
    tm = tY ## tX ## tZ
;
; Translation
;
    oPALM = self->GetObjectByname('PALM')
    oPALM->GetProperty, $
        VOLUME_XRANGE=xRangeVol, $
        VOLUME_YRANGE=yRangeVol, $
        VOLUME_ZRANGE=zRangeVol, $
        X_RANGE=xRange, $
        XCOORD_CONV=xs, $
        Y_RANGE=yRange, $
        YCOORD_CONV=ys, $
        Z_RANGE=zRange, $
        ZCOORD_CONV=zs
    x = xs[0]+xs[1]*(position[0]-xRange[0])/(xRange[1]-xRange[0])*xRangeVol[1]
    y = ys[0]+ys[1]*(position[1]-yRange[0])/(yRange[1]-yRange[0])*yRangeVol[1]
    z = zs[0]+zs[1]*(position[2]-zRange[0])/(zRange[1]-zRange[0])*zRangeVol[1]
    tmTranslation = identity(4)
    tmTranslation[3,[0,1,2]] = -[x,y,z]
    tm = tm ## tmTranslation
    tm[3,2] += eye

    (self->GetObjectByName('VolumeRotate'))->SetProperty, TRANSFORM=tm

    oModel = self->GetObjectByName('AxesModel')
    oModel->GetProperty, TRANSFORM=tmAxes
    tmAxes[0:2,0:2] = tm[0:2,0:2]
    oModel->SetProperty, TRANSFORM=tmAxes

    if ~keyword_set(noDraw) then begin
        self -> RenderScene
    endif

end

;-------------------------------------------------------------------------------------
;+
; This method prompts the user for an output file.
;
; @returns False (0) if the user cancels, otherwise true (1).
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @keyword EXTENSION {in}{type=string}{required}
;   The extension for filtering the files shown.
;
; @keyword OUTPUT_FILE {out}{type=string}{required}
;   The pathname of the selected output file.
;-
;-------------------------------------------------------------------------------------
function PALM_3DDisplay::GetExportFile, $
    EXTENSION=extension, $
    OUTPUT_FILE=file

    compile_opt StrictArr
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return, 0
    endif

    file = dialog_pickfile( $
        DEFAULT_EXTENSION=extension, $
        FILTER='*.'+extension, $
        TITLE='Select an output file', $
        /OVERWRITE $
        )
    if file EQ '' then $
        return, 0

    if file_test(file) then begin
        void = dialog_message(/QUESTION, $
            ['File already exists: '+file, $
             'Would you like to overwrite this file?'])
        if strupcase(void) EQ 'NO' then $
            return, 0
    endif

    return, 1

end


;-------------------------------------------------------------------------------------
;+
; This method captures an image from the specified window index
;
; @returns An image of the current contents of the 3D window.
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @keyword REVERSE {in}{type=scalar integer}{optional}
;   The axis number (1-based) to reverse in the image.
;-
;-------------------------------------------------------------------------------------
function PALM_3DDisplay::GetImage, $
    REVERSE=reverse

    if ~obj_valid(self.oBuffer) then begin
        self.oBuffer = obj_new('IDLgrBuffer')
        self.oBuffer -> SetProperty, GRAPHICS_TREE=self
    endif
    (self.oMainGUI->GetObjectByName('3DWindow')) -> GetProperty, DIMENSIONS=dims
    self.oBuffer -> SetProperty, DIMENSIONS=dims
    self.oBuffer -> Draw
    oImage = self.oBuffer -> Read()
    oImage -> GetProperty, DATA=image
    obj_destroy, oImage
    if n_elements(reverse) GT 0 then $
        image = reverse(image, reverse[0])

    return, image

end


;--------------------------------------------------------------
;+
;  @returns An object in the graphics tree by name.
;
;  @param name {in}{type=string}{required}
;    A string naming the object to be returned.
;-
function PALM_3DDisplay::GetObjectByName, name

    case name of
        'AxesModel': begin
            oReturn = self -> GetByName('MainModel/AxesModel')
        end
        'AxesShearModel' : begin
            oReturn = self -> GetByName('MainModel/AxesModel/AxesShearModel')
        end
        'EMVolume': begin
            oReturn = self->GetByName('MainModel/VolumeModel/Translation/Rotation/VolumeShearModel/EMVolume')
        end
        'PALM': begin
            oReturn = (self->GetObjectByName('VolumeShearModel')) -> Get(/ALL, ISA='PALMgr3DModel')
        end
        'VolumeModel': begin
            oReturn = self -> GetByName('MainModel/VolumeModel')
        end
        'VolumeRotate': begin
            oReturn = self -> GetByName('MainModel/VolumeModel/Translation/Rotation')
        end
        'VolumeShearModel' : begin
            oReturn = self -> GetByName('MainModel/VolumeModel/Translation/Rotation/VolumeShearModel')
        end
        'VolumeTranslate': begin
            oReturn = self -> GetByName('MainModel/VolumeModel/Translation')
        end
        'Volume' : begin
            oReturn = (self->GetObjectByName('PALM'))->GetByName('VolumeObject')
        end
        else: begin
            objs = self -> Get(/ALL, COUNT=count)
            for i = 0, count-1 do begin
                objs[i] -> GetProperty, NAME=tempName
                if tempName EQ name then begin
                    oReturn = objs[i]
                    break
                endif
            endfor
        end
    endcase

    if ~obj_valid(oReturn) then $
        oReturn = obj_new()

    return, oReturn

end


;------------------------------------------------------------------------------
;+
; This method is for retrieving object properties.
;
; @keyword
;    ERROR_MESSAGE {out}{optional}
;      Set this keyword to a named variable to retrieve
;      any error messages thrown by this method
; @keyword
;    MODEL {out}{optional}
;      Set this keyword to a named variable to retrieve
;      a reference to the PALMgr3DModel object
; @keyword
;    WINDOW {out}{optional}
;      Set this keyword to a named variable to retrieve
;      a reference to the PALM_3DWindow object
; @keyword
;    REFERENCE_RANGE {out}{optional}
;      Set this keyword to a named variable to retrieve
;      the  reference range for normalizing the volume
;      data
; @keyword
;    XRANGE {out}{optional}
;      Set this keyword to a named variable to retrieve
;      the x-range of the volume data
; @keyword
;    YRANGE {out}{optional}
;      Set this keyword to a named variable to retrieve the
;      y-range of the volume data
; @keyword
;    ZRANGE {out}{optional}
;      Set this keyword to a named variable to retrieve
;      z-range of the volume data
; @keyword
;    _REF_EXTRA {out}{optional}
;      Set this keyword to a named variable to retrieve
;      properties from the inherited IDLgrView
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::GetProperty, $
    COLOR_TABLE=ColorTable, $
    DRAG_QUALITY=DragQuality, $
    EM_COLOR_TABLE=emColorTable, $
    EM_OPACITY_FUNCTION=emOpacityFunction, $
    EM_OPACITY_TABLE=emOpacityTable, $
    ERROR_MESSAGE=errMsg, $
    HAVE_EM_VOLUME=haveEmVolume, $
    HIDE_SLICE_LINES=HideSliceLines, $
    HIDE_SLICE_PLANES=HideSlicePlanes, $
    HIDE_VOLUME=HideVolume, $
    MODEL=oModel, $
    OPACITY_FUNCTION=OpacityFunction, $
    OPACITY_TABLE=OpacityTable, $
    REFERENCE_RANGE=refRange, $
    WINDOW=oWindow, $
    XRANGE=xRange, $
    YRANGE=yRange, $
    ZRANGE=zRange, $
    _REF_EXTRA=_extra

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

    if arg_present(ColorTable) then begin
        oModel = self -> GetObjectByName('PALM')
        oModel -> GetProperty, COLOR_TABLE=ColorTable
    endif

    if arg_present(DragQuality) then begin
        DragQuality = self.DragQuality
    endif

    if arg_present(emColorTable) then begin
        (self->GetObjectByName('EMVolume'))->GetProperty, $
            RGB_TABLE0=emColorTable
    endif
    if arg_present(emOpacityFunction) then begin
        emOpacityFunction = self.emOpacityFunction
    endif
    if arg_present(emOpacityTable) then begin
        (self->GetObjectByName('EMVolume'))->GetProperty, $
            OPACITY_TABLE0=emOpacityTable
    endif
    if arg_present(haveEmVolume) then begin
        haveEmVolume = obj_valid(self->GetObjectByName('EMVolume'))
    endif
; Hide slice-lines/planes
    if arg_present(HideVolume) then begin
        oModel = self->GetObjectByName('PALM')
        oModel->GetProperty, HIDE=HideVolume
    endif
    if arg_present(HideSliceLines) then $
        (self.oSliceLines)[0] -> GetProperty, HIDE=HideSliceLines
    if arg_present(HideSlicePlanes) then $
        (self.oSlicePlanes)[0] -> GetProperty, HIDE=HideSlicePlanes

; Model
    if arg_present(oModel) then $
        oModel = self -> GetObjectByName('PALM')
; Opacity function
    if arg_present(OpacityFunction) GT 0 then $
        OpacityFunction = self.OpacityFunction
; Opacity table
    if arg_present(OpacityTable) then begin
        oModel = self -> GetObjectByName('PALM')
        if obj_valid(oModel) then $
            oModel -> GetProperty, OPACITY_TABLE=OpacityTable
    endif
; Reference range
    if arg_present(refRange) then begin
        oModel = self -> GetObjectByName('PALM')
        if obj_valid(oModel) then $
            oModel -> GetProperty, REFERENCE_RANGE=refRange
    endif
; Window
    if arg_present(oWindow) then $
        self.oMainGUI -> GetProperty, WINDOW_3D=oWindow
; X-Range
    if arg_present(xRange) then begin
        oModel = self -> GetObjectByName('PALM')
        if obj_valid(oModel) then $
            oModel -> GetProperty, VOLUME_XRANGE=xRange
    endif
; Y-Range
    if arg_present(yRange) then begin
        oModel = self -> GetObjectByName('PALM')
        if obj_valid(oModel) then $
            oModel -> GetProperty, VOLUME_YRANGE=yRange
    endif
; Z-Range
    if arg_present(zRange) then begin
        oModel = self -> GetObjectByName('PALM')
        if obj_valid(oModel) then $
            oModel -> GetProperty, VOLUME_ZRANGE=zRange
    endif
; Extra
    if n_elements(_extra) GT 0 then begin
        self -> IDLgrView::GetProperty, _EXTRA=_extra
    endif

end


;------------------------------------------------------------------------------
;+
; This method initializes the object.
;
; @returns
;    1 for success and 0 for failure
;
; @param
;    oMainGUI {in}{type=object reference}{required}
;
; @keyword
;    ERROR_MESSAGE {out}{optional}
;      Set this keyword to a named variable to retrieve any error
;      messages thrown.
;
; @keyword
;    VERBOSE {in}{type=boolean}{optional}
;      Set this keyword to have the object display error messages
;      in the IDL output log.
;
; @keyword
;    _EXTRA {in}{optional}
;      Any additional keywords will be passed to the inherited
;      IDLgrView object.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_3DDisplay::Init, oMainGUI, $
    ERROR_MESSAGE=errMsg, $
    VERBOSE=verbose, $
    _EXTRA=_extra

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

    self.verbose = keyword_set(verbose)
    self.DragQuality = 'MEDIUM'

; Make sure the reference to an IDLitWindow was passed in
    if ~obj_valid(oMainGUI) then begin
        errMsg = 'Input argument oMainGUI is not a valid object'
        if self.verbose then $
            print, errMsg
        return, 0
    endif
    if ~obj_isa(oMainGUI, 'PALM_MainGUI') then begin
        errMsg = 'Input argument oMainGUI is not of type PALM_MainGUI'
        if self.verbose then $
            print, errMsg
        return, 0
    endif
    self.OpacityFunction = 'Linear (Increasing)'
    self.emOpacityFunction = 'Linear (Increasing)'
; Initialize the parent class
    zClip = [10,-10]
    void = self -> IDLgrView::Init( $
        EYE=zClip[0]+0.1, $
        ZCLIP=[10,-10], $
        _EXTRA=_extra)
    if ~void then $
        return, 0
    self.xy = fltarr(2)
    self.xyzStart = fltarr(3)
    self.oMainGUI = oMainGUI
; Initialize the display
    if ~self->InitializeDisplay(ERROR_MESSAGE=errMsg) then $
        return, 0

    ;  Wire frame by default
    self->SetProperty, HIDE_SLICE_PLANES=1, HIDE_SLICE_LINES=0

    return, 1

end


;------------------------------------------------------------------------------
;+
; This method initializes the graphics objects used
;
; @returns
;    1 for success and 0 for failure
;
; @keyword
;    ERROR_MESSAGE {out}{optional}
;      Set this keyword to a named variable to retrieve any error
;      messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_3DDisplay::InitializeDisplay, $
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

; Main model
    oMainModel = obj_new('IDLgrModel', NAME='MainModel')
;
; Volume model hierarchy
;
    oVolumeModel = obj_new('IDLgrModel', NAME='VolumeModel')
    oTranslationModel = obj_new('IDLgrModel', NAME='Translation')
    oRotationModel = obj_new('IDLgrModel', NAME='Rotation')
    oVolumeShearModel = obj_new('IDLgrModel', NAME='VolumeShearModel')
    oRotationModel->Add, oVolumeShearModel
    oTranslationModel -> Add, oRotationModel
    oVolumeModel -> Add, oTranslationModel
    oMainModel -> Add, oVolumeModel
    self -> Add, oMainModel
;
; Axes model
;
    oAxesModel = obj_new('IDLgrModel', NAME='AxesModel')
    oAxesModel -> Translate, -0.75, -0.75, 0
    oAxesShearModel = obj_new('IDLgrModel', NAME='AxesShearModel')
    oMainModel -> Add, oAxesModel
    oAxesModel ->Add, oAxesShearModel
;
; Slicing planes
;
    colors = [ $
              [255,0,0], $
              [0,255,0], $
              [0,255,255] $
             ]
    self.oSliceLines = objarr(3)
    self.oSlicePlanes = objarr(3)

    oWindow = self.oMainGUI -> GetObjectByName('3DWindow')
    oWindow -> SetProperty, GRAPHICS_TREE=self
    AxisRange = [0,1]

    for i = 0, 2 do begin
        self.oAxes[i] = obj_new('IDlgrAxis', i, $
            COLOR=colors[*,i], $
            /EXACT, $
            LOCATION=[0,0,0], $
            MAJOR=0, $
            MINOR=0, $
            RANGE=AxisRange, $
            THICK=2, $
            TICKLEN=0, $
            UVALUE=AxisRange)
        if i LT n_elements(self.oText) then $
            self.oText[i] = obj_new('IDLgrText', $
                COLOR=colors[*,i], $
                /ONGLASS)
        self.oSliceLines[i] = obj_new('IDLgrPolygon', $
            COLOR=colors[*,i], $
            STYLE=1, $
            THICK=2)
        self.oSlicePlanes[i] = obj_new('IDLgrPolygon', $
            ALPHA=0.1, $
            COLOR=colors[*,i], $
            STYLE=2)
    endfor
    oVolumeShearModel->Add, [self.oSliceLines, self.oSlicePlanes]
    oAxesShearModel -> Add, [self.oAxes, self.oText]
    oAxesModel -> Translate, -0.1, -0.1, 0
    oWindow -> GetProperty, DIMENSIONS=dims
    self.oTrack = obj_new('Trackball', dims/2., (dims[0]<dims[1])/2.0)

    self -> SetViewplane, /NO_DRAW

    return, 1

end

;------------------------------------------------------------------------------
;+
; This method reads the contents of a flythrough file
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Global Services Group
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_3DDisplay::ReadFlythroughFile, $
    ANGLE=angle, $
    NUM_ITERATIONS=nIter, $
    POSITION=position, $
    ROTATION=rotation

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if n_elements(lun) GT 0 then $
            free_lun, lun
        help, /LAST, OUT=ErrMsg
        ErrMsg = ['Unexpected Error:', ErrMsg]
        void = dialog_message(ErrMsg, $
            /ERROR, $
            TITLE='Flythough I/O Error')
        return, 0
    endif

    self.oMainGUI->GetAnimationSettings, FLYTHROUGH_FILE=FlythroughFile

    if ~file_test(FlythroughFile, /READ) then begin
        void = dialog_message(['Unable to read or locate flythrought file: ',FlythroughFile], $
            /ERROR, $
            TITLE='Flythrough I/O Error')
        return, 0
    endif
;
; Read the contents of the flythrough file
;
    nIter = file_lines(FlythroughFile)-1
    if nIter LE 0 then begin
        void = dialog_message(['Invalid flythrough file: ', FlythroughFile], $
            /ERROR, $
            TITLE='Flythrough I/O Error')
        return, 0
    endif
    header = ''
    tempData = strarr(nIter)
    openr, lun, FlythroughFile, /GET_LUN
    readf, lun, header
    readf, lun, tempData
    free_lun, lun
    header = strupcase(strtok(header, ',', /EXTRACT, /PRESERVE_NULL))
    columnHeaders = [ $
        'X-POSITION', $
        'Y-POSITION', $
        'Z-POSITION', $
        'ELEVATION', $
        'AZIMUTH', $
        'ROLL' $
        ]
    nIndex = n_elements(columnHeaders)
    index = intarr(nIndex)
    for i = 0, nIndex-1 do $
        index[i] = (where(header EQ columnHeaders[i]))[0]
    missing = where(index LT 0, countMissing)
    if countMissing GT 0 then begin
        ErrMsg = $
            ['Invalid flythrough file: ', FlythroughFile, $
            '', $
            'Missing headers:', $
            columnHeaders[missing] $
            ]
        void = dialog_message(ErrMsg, $
            /ERROR, $
            TITLE='Flythrough I/O Error')
        return, 0
    endif
    data = dblarr(nIndex,nIter)
    for i = 0, nIter-1 do $
        data[*,i] = double(strtok(tempData[i], ',', /EXTRACT, /PRESERVE_NULL))
    position = data[index[0:2],*]
    angle = (temporary(data))[index[3:5],*]
    if max(angle) GT 2*!pi then $
        angle = angle * !dtor

    return, 1

end

;------------------------------------------------------------------------------
;+
; This method renderes the current scene.
;
; @keyword
;    ERROR_MESSAGE {out}{optional}
;      Set this keyword to a named variable to retrieve any error
;      messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::RenderScene, $
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

    (self.oMainGUI->GetObjectByName('3DWindow')) -> Draw

end


;------------------------------------------------------------------------------
;+
; Resets the rotation, translation and zoom.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @keyword NO_DRAW {in}{type=boolean}{optional}
;   If set, do not update the graphics displays.
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::Reset, $
    NO_DRAW=noDraw

    (self->GetObjectByName('VolumeModel')) -> reset
    (self->GetObjectByName('VolumeTranslate')) -> reset
    (self->GetObjectByName('VolumeRotate')) -> reset
    oAxesModel = self -> GetObjectByName('AxesModel')
    oAxesModel -> GetProperty, TRANSFORM=tm
    oAxesModel -> Reset
    oAxesModel -> Translate, tm[3,0], tm[3,1], tm[3,2]
    oShearModel = self -> GetObjectByName('VolumeShearModel')
    oShearModel -> Reset
    oAxesShearModel = self -> GetObjectByName('AxesShearModel')
    oAxesShearModel -> Reset

    if ~keyword_set(noDraw) then $
        self -> RenderScene

end


;------------------------------------------------------------------------------
;+
; This method rotates the volume a specified increment about each specified
; axis.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @param increment {in}{type=number}{required}
;   The amount by which the volume is rotated.
;
; @keyword ABSOLUTE {in}{type=number}{optional}
;   If set, do an absolute rotation.
;
; @keyword ERROR_MESSAGE {out}{type=string}{optional}
;   Error message, if any.
;
; @keyword NO_DRAW {in}{type=boolean}{optional}
;   If set, do not update the graphics displays.
;
; @keyword X {in}{type=boolean}{optional}
;   If set, rotate about this axis.
;
; @keyword Y {in}{type=boolean}{optional}
;   If set, rotate about this axis.
;
; @keyword Z {in}{type=boolean}{optional}
;   If set, rotate about this axis.
;
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::RotateVolume, increment, $
    ABSOLUTE=absolute, $
    ERROR_MESSAGE=ErrMsg, $
    NO_DRAW=noDraw, $
    X=x, $
    Y=y, $
    Z=z

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
; Rotate the data
    oModelRotate = self -> GetObjectByName('VolumeRotate')
    oModelRotate -> GetProperty, TRANSFORM=tm
    wDraw = self.oMainGUI -> Get('PALM_3DWindow')
    widget_control, wDraw, GET_VALUE=oWindow
    oWindow -> GetProperty, DIMENSIONS=winDims
    if (keyword_set(absolute)) then begin
        ;
        ; The current orientation is the product of the rotations about three axes.
        ; First get the equivalent ordered angle operations that would get us to this
        ; same orientation.
        ;
        XYZ = Angle3123(tm[0:2, 0:2])
        ;
        ; Next, change the angle in question
        ;
        subscript = keyword_set(x) ? 0 : (keyword_set(y) ? 1 : (keyword_set(z) ? 2 : 0))
        XYZ[subscript] = increment
        oModelFake = Obj_New('IDLgrModel')
        oModelFake->Rotate, [1, 0, 0], XYZ[0]
        oModelFake->Rotate, [0, 1, 0], XYZ[1]
        oModelFake->Rotate, [0, 0, 1], XYZ[2]
        oModelFake->GetProperty, TRANSFORM = newtm
        Obj_Destroy, oModelFake
        newtm[3, 0] = tm[3, 0]
        newtm[3, 1] = tm[3, 1]
        oModelRotate -> SetProperty, TRANSFORM = newtm
    endif else begin
        oModelRotate -> Reset
        oModelRotate -> Rotate, [keyword_set(x), keyword_set(y), keyword_set(z)], increment
        oModelRotate -> GetProperty, TRANSFORM=tmNew
        oModelRotate -> SetProperty, TRANSFORM=tmNew#tm
    endelse
;
; Rotate the axes
;
    oModel = self -> GetObjectByName('AxesModel')
    oModel -> GetProperty, TRANSFORM=tm
    if (keyword_set(absolute)) then begin
        newtm[3, 0] = tm[3, 0]
        newtm[3, 1] = tm[3, 1]
        oModel -> SetProperty, TRANSFORM = newtm
    endif else begin
        oModel -> Translate, -tm[3,0], -tm[3,1], -tm[3,2]
        oModel -> Rotate, [keyword_set(x), keyword_set(y), keyword_set(z)], increment
        oModel -> Translate, tm[3,0], tm[3,1], tm[3,2]
    endelse

    if ~keyword_set(noDraw) then $
        self -> RenderScene

end


;------------------------------------------------------------------------------
;+
; This method is for setting properties of the 3D display.
;
; @keyword OPACITY_TABLE {in}{type=bytarr}{optional}
;     Set this keyword to a 256-element byte array specifying the
;     opacity table for the volume.
;
; @keyword BACKGROUND_COLOR {in}{type=RGB vector}{optional}
;   Set the background color.
;
; @keyword COLOR_TABLE {in}{type=3x256 byte array}{optional}
;   Set the color table.
;
; @keyword DRAG_QUALITY {in}{type=string}{optional}
;   Set the drag quality.
;
; @keyword HIDE_SLICE_LINES {in}{type=boolean}{optional}
;   If true, hide the slice lines.
;
; @keyword HIDE_SLICE_PLANES {in}{type=boolean}{optional}
;   If true, hide the slice planes.
;
; @keyword HIDE_VOLUME {in}{type=boolean}{optional}
;   If true, hide the volume.
;
; @keyword OPACITY_FUNCTION {in}{type=256 element byte array}{optional}
;   Set the opacity function used for volume display.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::SetProperty, $
    BACKGROUND_COLOR=BackgroundColor, $
    COLOR_TABLE=Colortable, $
    DRAG_QUALITY=DragQuality, $
    EM_ALPHA_CHANNEL=emAlphaChannel, $
    EM_COLOR_TABLE=emColorTable, $
    EM_OPACITY_FUNCTION=emOpacityFunction, $
    EM_OPACITY_TABLE=emOpacityTable, $
    EM_VOLUME_HIDE=emHide, $
    HIDE_SLICE_LINES=HideSliceLines, $
    HIDE_SLICE_PLANES=HideSlicePlanes, $
    HIDE_VOLUME=HideVolume, $
    OPACITY_FUNCTION=OpacityFunction, $
    OPACITY_TABLE=OpacityTable

    if n_elements(BackgroundColor) GT 2 then begin
        self -> IDLgrView::SetProperty, COLOR=BackgroundColor[0:2]
    endif
    if n_elements(ColorTable) GT 0 then begin
        oModel = self -> GetObjectByName('PALM')
        oModel->GetProperty, HUE=h
        if (h ne 1) then  $
            oModel -> SetProperty, COLOR_TABLE=ColorTable
    endif
    if n_elements(DragQuality) GT 0 then begin
        self.DragQuality = strupcase(DragQuality)
    endif
;
; EM volume settings
;
    if n_elements(emAlphaChannel) GT 0 then begin
        oVolumeEM = self->GetObjectByName('EMVolume')
        if obj_valid(oVolumeEM) then begin
            emOpacity = replicate(byte(emAlphaChannel*255B),256)
            oVolumeEM->SetProperty, OPACITY_TABLE0=emOpacity
        endif
    endif
    if n_elements(emColorTable) EQ 768 then begin
        oVolumeEM = self->GetObjectByName('EMVolume')
        if obj_valid(oVolumeEM) then $
            oVolumeEM->SetProperty, RGB_TABLE0=reform(emColorTable,256,3)
    endif
    if n_elements(emHide) GT 0 then begin
        oVolumeEM = self->GetObjectByName('EMVolume')
        if obj_valid(oVolumeEM) then $
            oVolumeEM->SetProperty, HIDE=emHide, $
                UVALUE=emHide
    endif
    if n_elements(emOpacityFunction) GT 0 then begin
        self.emOpacityFunction = emOpacityFunction
    endif
    if n_elements(emOpacityTable) EQ 256 then begin
        oVolumeEM = self->GetObjectByName('EMVolume')
        if obj_valid(oVolumeEM) then $
            oVolumeEM->SetProperty, OPACITY_TABLE0=emOpacityTable
    endif
    
    if n_elements(HideSlicePlanes) GT 0 then begin
        hide = keyword_set(HideSlicePlanes)
        for i = 0, 2 do $
            self.oSlicePlanes[i] -> SetProperty, HIDE=hide
    endif
    if n_elements(HideSliceLines) GT 0 then begin
        hide = keyword_set(HideSliceLines)
        for i = 0, 2 do $
            self.oSliceLines[i] -> SetProperty, HIDE=hide
    endif
    if n_elements(HideVolume) GT 0 then begin
        oModel = self -> GetObjectByName('PALM')
        oModel -> SetProperty, HIDE=HideVolume, $
            UVALUE=HideVolume
    endif
    if n_elements(OpacityFunction) then begin
        self.OpacityFunction = OpacityFunction
    endif
    nOpacity = n_elements(OpacityTable)
    if nOpacity GT 0 then begin
        if nOpacity EQ 256 then begin
            oModel = self -> GetObjectByName('PALM')
            if obj_valid(oModel) then $
                oModel -> SetProperty, OPACITY_TABLE=OpacityTable
        endif else begin
            if self.verbose then $
                message, /CONTINUE, $
                    'Invalid opacity table'
        endelse
    endif

end


;------------------------------------------------------------------------------
;+
; This method adjusts the viewplane rectangle according to the window
; dimensions
;
; @keyword
;    ERROR_MESSAGE {out}{optional}
;      Set this keyword to a named variable to retrieve any error
;      messages thrown.
;
; @keyword
;    NO_DRAW {in}{type=boolean}{optional)
;      Set this keyword to have the display not update when finished.
;      By default the display will be redrawn
;
; @keyword
;    UPDATE_TRACKBALL {in}{type=boolean}{optional}
;      Set this keyword to have the trackball update its center and radius
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::SetViewplane, $
    ERROR_MESSAGE=errMsg, $
    NO_DRAW=noDraw, $
    UPDATE_TRACKBALL=updateTrack

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

    oWindow = self.oMainGUI -> GetObjectByName('3DWindow')
    oWindow -> GetProperty, DIMENSIONS=winDims

    bias = -1.0
    aspect = float(winDims[0])/float(winDims[1])
    viewplane = (aspect GT 1) ? $
        [(1.0-aspect)/2.0+bias, 0.0+bias, aspect*2, 2.0] : $
        [0.0+bias, (1.0-(1.0/aspect))/2.0+bias, 2.0, (2.0/aspect)]

    self-> IDLgrView::SetProperty, VIEWPLANE_RECT = viewplane
    
    offset = viewplane[0:1] + 1./3 * [-1,1] + [viewplane[2],0]
    tx = [[1./3, 0, 0, offset[0]], [0, 1./3, 0, offset[1]], $
      [0,0,1,0], [0,0,0,1]]
    

    ; Update the trackball:
    if keyword_set(updateTrack) then $
        self-> UpdateTrackball
    if ~keyword_set(noDraw) then $
        self -> RenderScene

end


;------------------------------------------------------------------------------
;+
; This method rotates the volume a specified increment about each specified
; axis.
;
; @History
;   December, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;   Feb, 2008 : AB - ITT : Change the z axis and add offset.
;
; @param increment {in}{type=number}{required}
;   Amount to shear.
;
; @keyword ABSOLUTE {in}{type=boolean}{required}
;   Set the volume & axes transforms to zero if set.
;
; @keyword ERROR_MESSAGE {in}{type=string}{required}
;   String representing any error message.
;
; @keyword NO_DRAW {in}{type=boolean}{required}
;   If set, do not update the graphics displays.
;
; @keyword SHEAR_PHASE {in}{type=number}{required}
;   Shear phase in radians (for circular shearing)
;
; @keyword X {in}{type=boolean}{required}
;   If set, shear along X.
;
; @keyword Y {in}{type=boolean}{required}
;   If set, shear along Y.
;
; @keyword Z {in}{type=boolean}{required}
;   If set, shear along X or Y, or both if MODE = 2
;
; @keyword MODE {in}{type=number}{required}
;   Integer representing the shear mode.
;
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::ShearVolume, increment, $
    ABSOLUTE=absolute, $
    ERROR_MESSAGE=ErrMsg, $
    NO_DRAW=noDraw, $
    SHEAR_PHASE=shearPhase, $
    X=x, $
    Y=y, $
    Z=z, MODE=shearMode

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

    if (n_elements(shearMode) eq 0) then shearMode = 0

    ; Shear the data and axes
    oVolumeShearModel = self -> GetObjectByName('VolumeShearModel')
    oVolume = self -> GetObjectByName('Volume')
    oVolumeShearModel -> GetProperty, TRANSFORM=volumeTransform
    oAxesShearModel = self -> GetObjectByName('AxesShearModel')
    oAxesShearModel -> GetProperty, TRANSFORM=axesTransform
    wDraw = self.oMainGUI -> Get('PALM_3DWindow')
    widget_control, wDraw, GET_VALUE=oWindow
    oWindow -> GetProperty, DIMENSIONS=winDims

    case 1 of
        keyword_set(x) : begin
            pt = [[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
            index = where(pt)
        end
        keyword_set(y) : begin
            pt = [[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]
            index = where(pt)
        end
        keyword_set(z) : begin
            xmode = (ymode = 0)
            case shearMode of
            0 : xmode = 1
            1 : ymode = 1
            2 : begin
                xmode = cos(shearPhase)
                ymode = sin(shearPhase)
            end
            endcase
            pt = [[0,0,xmode,0],[0,0,ymode,0],[0,0,0,0],[0,0,0,0]]
            index = [2,6]
        end
    endcase
    if (keyword_set(absolute)) then begin
        volumeTransform[index] = 0
        axesTransform[index] = 0
    endif
    volumeTransform += pt * increment
    axesTransform += pt * increment

    ; AB
    ; Need to lock Z=0 for the volume
    ; Lock (0,0,0) in the first volume should do it
    ; TODO not sure if we need to worry about multiple volumes
    oVol = self->GetObjectByName('Volume')
    if obj_valid(oVol) then begin
        ; These volumes are using ?COORD_CONV
        ; which causes the (0,0,0) to be offset
        oVol->GetProperty, XCOORD_CONV=xc, YCOORD_CONV=yc, ZCOORD_CONV=zc
        pt0 = [xc[0], yc[0], zc[0]]
        volumetransform[3,0:2] = 0
        volumeTransform[3,0:2] = pt0 - (matrix_multiply(volumeTransform, [pt0,1], $
            /ATRANSPOSE))[0:2]
    endif
    oVolumeShearModel -> SetProperty, TRANSFORM=volumeTransform
    oAxesShearModel -> SetProperty, TRANSFORM=axesTransform


    if ~keyword_set(noDraw) then begin
        self -> RenderScene
    endif

end


;------------------------------------------------------------------------------
;+
; This method handles interactive model transformations
;
; @param
;  xy {in}{required}{type=double}
;    The xy position of the mouse cursor (sEvent.x, sEvent.y).
;
; @keyword
;    ERROR_MESSAGE {out}{optional}
;      Set this keyword to a named variable to retrieve any error
;      messages thrown.
; @keyword
;    NO_DRAW {in}{type=boolean}{optional}
;     Set this keyword to prevent display update.
; @keyword
;    PAN {in}{type=boolean}{optional}
;      Set this keyword to indicate that movement should be constrained
;      to translation in the xy-plane.
; @keyword
;   RELEASE {in}{type=boolean}{optional}
;    Set this keyword to clear the starting point.  Use this on
;    mouse up for example.  The next time this is called without
;    /RELEASE will be recorded as the starting point for the
;    mouse y position.
; @keyword
;   START {in}{type=boolean}{optional}
;     Set this keyword to define the starting point, for example
;     on mouse down.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::TransformModel, xy, $
    ERROR_MESSAGE=errMsg, $
    NO_DRAW=noDraw, $
    PAN=Pan, $
    RELEASE=release, $
    START=start

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

    doPan = keyword_set(Pan)
    doRelease = keyword_set(release)
    doStart = keyword_set(start)
    self -> UpdateManip, $
        /DO_LOW, $
        NO_DRAW=doStart, $
        RELEASE=release, $
        START=start
    wDraw = self.oMainGUI -> Get('PALM_3DWindow')

    event={WIDGET_DRAW, $
           ID        : wDraw, $
           TOP       : self.oMainGUI->Get('tlb'), $
           HANDLER   : self.oMainGUI->Get('tlb'), $
           TYPE      : doStart ? 0B : (~doRelease+1), $
           X         : xy[0], $
           Y         : xy[1], $
           PRESS     : ~doRelease, $
           RELEASE   : doRelease, $
           CLICKS    : 1L, $
           MODIFIERS : 0L, $
           CH        : 0B, $
           KEY       : 0L }
    haveTransform = self.oTrack -> Update(event, $
        TRANSFORM=tmTrack, $
        TRANSLATE=doPan)
    if doRelease then $
        self.xyzStart = [0.0,0.0,0.0]
    if doStart && ~doPan then begin
    ;
    ; Set the center of rotation
    ;
        oModel = self -> GetObjectByName('VolumeTranslate')
        widget_control, wDraw, GET_VALUE=oWindow
        oWindow -> GetProperty, DIMENSIONS=winDIms
        void = oWindow->PickData(self, oModel, winDims/2, dataXYZ)
        self.xyzStart = (void GE 0) ? [dataXYZ[0:1],0.0] : [0,0,0]
    endif
    if haveTransform then begin
        oVolumeTranslate = self -> GetObjectByName('VolumeTranslate')
        if ~obj_valid(oVolumeTranslate) then $
            return
        if doPan then begin
        ; Adjust for zooming
            (self->GetObjectByName('VolumeModel')) -> GetProperty, TRANSFORM=tmZoom
            tmTrack[3,0:2] = tmTrack[3,0:2] / tmZoom[[0,1,2],[0,1,2]]
        ; Update the transform
            oVolumeTranslate -> GetProperty, TRANSFORM=tm
            oVolumeTranslate -> SetProperty, TRANSFORM=tm#tmTrack
        endif else begin
        ; Adjust the reference axes
            oAxesModel = self -> getObjectByName('AxesModel')
            oAxesModel -> GetProperty, TRANSFORM=tmAxes
            tempAxes = tmAxes[3,0:2]
            tmAxes[3,0:2] = 0.0
            tmAxes = tmAxes # tmTrack
            tmAxes[3,0:2] = tempAxes
            oAxesModel -> SetProperty, TRANSFORM=tmAxes
        ; Adjust the volume
            widget_control, wDraw, GET_VALUE=oWindow
            oWindow -> GetProperty, DIMENSIONS=winDIms
            oVolumeRotate = self -> GetObjectByName('VolumeRotate')
            oVolumeROtate->Translate, -self.xyzStart[0], -self.xyzStart[1], -self.xyzStart[2]
            oVolumeRotate -> GetProperty, TRANSFORM=tm
            tm = tm # tmTrack
            oVolumeRotate -> SetProperty, TRANSFORM=tm
            oVolumeRotate->Translate, self.xyzStart[0], self.xyzStart[1], self.xyzStart[2]
        endelse

        if ~keyword_set(noDraw) then $
            self -> RenderScene
    endif
    ; Update the trackball after panning is complete
    if doPan AND doRelease then $
        self -> UpdateTrackball

end



;------------------------------------------------------------------------------
;+
; This method updates the 3-dimensional display for manipulation
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @keyword DO_LOW {in}{type=boolean}{optional}
;   If set, render as if drag quality is low.
;
; @keyword NO_DRAW {in}{type=boolean}{optional}
;   If set, do no update the graphics display.
;
; @keyword RELEASE {in}{type=boolean}{optional}
;   If set, mouse button up.
;
; @keyword START {in}{type=boolean}{optional}
;   If set, drag starting.
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::UpdateManip, $
    DO_LOW=doLow, $
    NO_DRAW=noDraw, $
    RELEASE=release, $
    START=start

    names = ['EMVolume','PALM']
    n = n_elements(names)-1
    for i = 0, n do begin
        case self.DragQuality of
            'LOW': begin
                oModel = self -> GetObjectByName(names[i])
                if obj_valid(oModel) then begin
                    if keyword_set(release) then begin
                        oModel->GetProperty, UVALUE=isHidden
                        oModel -> SetProperty, HIDE=isHidden
                    endif
                    if keyword_set(start) then $
                        oModel -> SetProperty, /HIDE
                endif
            end
            'MEDIUM': begin
                if keyword_set(release) then begin
                    self -> UpdateSlicePlanes, NO_RENDER=((i LT n) || noDraw || keyword_set(doLow))
                endif
                if keyword_set(doLow) then begin
                    oModel = self -> GetObjectByName(names[i])
                    if obj_valid(oModel) then begin
                        if keyword_set(release) then begin
                            oModel->GetProperty, UVALUE=isHidden
                            oModel -> SetProperty, HIDE=isHidden
                            if i EQ n then begin
                                if ~keyword_set(noDraw) then begin
                                    self -> RenderScene
                                endif
                                return
                            endif
                        endif
                        if keyword_set(start) then begin
                            oModel -> SetProperty, /HIDE
                            if i EQ n then begin
                                self -> RenderScene
                                return
                            endif
                        endif
                    endif
                endif
                if i EQ n then $
                    return
            end
            'HIGH': return
            else: return
        endcase
    endfor

    if ~keyword_set(noDraw) then $
        self -> RenderScene

end


;------------------------------------------------------------------------------
;+
; This method adds a display model to the display destroying the
; existing model if present.
;
; @param
;    oModel {in}{type=object reference}{required}
;      A reference to an instance of the PALMgr3DModel object
;
; @keyword
;    ERROR_MESSAGE {out}{type=string}{optional}
;      Set this keyword to a named variable to retrieve any error
;      messages thrown.
;
; @keyword NO_DRAW {in}{type=boolean}{optional}
;   If set, do not update the graphics displays.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::UpdateModel, oModel, $
    ERROR_MESSAGE=errMsg, $
    NO_DRAW=noDraw

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

    if ~obj_valid(oModel) then $
        return
    if ~obj_isa(oModel, 'PALMgr3DModel') then $
        return

    oModelOriginal = self -> GetObjectByName('PALM')
    if obj_valid(oModelOriginal) then begin
        oModelOriginal->GetProperty, UVALUE=isHidden
        obj_destroy, oModelOriginal
    endif else $
        isHidden = 0

    oModel -> GetProperty, X_RANGE=xRange, $
                           Y_RANGE=yRange, $
                           Z_RANGE=zRange, $
                           VOLUME_XRANGE=vxr, $
                           VOLUME_YRANGE=vyr, $
                           VOLUME_ZRANGE=vzr, $
                           XCOORD_CONV=xc, $
                           YCOORD_CONV=yc, $
                           ZCOORD_CONV=zc

    mini = min([vxr, vyr, vzr], MAX=maxi)
    minir = [total(vxr)/2, total(vyr)/2, total(vzr)/2] - (maxi-mini)/2.
    maxir = [total(vxr)/2, total(vyr)/2, total(vzr)/2] + (maxi-mini)/2.
    corners = [ $
      [minir[0], minir[1], minir[2]], [maxir[0], minir[1], minir[2]], $
      [maxir[0], maxir[1], minir[2]], [minir[0], maxir[1], minir[2]], $
      [minir[0], minir[1], maxir[2]], [maxir[0], minir[1], maxir[2]], $
      [maxir[0], maxir[1], maxir[2]], [minir[0], maxir[1], maxir[2]] ]

    ;  Get the z scale factor
    widget_control, self.oMainGUI->Get('zScaleFactorLabel'), GET_VALUE=zscale
    zscale = float(zscale)

    ;  Get the volume range including z-scale factor
    range = [xRange[1]-xRange[0], $
             yRange[1]-yRange[0], $
             zRange[1]-zRange[0]]

    ;  Scale to a nice number
    nm = range/8.0
    lengths = [5, 10, 20, 30, 50, 100, 200, 300, 400, 500, 1000,  $
               1500, 2000, 2500, 3000, 3500, 4000]
    d = [0L,0,0]
    for i=0,2 do begin
        void = min(abs(lengths-nm[i]), idx)
        d[i] = lengths[idx]
    endfor
    nm = d
    vm = float(max([vxr[1],vyr[1],vzr[1]]))
    fr = (nm/range)*[vxr[1]/vm, vyr[1]/vm, vzr[1]/vm]

    for i = 0, 2 do begin
    ; Update the axes
        self.oAxes[i] -> GetProperty, UVALUE=AxisRange
        self.oAxes[i] -> SetProperty, RANGE=AxisRange*fr[i]
    ; Update the axis text
        if i LT n_elements(self.oText) then begin
            loc = [(i eq 0),(i eq 1),(i eq 2)]*AxisRange[1]*fr[i]*1.1
            self.oText[i]->SetProperty, CHAR_DIMENSIONS=[3,3], LOCATION=loc, $
                                        STRINGS=strtrim(nm[i],2) + ' nm'
        endif
    endfor

    oModel->SetProperty, HIDE=isHidden, $
        UVALUE=isHidden

    oVolumeModel = self -> GetObjectByName('VolumeShearModel')
    oVolumeModel -> Add, oModel, POSITION=1
    self -> UpdateSlicePlanes, /NO_RENDER, /RESET

    if ~keyword_set(noDraw) then $
        self -> RenderScene

end



;------------------------------------------------------------------------------
;+
; Update the 3D slice plane positions.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @keyword NO_RENDER {in}{type=boolean}{optional}
;   If set, do not render.
;
; @keyword RESET {in}{type=boolean}{optional}
;   If set, reset the display.
;
; @keyword X_LOCATION {in}{type=number}{optional}
;   If present, the X plane location.
;
; @keyword Y_LOCATION {in}{type=number}{optional}
;   If present, the Y plane location.
;
; @keyword Z_LOCATION {in}{type=number}{optional}
;   If present, the Z plane location.
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::UpdateSlicePlanes, $
    NO_RENDER=noRender, $
    RESET=doReset, $
    X_LOCATION=xLoc, $
    Y_LOCATION=yLoc, $
    Z_LOCATION=zLoc

    oPALMModel = self -> GetObjectByName('PALM')
    if ~obj_valid(oPALMModel) then $
        return

    oPALMModel -> GetProperty, VOLUME_XRANGE=xRange, $
                               XCOORD_CONV=xs, $
                               VOLUME_YRANGE=yRange, $
                               YCOORD_CONV=ys, $
                               VOLUME_ZRANGE=zRange, $
                               ZCOORD_CONV=zs

; Reset
    if keyword_set(doReset) then begin
        xPad = (xRange[1]-xRange[0])/10.
        yPad = (yRange[1]-yRange[0])/10.
        zPad = (zRange[1]-zRange[0])/5.
        xPadRange = xRange + [-xPad,xPad]
        yPadRange = yRange + [-yPad,yPad]
        zPadRange = zRange + [-zPad,zPad]
        xPlane = [[0,yPadRange[0],zPadRange[0]], $
                  [0,yPadRange[1],zPadRange[0]], $
                  [0,yPadRange[1],zPadRange[1]], $
                  [0,yPadRange[0],zPadRange[1]], $
                  [0,yPadRange[0],zPadRange[0]]]
        self.oSliceLines[0] -> SetProperty, DATA=xPlane, $
            XCOORD_CONV=xs, $
            YCOORD_CONV=ys, $
            ZCOORD_CONV=zs
        self.oSlicePlanes[0] -> SetProperty, DATA=xPlane, $
            XCOORD_CONV=xs, $
            YCOORD_CONV=ys, $
            ZCOORD_CONV=zs
        yPlane = [[xPadRange[0],0,zPadRange[0]], $
                  [xPadRange[1],0,zPadRange[0]], $
                  [xPadRange[1],0,zPadRange[1]], $
                  [xPadRange[0],0,zPadRange[1]], $
                  [xPadRange[0],0,zPadRange[0]]]
        self.oSliceLines[1] -> SetProperty, DATA=yPlane, $
            XCOORD_CONV=xs, $
            YCOORD_CONV=ys, $
            ZCOORD_CONV=zs
        self.oSlicePlanes[1] -> SetProperty, DATA=yPlane, $
            XCOORD_CONV=xs, $
            YCOORD_CONV=ys, $
            ZCOORD_CONV=zs
        zPlane = [[xPadRange[0],yPadRange[0],0], $
                  [xPadRange[1],yPadRange[0],0], $
                  [xPadRange[1],yPadRange[1],0], $
                  [xPadRange[0],yPadRange[1],0], $
                  [xPadRange[0],yPadRange[0],0]]
        self.oSliceLines[2] -> SetProperty, DATA=zPlane, $
            XCOORD_CONV=xs, $
            YCOORD_CONV=ys, $
            ZCOORD_CONV=zs
        self.oSlicePlanes[2] -> SetProperty, DATA=zPlane, $
            XCOORD_CONV=xs, $
            YCOORD_CONV=ys, $
            ZCOORD_CONV=zs
    endif

; X-Location
    if n_elements(xLoc) GT 0 then begin
        xLocation = (xLoc>xRange[0])<xRange[1]
        self.oSlicePlanes[0] -> GetProperty, DATA=xPlane
        xPlane[0,*] = xLoc[0]
        self.oSliceLines[0] -> SetProperty, DATA=xPlane
        self.oSlicePlanes[0] -> SetProperty, DATA=xPlane
    endif

; Y-Location
    if n_elements(yLoc) GT 0 then begin
        yLocation = (yLoc>yRange[0])<yRange[1]
        self.oSlicePlanes[1] -> GetProperty, DATA=yPlane
        yPlane[1,*] = yLoc[0]
        self.oSliceLines[1] -> SetProperty, DATA=yPlane
        self.oSlicePlanes[1] -> SetProperty, DATA=yPlane
    endif

; Z-Location
    if n_elements(zLoc) GT 0 then begin
        zLocation = (zLoc>zRange[0])<zRange[1]
        self.oSlicePlanes[2] -> GetProperty, DATA=zPlane
        zPlane[2,*] = zLoc[0]
        self.oSliceLines[2] -> SetProperty, DATA=zPlane
        self.oSlicePlanes[2] -> SetProperty, DATA=zPlane
    endif

    if ~keyword_set(noRender) then $
        self -> RenderScene

end



;------------------------------------------------------------------------------
;+
; This method updates the trackball center and radius according to
; the current display.
;
; @keyword
;    ERROR_MESSAGE {out}{optional}
;      Set this keyword to a named variable to retrieve any error
;      messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::UpdateTrackball, $
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

    oModel = self -> GetObjectByName('MainModel')
    oWindow = self.oMainGUI -> GetObjectByName('3DWindow')
    oModel -> GetProperty, TRANSFORM=tm
    oWindow -> getProperty, DIMENSIONS=winDims
    ctm = oModel -> GetCTM(DESTINATION=oWindow)
    newCenter = matrix_multiply(ctm, [0,0,0,1], /ATRANSPOSE)
    newCenter = (newCenter[0:1] + 1) * winDims / 2
    self.oTrack -> Reset, newCenter, min(winDims/2)

end


;------------------------------------------------------------------------------
;+
; This method adjusts the zoom of the display
;
; @param
;  xy {in}{required}{type=double}
;    The xy position of the mouse cursor (sEvent.x, sEvent.y).
;
; @keyword
;    ERROR_MESSAGE {out}{optional}
;      Set this keyword to a named variable to retrieve any error
;      messages thrown.
; @keyword
;    NO_DRAW {in}{type=boolean}{optional}
;     Set this keyword to prevent display update.
; @keyword
;   START {in}{type=boolean}{optional}
;     Set this keyword to define the starting point, for example
;     on mouse down.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay::Zoom, xy, $
    ERROR_MESSAGE=errMsg, $
    NO_DRAW=noDraw, $
    RELEASE=release, $
    START=start

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
    doStart = keyword_set(start)
    oVolumeModel = self->GetObjectByName('VolumeModel')
    if doStart then begin
        widget_control, self.oMainGUI->Get('PALM_3DWindow'), $
            GET_VALUE=oWindow
        void = oWindow -> PickData(self, oVolumeModel, xy, dataXYZ)
        self.xyzStart = dataXYZ
        self.xy = xy
    endif

    self -> UpdateManip, $
        /DO_LOW, $
        /NO_DRAW, $
        RELEASE=release, $
        START=start

    ;  Scale the axes as well
    oAxesModel = self -> GetObjectByName('AxesModel')
    if ~(doStart || keyword_set(release)) then begin
        scale = (xy[1]-self.xy[1]) GT 0 ? $
            replicate(0.95,3) : replicate(1.05,3)
        if scale[0] GT 1 then begin
        ;
        ; Check how far we're already zoomed in....If it's too far do not zoom
        ; in any more
        ;
            widget_control, self.oMainGUI->Get('PALM_3DWindow'), $
                GET_VALUE=oWindow
            void = oWindow->PickData(self, oVolumeModel, [0,0], dataXYZ_0)
        ; Increase the x-value to allow more zooming
            void = oWindow->PickData(self, oVolumeModel, [20,0], dataXYZ_1)
            (self->GetObjectByName('PALM'))->GetProperty, NANOS_PER_CCD_PIXEL=Nanos
            if abs(dataXYZ_1[0]-dataXYZ_0[0])*Nanos LT 1.0 then begin
                if ~keyword_set(noDraw) then $
                    self->RenderScene
                return
            endif
        endif
        oVolumeModel -> Translate, -(self.xyzStart)[0], -(self.xyzStart)[1], 0;, -(self.xyzStart)[2]
        oVolumeModel -> Scale, scale[0], scale[1], scale[2]
        oAxesModel->Scale, scale[0], scale[1], scale[2]
        oVolumeModel -> Translate, (self.xyzStart)[0], (self.xyzStart)[1], 0;, (self.xyzStart)[2]
        oAxesModel->GetProperty, TRANSFORM=t
        oAxesModel->Translate, -0.75-t[3,0], -0.75-t[3,1], 0
        self.xy = xy
    endif else begin
        if self.DragQuality EQ 'HIGH' then $
            return
    endelse

    if ~keyword_set(noDraw) then $
        self -> RenderScene

end


;------------------------------------------------------------------------------
;+
; Define the member variables of the PALM_3DDisplay class, along with various internal
;  structures.
; @File_comments
; This file defines the PALM_3DDisplay class for the framework of the application.
;
; @field
;   DragQuality
;     A boolean value specifying the drag quality
; @field
;   oMainGUI
;     A reference to an instance of the PALM_MainGUI object
; @field
;   oSlicePlanes
;     An object array for the three slicing planes
; @field
;   oTrack
;     A reference to an instance of the Trackball object
; @field
;   verbose
;     A boolean field that when set will have error messages printed to
;     the IDL output log.
; @field
;   xy
;     A floating-point array for holding the current cursor position.
; @field
;   xyStart
;     A floating-point array for holding the position of the cursor when
;     a manipulator was activated (Used for zooming)
; @field
;   LastFileDirectory
;     The path to the file that was last opened for read or write.
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Global Services Group
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DDisplay__Define

    void = {PALM_3DDisplay, $
            Inherits IDLgrView,            $
            DragQuality       : '',        $
            emOpacityFunction : '',        $
            oAxes             : objarr(3), $
            oBuffer           : obj_new(), $
            oMainGUI          : obj_new(), $
            OpacityFunction   : '',        $
            oSliceLines       : objarr(3), $
            oSlicePlanes      : objarr(3), $
            oText             : objarr(3), $
            oTrack            : obj_new(), $
            verbose           : 0B,        $
            xy                : fltarr(2), $
            xyzStart          : fltarr(3), $
            LastFileDirectory : ''         $
           }

end
