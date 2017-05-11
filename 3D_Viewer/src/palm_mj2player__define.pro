;$File: //depot/gsg/HHMI/Phase2/src/palm_mj2player__define.pro $
;$Revision: #2 $
;$DateTime: 2009/06/04 13:22:51 $
;$Author: rkneusel $

;--------------------------------------------------------------
;+
; Clean up objects and frame buffer
;
; @History
;   Mar 2009, AB - ITT, Initial version <br>
;-
;--------------------------------------------------------------
pro palm_mj2player::Cleanup
    compile_opt idl2, logical_predicate
    ; Stop reading
    if obj_valid(self.oMJ) then void = self.oMJ->StopSequentialReading()
    ; Destroy objects
    self->IDL_Container::Cleanup
end

;--------------------------------------------------------------
;+
; Creates the GUI. This must be called only once per object.
;
; @Keyword
;   NO_BLOCK {in} {optional} {type=boolean} {default=0} 
;     Whether to block on xmanager.
;
; @Keyword
;   _REF_EXTRA
;     Passed along to WIDGET_BASE
;
; @History
;   Mar 2009, AB - ITT, Initial version <br>
;-
;--------------------------------------------------------------
pro palm_mj2player::CreateGui, NO_BLOCK=noblock, _REF_EXTRA=extra
    compile_opt idl2, logical_predicate

    ; determine initial params
    if obj_valid(self.oMJ) then begin
        self.oMJ->GetProperty, DIMENSIONS=dims, N_FRAMES=nf
    endif else begin
        dims = [500,500]
        nf = 1
    endelse

    ; top level base widget, use "EXTRA" keywords to set
    ; group_leader, ...
    tlb = widget_base(UVALUE=self, /ROW, /TLB_KILL_REQUEST_EVENTS, $
        MBAR=wmbar, _STRICT_EXTRA=extra, $
        TITLE='Motion JPEG2000 Player, Howard Hughes Medical Institute')
    ; menu
    file = widget_button(wmbar, VALUE='File', /MENU)
    fopen = widget_button(file, VALUE='Open...', UNAME='open')
    fexit = widget_button(file, VALUE='Exit', UNAME='exit', /SEPARATOR)
    ; draw widget
    wdraw = widget_draw(tlb, XSIZE=dims[0], YSIZE=dims[1], $
        GRAPHICS_LEVEL=2, RETAIN=0, UNAME='draw', $
        /EXPOSE_EVENTS)
    ; controls and buttons
    wCol = widget_base(tlb, /COLUMN)
    wpos = widget_slider(wCol, VALUE=0, MIN=0, MAX=(nf-1)>1, $
        UNAME='frame', /DRAG)
    wrow = widget_base(wCol, /ROW, XPAD=0, YPAD=0)
    wl = lonarr(3)
    wl[0] = widget_label(wrow, VALUE='# of frames: ', /ALIGN_RIGHT)
    wnf = widget_text(wrow, VALUE=string(nf, FORMAT='(i0)'), $
        UNAME='num_frames')
    wrow = widget_base(wCol, /ROW, XPAD=0, YPAD=0)
    wl[1] = widget_label(wrow, VALUE='subset: ', /ALIGN_RIGHT)
    wsub = widget_text(wrow, VALUE=string([0, nf-1], FORMAT='(i0," - ",i0)'), $
        /EDITABLE, UNAME='subset', /TRACKING_EVENTS)
    wrow = widget_base(wCol, /ROW, XPAD=0, YPAD=0)
    wl[2] = widget_label(wrow, VALUE='Speed Factor:', /ALIGN_RIGHT)
    wts = widget_text(wrow, VALUE='1.00', UNAME='timescale', /EDITABLE, $
        /TRACKING_EVENTS)
    wsl = widget_slider(wrow, VALUE=50, MIN=0, MAX=100, $
        UNAME='timescaleslider', /SUPPRESS_VALUE, /DRAG)
    wblank = widget_label(wCol, VALUE=' ') ; leave some space
    wblank = widget_label(wCol, VALUE=' ') ; leave some space
    wrow = widget_base(wCol, /ROW, XPAD=0, YPAD=0)
    wplay = widget_button(wrow, VALUE='>', UNAME='play', XSIZE=90, $
        TOOLTIP='Play')
    wpause = widget_button(wrow, VALUE='||', UNAME='pause', XSIZE=90, $
        TOOLTIP='Pause')
    wstep = widget_button(wrow, VALUE='.', UNAME='step', XSIZE=90, $
        TOOLTIP='Single Step')
    widget_control, tlb, /REALIZE
    maxw = 0
    for i=0, n_elements(wl)-1 do begin
        geom = widget_info(wl[i], /GEOMETRY)
        maxw >= geom.scr_xsize
    endfor
    for i=0, n_elements(wl)-1 do begin
        widget_control, wl[i], SCR_XSIZE=maxw
    endfor
    widget_control, wdraw, GET_VALUE=oWin
    self->SetWindow, oWin
    widget_control, wplay, /INPUT_FOCUS
    
    ; Always run a timer, to keep track
    widget_control, tlb, TIMER=0.1

    ; Show first frame if valid
    if obj_valid(self.oMJ) then self.isplaying = 1

    ; Start xmanager
    xmanager, 'mj2player', tlb, NO_BLOCK=noblock, $
        EVENT_HANDLER='genericclassevent'
end


;--------------------------------------------------------------
;+
; Main event handler entry point
;
; @Param
;   sEvent {in} {required} {type=struct}
;     IDL event structure.
;
; @History
;   Mar 2009, AB - ITT, Initial version <br>
;-
;--------------------------------------------------------------
pro palm_mj2player::Event, sEvent
    compile_opt idl2, logical_predicate
    catch, err
    if err then begin
        catch, /CANCEL
        help, /LAST_MESSAGE
        message, /RESET
        return
    endif
    ; Event info
    type = tag_names(sEvent, /STRUCTURE_NAME)
    uname = widget_info(sEvent.id, /UNAME)

    ; Branching
    if (type eq 'WIDGET_KILL_REQUEST') then begin
        ; Destroy widgets and clean objects
        widget_control, sEvent.top, /DESTROY
        obj_destroy, self
    endif
    if (uname eq 'exit') then begin
        ; Destroy widgets and clean objects
        widget_control, sEvent.top, /DESTROY
        obj_destroy, self
    endif
    if (type eq 'WIDGET_DRAW' && sEvent.type eq 4) then begin
        self.oWin->Draw, self.oView
    endif
    if (uname eq 'play') then begin
        ; set the flag, the timer event does the rest
        self.isplaying = -1
    endif
    if (uname eq 'pause') then begin
        ; set the flag, the timer event does the rest
        self.isplaying = 0
    endif
    if (uname eq 'step') then begin
        ; set the flag, the timer event does the rest
        self.isplaying = 1
    endif
    if (uname eq 'open') then begin
        ; Start in the same directory as previous file, if any
        if obj_valid(self.oMJ) then begin
            self.oMJ->GetProperty, FILENAME=oldfile
            dir = file_dirname(oldfile)
        endif
        file = dialog_pickfile(PATH=dir)
        if (file ne '') then begin
            ok = self->LoadFile(file, TLB=sEvent.top)

            ; Play a single frame to show initial image
            if ok then self.isplaying = 1
        endif
    endif
    if (type eq 'WIDGET_TIMER') then begin
        ; The timer defaults to 0.1 if we are paused
        dt = 0.1
        ok = 1

        ; If we are in play mode, then the delay will be returned
        if self.isplaying then begin
            ok = self->Play(TIMER=dt)
            if obj_valid(self.oMJ) then begin
                self.oMJ->GetProperty, CURRENT_FRAME=cf
                wframe = widget_info(sEvent.top, FIND_BY_UNAME='frame')
                widget_control, wframe, SET_VALUE=cf
            endif
        endif
        if ~ok then begin
            ; This is where we are unable to keep
            ; up with the desired playback speed
            ; so wait a small interval and try again
            dt = 0.01
        endif
        widget_control, sEvent.top, TIMER=dt
    endif
    if (uname eq 'timescaleslider') then begin
        ; This sets the playback speed multiplier
        wtxt = widget_info(sEvent.top, FIND_BY_UNAME='timescale')
        ; compute speed factor from integer using exp
        self.rate = exp((sEvent.value - 50) * (alog(10)/50))
        widget_control, wtxt, SET_VALUE=string(self.rate, FORMAT='(f0.2)')
    endif
    if (uname eq 'timescale') then begin
        ; This sets the playback speed multiplier
        widget_control, sEvent.id, GET_VALUE=str
        rate = float(str[0])
        if (rate gt 0.) then begin
            self.rate = rate
            ; compute slider integer using ln function
            val = round(alog(self.rate) * (50/alog(10)) + 50) < 100 > 0

            ; Set slider position
            wslide = widget_info(sEvent.top, FIND_BY_UNAME='timescaleslider')
            widget_control, wslide, SET_VALUE=val
        endif else begin
            widget_control, sEvent.id, $
                SET_VALUE=string(self.rate, FORMAT='(f0.2)')
        endelse
    endif
    if (uname eq 'subset' && obj_valid(self.oMJ)) then begin
        widget_control, sEvent.id, GET_VALUE=str
        sub = long(strtok(str[0], '-,', /EXTRACT, COUNT=n))
        self.oMJ->GetProperty, N_FRAMES=nf
        if (n eq 2 && $
            sub[0] le sub[1] && $
            sub[0] ge 0 && $
            sub[1] lt nf) then begin

            ; Valid subset update only do work if change
            if ~array_equal(self.subset, sub) then begin
                self.subset = sub
                catch, err
                if ~err then begin
                    void = self.oMJ->StopSequentialReading()
                endif
                catch, /CANCEL
                void = self.oMJ->StartSequentialReading( $
                    START_FRAME_NUMBER=self.subset[0], $
                    STOP_FRAME_NUMBER=self.subset[1] + 1)
                wslide = widget_info(sEvent.top, FIND_BY_UNAME='frame')
                widget_control, wslide, GET_VALUE=val
                widget_control, wslide, SET_SLIDER_MIN=self.subset[0], $
                    SET_SLIDER_MAX=self.subset[1], SET_VALUE=val > self.subset[0] < self.subset[1]
            endif
        endif
        widget_control, sEvent.id, $
            SET_VALUE=string(self.subset, FORMAT='(i0," - ",i0)')
    endif
    if (uname eq 'frame') then begin
        ; Stop playing, and jump to the requested frame
        self->GotoFrame, sEvent.value
    endif
end


;--------------------------------------------------------------
;+
; Skips to any frame, out of sequence reading.
;
; @Param
;   num {in} {required} {type=int}
;     The frame number to jump to.
;
; @History
;   Mar 2009, AB - ITT, Initial version <br>
;-
;--------------------------------------------------------------
pro palm_mj2player::GotoFrame, num
    compile_opt idl2, logical_predicate
    
    ; Catch is needed for kakadu error if we try to read to quickly
    try = 0
    catch, err
    if err then begin
        try++
        if (try eq 10) then begin
            catch, /CANCEL
            return
        endif
        ; Wait a little and try again
        ; half second maximum wait
        wait, 0.05
    endif
    ; Get the requested frame
    data = self.oMJ->GetData(num)
    catch, /CANCEL

    ; Setting the step to 1 causes the next GetSequentialData
    ; to get back in sync, otherwise it jumps back to where
    ; it was before dragging the slider
    self.step = 1

    ; Stop playing to avoid conflict while dragging the slider
    self.isplaying = 0

    ; Display the frame
    self.oImage->SetProperty, DATA=data
    self.oWin->Draw, self.oView
end

;--------------------------------------------------------------
;+
; Life cycle method called from OBJ_NEW.
;
; @Param
;   filename {in} {optional} {type=string}
;     A filename to an MJ2 file to be displayed initially.
;
; @History
;   Mar 2009, AB - ITT, Initial version <br>
;
; @Returns
;   Returns 1 for success and 0 for failure.
;-
;--------------------------------------------------------------
function palm_mj2player::Init, filename
    compile_opt idl2, logical_predicate

    ; Member variable initialization
    self.rate = 1.

    ; Super class init
    if ~self->IDL_Container::Init() then return, 0

    ; Object hierarchy creation
    self.oView = obj_new('IDLgrView', NAME='AnimationView', COLOR=[0,0,0])
    self->Add, self.oView
    self.oModel = obj_new('IDLgrModel', NAME='Model')
    self.oView->Add, self.oModel
    self.oImage = obj_new('IDLgrImage', NAME='Image')
    self.oModel->Add, self.oImage

    ; Load file if specified
    if keyword_set(filename) then void = self->LoadFile(filename)
    return, 1
end

;--------------------------------------------------------------
;+
; Opens a new Motion JPEG2000 file
;
; @Param
;   filename {in} {optional} {type=string}
;     The filename of the mj2 file to be opened.
;
; @Keyword
;   TLB {in} {optional} {type=long}
;     The top level base ID, if supplied, then informational
;     widgets will be updated to reflect the file that was loaded.
;
; @History
;   Mar 2009, AB - ITT, Initial version <br>
;
; @Returns
;   Returns 1 for success and 0 for failure.
;-
;--------------------------------------------------------------
function palm_mj2player::LoadFile, filename, TLB=tlb
    compile_opt idl2, logical_predicate

    ; Stop reading
    if obj_valid(self.oMJ) then begin
        catch, err
        if ~err then void = self.oMJ->StopSequentialReading()
        catch, /CANCEL
        self->Remove, self.oMJ
        obj_destroy, self.oMJ
    endif

    ; Create new object
    catch, err
    if ~err then begin
        self.oMJ = obj_new('IDLffMJPEG2000', filename, /PERSISTENT)
        void = self.oMJ->StartSequentialReading()
    endif else begin
        help, /LAST_MESSAGE
        void = dialog_message('Failed to open file: ' + filename)
        if obj_valid(self.oMJ) then obj_destroy, self.oMJ
        return, 0
    endelse
    catch, /CANCEL
    self->Add, self.oMJ

    ; Verify transform
    self.oMJ->GetProperty, DIMENSIONS=dims, N_FRAMES=nf
    self.oModel->GetProperty, TRANSFORM=otx
    tx = [ [2d / dims[0], 0, 0, -1], $
           [0, 2d / dims[1], 0, -1], $
           [0, 0, 1d, 0], $
           [0, 0, 0, 1d] ]
    if ~array_equal(tx, otx) then begin
        self.oModel->SetProperty, TRANSFORM=tx
    endif

    ; Resize window if needed
    if obj_valid(self.oWin) then begin
        self.oWin->GetProperty, DIMENSIONS=windim
        if ~array_equal(windim, dims) then begin
            self.oWin->SetProperty, DIMENSIONS=dims
        endif
    endif

    ; Set info
    if n_elements(tlb) && widget_info(tlb, /VALID_ID) then begin
        wnum = widget_info(tlb, FIND_BY_UNAME='num_frames')
        widget_control, wnum, SET_VALUE=string(nf, FORMAT='(i0)')
        wf = widget_info(tlb, FIND_BY_UNAME='frame')
        widget_control, wf, SET_VALUE=0, SET_SLIDER_MIN=0, $
            SET_SLIDER_MAX=(nf-1) > 1
        wsub = widget_info(tlb, FIND_BY_UNAME='subset')
        widget_control, wsub, SET_VALUE=string([0,nf-1], FORMAT='(i0," - ",i0)')
    endif
    return, 1
end

;--------------------------------------------------------------
;+
; Displays the next frame.
;
; @Keyword
;   TIMER {out} {optional} {type=double}
;     The amount of time needed to wait before displaying
;     the next frame.
;
; @History
;   Mar 2009, AB - ITT, Initial version <br>
;
; @Returns
;   Returns 1 for success and 0 for failure.
;-
;--------------------------------------------------------------
function palm_mj2player::Play, TIMER=dt
    compile_opt idl2, logical_predicate
    t0 = systime(1)

    if ~obj_valid(self.oMJ) then return, 0

    ; Catch errors with reading frame
    catch, err
    if err then begin
        catch, /CANCEL
        help, /LAST_MESSAGE
        return, 0
    endif
    ; Get info
    self.oMJ->GetProperty, TIMESCALE=ts, N_FRAMES=nf, CURRENT_FRAME=cf

    ; Display next frame
    fbi = self.oMJ->GetSequentialData(data, FRAME_PERIOD=fp, STEP=self.step)
    if (fbi ne -1) then begin
        ; Reset step to point to next frame
        self.step = 0
        ; Set data
        self.oImage->SetProperty, DATA=data
        ; Release data
        self.oMJ->ReleaseSequentialData, fbi
        ; Render
        self.oWin->Draw, self.oView
        ; Compute delay time, avoid zero because it causes 100% cpu
        dt = (fp / double(ts) / self.rate - (systime(1) - t0)) > 0.01d
        ; isplaying > 0 means play n frames and stop
        ; isplaying < 0 means play indefinetly
        if (self.isplaying gt 0) then self.isplaying--
        return, 1
    endif
    return, 0
end

;--------------------------------------------------------------
;+
; Sets the IDLgrWindow reference.
;
; @Param
;   oWin {in} {required} {type=objref}
;     A reference to the IDLgrWindow object.
;
; @History
;   Mar 2009, AB - ITT, Initial version <br>
;-
;--------------------------------------------------------------
pro palm_mj2player::SetWindow, oWin
    compile_opt idl2, logical_predicate
    self.oWin = oWin
end

;--------------------------------------------------------------
;+
; Class definition routine.
;
; @Field
;   oWin
;     A reference to the IDLgrWindow object.
;
; @Field
;   oView
;     A reference to the IDLgrView object.
;
; @Field
;   oModel
;     A reference to the IDLgrModel object.
;
; @Field
;   oImage
;     A reference to the IDLgrImage object.
;
; @Field
;   oMJ
;     A reference to the IDLffMJPEG2000 object.
;
; @Field
;   isplaying
;     An integer indicating how many frames to play.
;     isplaying = 0 : Movie is paused.
;     isplaying = -1 : Play indefinetly.
;     isplaying > 0 : Play "isplaying" number of frames, then stop
;
; @Field
;   rate
;     Playback speed scale factor. A value of 1.0 means
;     playing back at the rate stored in the file. A
;     value of 2.0 means twice as fast, 0.5 means half.
;
; @Field
;   subset
;     The current subset of frames being animated through.
;
; @Field
;   step
;     A step applied when reading data sequentially.
;     Documented as number of frames to skip.
;     Could be -1 to play backwards, but that has not
;     been exposed to the user.
;
; @Author
;   Atle Borsholm
;
; @History
;   Mar 2009, AB - ITT, Initial version <br>
;-
;--------------------------------------------------------------
pro palm_mj2player__define
    compile_opt idl2, logical_predicate
    s = { palm_mj2player $
        , Inherits IDL_Container $
        , oWin: obj_new() $
        , oView: obj_new() $
        , oModel: obj_new() $
        , oImage: obj_new() $
        , oMJ: obj_new() $
        , isplaying: 0 $
        , rate: 0. $
        , subset: lonarr(2) $
        , step: 0 $
    }
end
