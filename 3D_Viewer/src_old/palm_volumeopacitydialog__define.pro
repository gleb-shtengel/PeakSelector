;------------------------------------------------------------------------------
;+
; This method handles button events
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event a the PALM_VolumeOpacityDialog button widget.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::ButtonEvent, event

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST
        return
    endif

    widget_control, event.id, GET_UVALUE=uval
    case uval of
        'apply': begin
            if self.EM then begin
                self.o3DDisplay->SetProperty, EM_OPACITY_FUNCTION=self.OpacityFunction, $
                    EM_OPACITY_TABLE=self->GetOpacityTable()
            endif else begin
                self.o3DDisplay -> SetProperty, OPACITY_FUNCTION=self.OpacityFunction, $
                    OPACITY_TABLE=self->GetOpacityTable()
            endelse
            self.o3DDisplay -> RenderScene
        end
        'close': self -> Destruct
        'open': self -> OpenOpacity
        'save': self -> SaveOpacity
        else:
    endcase

end

;------------------------------------------------------------------------------
;+
; This method calculates a Gaussian
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::CalculateGaussian

    width = self.OpacityRange[1]-self.OpacityRange[0]
    center = (self.OpacityRange[1]-self.OpacityRange[0])/2 + self.OpacityRange[0]
    range = self -> GetRange()
    self.OpacityRamp = byte((range[1]-range[0])* $
        exp(-0.5*((findgen(256)-center)/width)^2)+range[0])
end

;------------------------------------------------------------------------------
;+
; This method calculates the minear ramp
;
; @History
;   June, 2009 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::CalculateLinearRamp, $
    CALCULATE_RANGE=CalcRange, $
    DECREASING=decreasing, $
    TABLE=OpacityTable, $
    Y_RANGE=range

    decreasing = keyword_set(decreasing)
    CalcRange = keyword_set(CalcRange)
    if n_elements(range) EQ 0 then begin
        if CalcRange then $
            range = [min(self.OpacityRamp, MAX=maxVal), maxVal] $
        else $
            range = self->GetRange()
    endif
    if CalcRange then begin
        indexMin = where(self.OpacityRamp LE range[0])
        indexMax = where(self.OpacityRamp GE range[1])
        self.OpacityRange = decreasing ? $
            [max(indexMax),min(indexMin)] : $
            [max(indexMin),min(indexMax)]
    endif

    index = decreasing ? [1,0] : [0,1]
    p0 = float([self.OpacityRange[0],range[index[0]]])
    p1 = float([self.OpacityRange[1],range[index[1]]])
    m = (p1[1]-p0[1])/(p1[0]-p0[0])
    b = p0[1]-m*p0[0]
    self.OpacityRamp = findgen(256)*m+b
    if arg_present(OpacityTable) then $
        OpacityTable = (self.OpacityRamp>range[0])<range[1]

end

;------------------------------------------------------------------------------
;+
; This lifecycle method is called during the destruction of the
; PALM_VollumeOpacityDialog object via OBJ_DESTROY.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::Cleanup

    self -> Destruct

end


;------------------------------------------------------------------------------
;+
; This method constructs the widgets used by PALM_VolumeOpacityDialog
;
; @Returns
;   1 for success and 0 fir failure
;
; @Param
;   GroupLeader {in}{optional}{type=long}
;     Set this parameter to the widget ID of the gruop leader
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
; @keyword
;   OPACITY_FUNCTION {in}{optional}{type=string}
;     Set this keyword to a string specifying the name of the opacity
;     function used.  Acceptable values are: "Free hand", "Gaussian",
;     "Linear (Decreasing)", "Linear (Increasing)"
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_VolumeOpacityDialog::ConstructWidgets, GroupLeader, $
    ERROR_MESSAGE=errMsg, $
    OPACITY_FUNCTION=OpacityFunction

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

    drawDims = [500,500]

    if n_elements(GroupLeader) EQ 0 then $
        GroupLeader = 0
    self.tlb = widget_base( $
        /COLUMN, $
        EVENT_PRO='GenericClassEvent', $
        GROUP_LEADER=GroupLeader, $
        KILL_NOTIFY='GenericClassKillNotify', $
        MBAR=menu, $
        NOTIFY_REALIZE='GenericClassNotifyRealize', $
        TITLE=self.EM ? 'EM Volume Opacity':'Molecule Volume Opacity', $
        TLB_FRAME_ATTR=1, $
        UVALUE=self)
; Menu
    wFile = widget_button(menu, $
        /MENU, $
        VALUE='File')
    wButton = widget_button(wFile, $
        UVALUE='open', $
        VALUE='Open Opacity Table...')
    wButton = widget_button(wFile, $
        UVALUE='save', $
        VALUE='Save Opacity Table...')
    wDraw = widget_draw(self.tlb, $
        /BUTTON_EVENTS, $
        /EXPOSE_EVENTS, $
        GRAPHICS_LEVEL=2, $
        /MOTION_EVENTS, $
        UNAME='Draw', $
        XSIZE=drawDims[0], $
        YSIZE=drawDims[1])
    wBase = widget_base(self.tlb, /ROW)
; Function droplist
    functionValues = ['Free hand', $
                      'Gaussian', $
                      'Linear (Decreasing)', $
                      'Linear (Increasing)']
    wDroplist = widget_droplist(wBase, $
        UNAME='OpacityFunction', $
        UVALUE='OpacityFunction', $
        VALUE=functionValues)

    index = where(functionValues EQ self.OpacityFunction, count)
    if count EQ 0 then $
        index = 3
    self.OpacityFunction = functionValues[index]
    widget_control, wDropList, SET_DROPLIST_SELECT=index[0]
    wSlider = cw_slider(wBase, $
        /DRAG, $
        /DUAL, $
        MAXIMUM=255, $
        MINIMUM=0, $
        UNAME='DualSlider', $
        UVALUE='DualSlider')
    wButton = widget_button(wBase, $
        UVALUE='apply', $
        VALUE='Apply')
    wButton = widget_button(wBase, $
        UVALUE='close', $
        VALUE='Close')

    widget_control, self.tlb, /REALIZE

    return, 1

end


;------------------------------------------------------------------------------
;+
; This method has the primary responsibility for cleaning up the
; PALM_VolumeOpacityDialog object at the end of its lifecycle.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::Destruct

    if (widget_info(self.tlb, /VALID_ID)) then begin
        widget_control, self.tlb, KILL_NOTIFY = ''
        widget_control, self.tlb, /DESTROY
    endif

    oWindow = self -> GetObjectByName('WINDOW')
    obj_destroy, oWindow

end


;------------------------------------------------------------------------------
;+
; This method handles events from the draw widget
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event a the PALM_VolumeOpacityDialog button widget.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::DrawEvent, event
    case event.type of
        0: begin
            if ~(self->GetDataCoordinates([event.x,event.y], dataXY)) then $
                return
            self.HoldLeft = 1B
            self.X = dataXY[0]
            if strupcase(self->GetFunction()) EQ 'GAUSSIAN' then begin
                if self.OpacityRamp[dataXY[0]] LE dataXY[1] then $
                    self.HoldLeft++
            endif else $
                self -> UpdatePlot, round(dataXY)
        end
        1:begin
            self.HoldLeft = 0B
        end
        2:begin
            if (self.HoldLeft EQ 0) then $
                return
            if ~(self->GetDataCoordinates([event.x,event.y], dataXY)) then $
                return
            self -> UpdatePlot, round(dataXY)
        end
        4: self -> RenderScene
        else:
    endcase

end


;------------------------------------------------------------------------------
;+
; This method handles events from a droplist widget
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event a the PALM_VolumeOpacityDialog button widget.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::DroplistEvent, event

    widget_control, event.id, GET_UVALUE=uval
    case uval of
        'OpacityFunction': self -> UpdateOpacityFunction
        else:
    endcase

end


;------------------------------------------------------------------------------
;+
; This method handles events from the dual-headed slider
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event a the PALM_VolumeOpacityDialog button widget.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::DualSliderEvent, event

    compile_opt StrictArr
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST
        return
    endif

    case strupcase(self.OpacityFunction) of
        'FREE HAND': begin
            self.OpacityRamp = (self.OpacityRamp>event.value[0])<event.value[1]
        end
        'GAUSSIAN': begin
           self->CalculateGaussian
        end
        else: begin
        ;
        ; Linear
        ;
            self->UpdateRange
        end
    endcase

    self->UpdatePlot

end


;------------------------------------------------------------------------------
;+
; Main event handler
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event a the PALM_VolumeOpacityDialog button widget.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::Event, event

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST
        return
    endif

    case tag_names(event, /STRUCT) of
        'WIDGET_BUTTON': self -> ButtonEvent, event
        'WIDGET_DRAW': self -> DrawEvent, event
        'WIDGET_DROPLIST': self -> DroplistEvent, event
        else: self -> DualSliderEvent, event
    endcase

end


;------------------------------------------------------------------------------
;+
; This method converts device coordinates to data coordinates
;
; @Returns
;   1 for success and 0 otherwise
;
; @Param
;   XY {in}{required}{type=intarr)
;     A 2-element vector defining an (x,y)-position in device coordinates
; @Param
;   dataXY {out}{type=fltarr}
;     The data coordinates of XY
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_VolumeOpacityDialog::GetDataCoordinates, xy, dataXY

    if n_elements(xy) LT 2 then $
        return, 0

    oWindow = self -> GetObjectByName('WINDOW')
    oWindow -> GetProperty, GRAPHICS_TREE=oView
    oModel = oView -> GetByName('model')
    void = oWindow -> PickData(oView, oModel, xy[0:1], dataXY)
    dataXY = dataXY[0:1]
    if (dataXY[0] LT 0) OR (dataXY[0] GT 255) then $
        return, 0

    return, 1

end


;------------------------------------------------------------------------------
;+
; This method returns a reference to a specified object.  If no match is found
; a null object will be returned
;
; @param
;    name {in}{type=string}{required}
;      The name of the object reference to be retrieved.
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_VolumeOpacityDialog::GetObjectByName, name, $
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
        return, obj_new()
    endif

    if ~widget_info(self.tlb, /VALID) then $
        return, obj_new()

    case strupcase(name) of
        'MODEL': begin
            oReturn = (self->GetObjectByName('VIEW'))->GetByName('model')
        end
        'PLOT': begin
            oReturn = (self->GetObjectByName('VIEW'))->GetByName('model/plot')
        end
        'VIEW': begin
            (self->GetObjectByName('WINDOW'))->GetProperty, $
                GRAPHICS_TREE=oReturn
        end
        'WINDOW': begin
            wDraw = widget_info(self.tlb, FIND_BY_UNAME='Draw')
            if ~widget_info(wDraw, /VALID) then $
                return, obj_new()
            widget_control, wDraw, GET_VALUE=oReturn
        end
        else: oReturn = obj_new()
    endcase

    return, oReturn

end


;------------------------------------------------------------------------------
;+
; Gets the current opacity table function
;
; @Returns
;   A string containing the name of the current opacity table function
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_VolumeOpacityDialog::GetFunction

    wDroplist = widget_info(self.tlb, FIND_BY_UNAME='OpacityFunction')
    widget_control, wDroplist, GET_VALUE=OpacityFunctions
    return, OpacityFunctions[widget_info(wDroplist, /DROPLIST_SELECT)]

end

;------------------------------------------------------------------------------
;+
; This method gets the opacity table stored in the IDLgrPlot object
;
; @History
;   June, 2009 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_VolumeOpacityDialog::GetOpacityTable
    (self->GetObjectByName('PLOT'))->GetProperty, DATA=data
    return, reform(data[1,*])
end

;------------------------------------------------------------------------------
;+
; Gets the y-range from the dual slider
;
; @Returns
;   A 2-element vector containing the range of the dual-headed slider
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_VolumeOpacityDialog::GetRange

    wSlider = widget_info(self.tlb, FIND_BY_UNAME='DualSlider')
    if ~widget_info(wSlider, /VALID) then $
        return, [0,255]
    cw_slider_get, wSlider, VALUE=value
    return, value

end



;------------------------------------------------------------------------------
;+
; This method initializes the PALM_VolumeOpacityTable object class
;
; @Returns
;   1 for success and 0 otherwise
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
; @keyword
;   OPACITY_FUNCTION {in}{optional}{type=string}
;     Set this keyword to a string specifying the name of the opacity
;     function used.  Acceptable values are: "Free hand", "Gaussian",
;     "Linear (Decreasing)", "Linear (Increasing)"
; @Keyword
;   OPACITY_TABLE {in}{optional}{type=bytarr}
;     Set this keyword to a 256-element byte array defining an opacity table
; @keyword
;    VERBOSE {in}{type=boolean}{optional}
;      Setting this keyword will result in the object displaying any
;      error messages to the IDL output log.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_VolumeOpacityDialog::Init, GroupLeader, o3DDisplay, $
    EM_VOLUME=useEM, $
    ERROR_MESSAGE=errMsg, $
    OPACITY_FUNCTION=OpacityFunction, $
    OPACITY_TABLE=OpacityTable, $
    VERBOSE=verbose

    compile_opt idl2
    on_error, 2
    
    self.verbose = keyword_set(verbose)
    
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

    if ~obj_valid(o3DDisplay) then $
        return, 0
    if ~obj_isa(o3DDisplay, 'PALM_3DDisplay') then $
        return, 0

    self.EM = keyword_set(useEM)
    if self.EM then begin
        o3DDisplay->GetProperty, EM_OPACITY_FUNCTION=OpacityFunction, $
            EM_OPACITY_TABLE=OpacityTable
    endif else begin
        o3DDisplay->GetProperty, OPACITY_FUNCTION=OpacityFunction, $
            OPACITY_TABLE=OpacityTable
    endelse

    self.o3DDisplay = o3DDisplay
    self.OpacityFunction = n_elements(OpacityFunction) GT 0 ? OpacityFunction[0] : 'Linear (Increasing)'
    OriginalOpacityTable = n_elements(OpacityTable) GE 256 ? $
        OpacityTable[0:255] : bindgen(256)
    self.OpacityRamp = OriginalOpacityTable
    self.HoldLeft = 0B

    if ~(self->ConstructWidgets(GroupLeader, ERROR_MESSAGE=errMsg)) then $
        return, 0

    return, 1

end


;------------------------------------------------------------------------------
;+
; This method initializes the graphics objects used by PALM_VolumeOpacityDialog
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
function PALM_VolumeOpacityDialog::InitializeDisplay, $
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

    oAxisX = obj_new('IDLgrAxis', 0, $
        /EXACT, $
        MAJOR=6, $
        RANGE=[0,255])
    oAxisY = obj_new('IDLgrAxis', 1, $
        /EXACT, $
        MAJOR=6, $
        RANGE=[0,255])
    oPlot = obj_new('IDLgrPlot', self.OpacityRamp, $
        NAME='plot', $
        THICK=2)
    oModel = obj_new('IDLgrModel', $
        NAME='model')
    oView = obj_new('IDLgrView', VIEWPLANE_RECT=[-20,-10,290,280])
    oModel -> Add, oAxisX
    oModel -> Add, oAxisY
    oModel -> Add, oPlot
    oView -> Add, oModel
    oWindow = self -> GetObjectByName('WINDOW')
    oWindow -> SetProperty, GRAPHICS_TREE=oView
    oWindow -> Draw

    return, 1

end


;------------------------------------------------------------------------------
;+
; This method is called when the top level base is realized
;
; @Param
;   tlb {in}{required}{type=long}
;     The widget ID of the top level base
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::Notify_Realize, tlb

    if ~(self->InitializeDisplay(ERROR_MESSAGE=ErrMsg)) then begin
    ; Something failed initializing the display...Need to handle this case

    endif
    case strupcase(self->GetFunction()) of
        'FREE HAND': begin
            yRange = [0,255]
            self.OpacityRange = [0,255]
        end
        'GAUSSIAN': begin
            x = findgen(256)
            gFit = gaussfit(x, self.OpacityRamp, coeffs, $
                NTERMS=3)
            delta = coeffs[2]/2
            self.OpacityRange = [coeffs[1]-delta,coeffs[1]+delta]
            yRange = [min(self.OpacityRamp, MAX=maxVal), maxVal]
        end
        'LINEAR (DECREASING)': begin
            self->CalculateLinearRamp, /CALCULATE_RANGE, $
                /DECREASING, Y_RANGE=yRange
        end
        'LINEAR (INCREASING)': begin
            self->CalculateLinearRamp, /CALCULATE_RANGE, $
                Y_RANGE=yRange
        end
        else: begin
            yRange = [0,255]
            self.OpacityRange = [0,255]
        end
    endcase

    widget_control, widget_info(self.tlb, FIND_BY_UNAME='DualSlider'), $
        SET_VALUE=yRange

    XManager, 'GenericClass', tlb, $
        EVENT_HANDLER = 'GenericClassEvent'

end


;------------------------------------------------------------------------------
;+
; This method allows the user to select a saved opacity table to be loaded
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::OpenOpacity

    compile_opt StrictArr
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if n_elements(lun) GT 0 then $
            free_lun, lun
        if self.verbose then $
            help, /LAST
        return
    endif

    file = dialog_pickfile(FILTER='*.txt', $
        /MUST_EXIST, $
        TITLE='Open Opacity Table')
    if file EQ '' then $
        return

    if file_lines(file) NE 257 then begin
        ErrMsg = ['Invalid opacity table file.', $
                  'File must contain 257 rows', $
                  'The first row contains the name of the opacity function.', $
                  'Acceptable values are: ["Free hand","Gausssan","Linear (Decreasing)","Linear (Increasing)]', $
                  '', 'The remaining 256 lines contain  the opacity function values:', $
                  '  One value per line', $
                  '  Values must be in the range [0,255]']
        void = dialog_message(ErrMsg, /ERROR)
        return
    endif

    functionType=''
    temp = intarr(256)
    openr, lun, file, /GET_LUN
    readf, lun, functionType
    readf, lun, temp
    free_lun, lun

    index = where(temp LT 0, count)
    if count GT 0  then begin
        msg = ['Values less than 0 found', $
               'These values will be set to 0', $
               'Continue?']
        if strlowcase(dialog_message(msg, /QUESTION)) EQ 'no' then $
            return
        temp[index] = 0
    endif

    index = where(temp GT 255, count)
    if count GT 0  then begin
        msg = ['Values greater than 255 found', $
               'These values will be set to 255', $
               'Continue?']
        if strlowcase(dialog_message(msg, /QUESTION)) EQ 'no' then $
            return
        temp[index] = 255
    endif
; Set the function type to freehand
    wDroplist = widget_info(self.tlb, FIND_BY_UNAME='OpacityFunction')
    widget_control, wDroplist, GET_VALUE=dlValues

    self.OpacityFunction = functionType
    index = (where(dlValues EQ functionType))[0]
    widget_control, wDroplist, SET_DROPLIST_SELECT=index
;    widget_control, widget_info(self.tlb, FIND_BY_UNAME='DualSlider'), MAP=0
; Update the opacity table

    (self->GetObjectByName('PLOT'))->SetProperty, DATAY=byte(temp)
    if strupcase(functionType) NE 'FREE HAND' then begin
        yRange = [min(temp, MAX=maxVal), maxVal]
        if strupcase(functionType) NE 'GAUSSIAN' then begin
        ;
        ; Linear
        ;
            self.OpacityRamp = temp
            self->CalculateLinearRamp, /CALCULATE_RANGE, $
                DECREASING=decreasing, Y_RANGE=yRange
        endif else begin
        ;
        ; Gaussian
        ;
            x = findgen(256)
            gFit = gaussfit(x, temp, coeffs, $
                NTERMS=3)
            delta = coeffs[2]/2
            self.OpacityRange = [coeffs[1]-delta,coeffs[1]+delta]
            self.OpacityRamp = temp
        endelse

    endif else begin
        yRange = [0,255]
        self.OpacityRange = [0,255]
        self.OpacityRamp = temp
    endelse
    widget_control, widget_info(self.tlb, FIND_BY_UNAME='DualSlider'), SET_VALUE=yRange
    self -> UpdatePlot

end


;------------------------------------------------------------------------------
;+
; This method draws the current scene
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::RenderScene

    compile_opt StrictArr
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST
        return
    endif

    oWindow = self -> GetObjectByName('WINDOW')
    oWindow -> Draw

end


;------------------------------------------------------------------------------
;+
; This method prompts the user to select a file to which the current opacity
; table will be written.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::SaveOpacity

    compile_opt StrictArr
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if n_elements(lun) GT 0 then $
            free_lun, lun
        if self.verbose then $
            help, /LAST
        return
    endif

    file = dialog_pickfile(DEFAULT_EXTENSION='txt', $
        FILTER='*.txt', TITLE='Save Opacity Table')
    if file_test(file) then begin
        void = dialog_message(/QUESTION, $
            ['File already exists: ' + file, $
             'Would you like to overwrite this file?'])
        if strlowcase(void) EQ 'no' then $
            return
        file_delete, file
    endif

    openw, lun, file, /GET_LUN
    printf, lun, self.OpacityFunction
    printf, lun, transpose(self->GetOpacityTable())
    free_lun, lun

end


;------------------------------------------------------------------------------
;+
; This method updates the opacity function
;
; @Keyword
;   NO_DRAW {in}{optional}{type=boolean}
;     Set this keyword to prevent the display from updating.  By default the
;     scnene will be rendered.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::UpdateOpacityFunction, $
    NO_DRAW=noDraw

    self.OpacityRange = [0,255]
    self.OpacityFunction = self->GetFunction()
    wSlider = widget_info(self.tlb, FIND_BY_UNAME='DualSlider')
    case strupcase(self.OpacityFunction) of
        'FREE HAND': return
        'GAUSSIAN': begin
            self.OpacityRange = [113,143]
            self->CalculateGaussian
        end
        else: begin
        ;
        ; Linear
        ;
            x = bindgen(256)
            self.OpacityRamp = strupcase(self.OpacityFunction) EQ 'LINEAR (DECREASING)' ? $
                reverse(x) : x
            self->UpdateRange
        end
    endcase
    self->UpdatePlot

end


;------------------------------------------------------------------------------
;+
; This method updates the opacity table plot
;
; @Keyword
;   NO_DRAW {in}{optional}{type=boolean}
;     Set this keyword to prevent the display from updating.  By default the
;     scnene will be rendered.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::UpdatePlot, xy, $
    NO_DRAW=noDraw

    oPlot = self -> GetObjectByName('PLOT')
    range = self -> GetRange()
    if n_elements(xy) EQ 2 then begin
        xy[1] = (xy[1]>range[0])<range[1]
        functionName = strupcase(self->GetFunction())
        case functionName of
            'FREE HAND': begin
                xy[1] = xy[1]<255B
                xRange = [min([self.X,xy[0]], MAX=maxVal), maxVal]
                if xRange[0] NE xRange[1] then begin
                    m = (xy[1]-(self.OpacityRamp)[self.X])/(xy[0]-self.X)
                    b = xy[1] - m*xy[0]
                    for i = xRange[0], xRange[1] do $
                        self.OpacityRamp[i] = m*i+b
                endif else $
                    self.OpacityRamp[xy[0]] = xy[1]
                self.X = xy[0]
                self.OpacityRamp = (self.OpacityRamp>range[0])<range[1]
                OpacityTable = self.OpacityRamp
            end
            'GAUSSIAN': begin
                if self.HoldLeft EQ 2 then begin
                ; Moving a single point
                    case 1 of
                        xy[0] GT self.X: move = 1
                        xy[0] LT self.X: move = -1
                        else: return
                    endcase
                ; Move a single point
                    minVal = min(abs(xy[0]-self.OpacityRange), index)
                    self.OpacityRange[index] += move
                    self.X += move
                endif else begin
                ; Moving both points
                    self.OpacityRange += (xy[0]-self.X) ; Move the center
                    self.X = xy[0]
                endelse
    
            ; Don't let the min and max be the same value
                if self.OpacityRange[0] EQ self.OpacityRange[1] then begin
                    if self.OpacityRange[0] GT 128 then $
                        self.OpacityRange[0] -= 1 $
                    else $
                        self.OpacityRange[1] += 1
                endif
                self->CalculateGaussian
            end
            'LINEAR (DECREASING)': begin
                minVal = min(abs(xy[0]-self.OpacityRange), index)
                self.OpacityRange[index] = xy[0]
                self->CalculateLinearRamp, /DECREASING
            end
            'LINEAR (INCREASING)': begin
                minVal = min(abs(xy[0]-self.OpacityRange), index)
                self.OpacityRange[index] = xy[0]
                self->CalculateLinearRamp
            end
            else: return
        endcase
    endif

    oPlot -> SetProperty, DATAY=byte((self.OpacityRamp>range[0])<range[1])
    if ~keyword_set(noDraw) then $
        self -> RenderScene

end

;------------------------------------------------------------------------------
;+
; This method updates the x-range for the data
;
; @History
;   June, 2009 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog::UpdateRange
    range = self->GetRange()
    indexMin = where(self.OpacityRamp LE range[0]) > 0
    indexMax = where(self.OpacityRamp GE range[1], count)
    if count EQ 0 then $
        indexMax = 255
    self.OpacityRange = strupcase(self.OpacityFunction) EQ 'LINEAR (DECREASING)' ? $
        [max(indexMax),min(indexMin)] : $ 
        [max(indexMin),min(indexMax)]
end

;------------------------------------------------------------------------------
;+
; Define the member variables of the PALM_VolumeOpacityDialog class, along
; with various internal structures.
;
; @field
;   HoldLeft
;     A boolean value indicating whether the left mouse button is being
;     held down.
; @field
;   o3DDisplay
;     A reference to an instance of the PALM_3DDisplay object
; @field
;   OpacityRange
;     A 2-element vector holding the limits on the y-range for the opacity
;     table
; @field
;   OpacityRamp
;     A 256-element float array for the opacity-table ramp
; @field
;   tlb
;     The widget ID of the top level base
; @field
;   verbose
;      A boolean value that if set, will result in error messages being displayed
;      in the IDL ouput log.
; @field
;   X
;     An integer for storing an x-location
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Global Services Group
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_VolumeOpacityDialog__Define

    void = {PALM_VolumeOpacityDialog, $
            EM              : 0B, $
            HoldLeft        : 0B, $
            o3DDisplay      : obj_new(), $
            OpacityFunction : '', $
            OpacityRamp     : fltarr(256), $
            OpacityRange    : intarr(2), $
            tlb             : 0L, $
            verbose         : 0B, $
            X               : 0  $
           }

end
