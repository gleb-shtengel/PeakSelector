;------------------------------------------------------------------------------
;+
; This method handles events from button widgets
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from a button on the PALM_ColorTableDialog widget.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_ColorTableDialog::ButtonEvent, event

    widget_control, event.id, GET_UVALUE=uval
    case uval of
        'Apply': begin
            self->UpdatePalette
            self -> GetPalette, BLUE=blue, $
                GREEN=green, $
                RED=red
            if self.InvertTable then begin
                blue = reverse(blue)
                green = reverse(green)
                red = reverse(red)
            endif
            self.o3DDisplay -> SetProperty, COLOR_TABLE=[[red],[green],[blue]]
            self.o3DDisplay -> RenderScene
            self.oXYZDisplay -> SetProperty, COLOR_TABLE=[[red],[green],[blue]]
            self.oXYZDisplay -> RenderScene
        end
        'Close': self -> Destruct
        else:
    endcase

end


;------------------------------------------------------------------------------
;+
; This lifecycle method is called during the destruction of the
; PALM_ColorTableDialog object via OBJ_DESTROY.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_ColorTableDialog::Cleanup

    compile_opt idl2
    on_error, 2

    self->Destruct

end


;------------------------------------------------------------------------------
;+
; This method constructs the widgets for the color table selection dialog
;
; @Param
;   GroupLeader {in}{optional}{type=long}
;     The widget ID of the group leader of the color table selction dialog
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_ColorTableDialog::ConstructWidgets, GroupLeader

    compile_opt idl2
    on_error, 2

;  Roughly center on the screen
    xy = (get_screen_size() - [300, 400]) / 2

;  Build this dialog's top level base
    self.tlb = widget_base(/COLUMN, $
        EVENT_PRO='GenericClassEvent', $
        GROUP_LEADER=GroupLeader, $
        KILL_NOTIFY='GenericClassKillNotify', $
        /MODAL, $
        NOTIFY_REALIZE='GenericClassNotifyRealize', $
        UVALUE=self, $
        TITLE='Color Table', $
        XOFFSET=xy[0], $
        YOFFSET=xy[1])

;  Draw region to show the color tables
    wLabel = widget_label(self.tlb, $
        VALUE=' ', $
        YSIZE=10)
    wDraw = widget_draw(self.tlb, $
        /ALIGN_CENTER, $
        GRAPHICS_LEVEL=2, $
        UNAME='Draw', $
        XSIZE=256, $
        YSIZE=50)
    wLabel = widget_label(self.tlb, $
        VALUE=' ', $
        YSIZE=10)
    sliderBase = widget_base(self.tlb, $
        COL=1, $
        XSIZE=256)
    sliderID = lonarr(3)
    sliderID[0] = widget_slider(sliderBase, $
        /DRAG, $
        MAXIMUM=100, $
        MINIMUM=0, $
        UNAME='onStretch', $
        UVALUE='Stretch', $
        VALUE = 0)
    wLabel = widget_label(sliderBase, $
        /ALIGN_LEFT, $
        VALUE='Stretch Bottom')
    sliderID[1] = widget_slider(sliderBase, $
        /DRAG, $
        MAXIMUM=100, $
        MINIMUM=0, $
        UNAME='onStretch', $
        UVALUE='Stretch', $
        VALUE=100)
    wLabel = widget_label(sliderBase, $
        /ALIGN_LEFT, $
        VALUE='Stretch Top')
    wLabel = widget_label(sliderBase, $
        /DYNAMIC, $
        UNAME='SliderLabel', $
        VALUE = string(1.0, FORMAT='(f6.3)'))
    sliderID[2] = widget_slider(sliderBase, $
        /DRAG, $
        MINIMUM=0, $
        MAXIMUM=100, $
        /SUPPRESS_VALUE, $
        UNAME='onStretch', $
        UVALUE='Stretch', $
;        UVALUE=wLabel, $
        VALUE=50)
    wLabel = widget_label(sliderBase, $
        /ALIGN_LEFT, $
        VALUE='Gamma Correction')
    widget_control, sliderBase, SET_UNAME = 'STRETCH_BASE', $
        SET_UVALUE = sliderID

;  Scrolling list of available color tables
    loadct, GET_NAMES=names
    wList = widget_list(self.tlb, $
        UNAME='onColorTableSelect', $
        UVALUE='ColorTableSelect', $
        VALUE=names, $
        XSIZE=40, $
        YSIZE=10)
    widget_control, wList, SET_LIST_SELECT=ctIndex
    wList = widget_label(self.tlb, $
        VALUE=' ', $
        YSIZE=10)
;  OK & Close buttons
    wBase = widget_base(self.tlb, /ROW, /ALIGN_CENTER)
    void = widget_button(wBase, $
        UNAME='onApply', $
        UVALUE='Apply', $
        VALUE='Apply', $
        XSIZE=80, $
        YSIZE=30)
    void = widget_button(wBase, $
        UNAME='onClose', $
        UVALUE='Close', $
        VALUE='Close', $
        XSIZE=80, $
        YSIZE=30)

  ;  Realize the widgets
    widget_control, self.tlb, /REALIZE

    return, 1

end


;------------------------------------------------------------------------------
;+
; This method has the primary responsibility for cleaning up the
; PALM_ColorTableDialog object at the end of its lifecycle.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_ColorTableDialog::Destruct

    compile_opt idl2
    on_error, 2

    if widget_info(self.tlb, /VALID) then begin
        wDraw = self -> Get('Draw')
        if widget_info(wDraw, /VALID) then begin
            widget_control, self->Get('Draw'), GET_VALUE=oWindow
            self -> GetPalette, PALETTE=oPalette
            obj_destroy, [oPalette, oWindow]
        endif
    endif

    if widget_info(self.tlb, /VALID) then begin
        widget_control, self.tlb, KILL_NOTIFY = ''
        widget_control, self.tlb, /DESTROY
    endif

    if obj_valid(self.oPalette) then $
        obj_destroy, self.oPalette

end


;------------------------------------------------------------------------------
;+
; Main event handler for the color table selection dialog
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from the PALM_ColorTableDialog.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_ColorTableDialog::Event, event

    compile_opt idl2
    on_error, 2

    case strupcase(tag_names(event, /STRUCT)) of
        'WIDGET_BUTTON': self -> ButtonEvent, event
        'WIDGET_LIST': self -> ListEvent, event
        'WIDGET_SLIDER': self -> SliderEvent, event
        else:
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
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_ColorTableDialog::Get, name

    compile_opt idl2
    on_error, 2

    return, widget_info(self.tlb, FIND_BY_UNAME=name)

end


;------------------------------------------------------------------------------
;+
; This method retrieves the red, green and blue vectors for the current
; color table
;
; @Keyword
;   BLUE {out}{optional}
;     Set this keyword to a named variable to retrieve the blue vector
;     of the current color table
; @Keyword
;   GREEN {out}{optional}
;     Set this keyword to a named variable to retrieve the green vector
;     of the current color table
; @Keyword
;   PALETTE {out}{optional}
;     Set this keyword to a named variable to retrieve a reference to
;     current IDLgrPalette object
; @Keyword
;   RED {out}{optional}
;     Set this keyword to a named variable to retrieve the red vector
;     of the current color table
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_ColorTableDialog::GetPalette, $
    BLUE=blue, $
    GREEN=green, $
    PALETTE=oPalette, $
    RED=red

    widget_control, self->Get('Draw'), GET_VALUE=oWindow
    oWindow -> GetProperty, GRAPHICS_TREE=oView
    oImage = oView -> GetByName('Model/Image')
    oImage -> GetProperty, PALETTE=oPalette
    oPalette -> GetProperty, BLUE=blue, $
        GREEN=green, $
        RED=red

end


;------------------------------------------------------------------------------
;+
; This method initializes the color table selection class
;
; @Returns
;   1 for success and 0 otherwise
;
; @Param
;   GroupLeader {in}{required}{type=long}
; @Param
;   o3DDisplay {in}{required}{type=objref}
; @Param
;   oXYZDisplay {in}{required}{type=objref}
;
; @keyword
;   ERROR_MESSAGE {out}{type=string}{optional}
;     Set this keyword to a named variable to retrieve any error
;     messages thrown.
; @Keyword
;   INVERT_TABLE {in}{optional}{type=boolean}
;     Set this keyword to indicate that the current color table
;     is inverted
; @keyword
;    VERBOSE {in}{type=boolean}{optional}
;      Setting this keyword will result in the object displaying any
;      error messages to the IDL output log.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_ColorTableDialog::Init, GroupLeader, o3DDisplay, oXYZDisplay, $
    ERROR_MESSAGE=ErrMsg, $
    INVERT_TABLE=doInvert, $
    VERBOSE=verbose

    compile_opt idl2
    on_error, 2

    if (n_params() ne 3) then begin
        message, 'Three parameters are required.', /TRACEBACK
    endif

    if ~obj_valid(o3DDisplay) then $
        return, 0
    if ~obj_isa(o3DDisplay, 'PALM_3DDisplay') then $
        return, 0
    if ~obj_valid(oXYZDisplay) then $
        return, 0
    if ~obj_isa(oXYZDisplay, 'PALM_XYZDisplay') then $
        return, 0
    self.verbose = keyword_set(verbose)

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST $
        else $
            help, /LAST, OUT=ErrMsg
        return, 0
    endif

    self.cRamp = bindgen(256)
    self.gamma = 1.0
    self.InvertTable = keyword_set(doInvert)
    self.o3DDisplay = o3DDisplay
    self.oXYZDisplay = oXYZDisplay
   
    self.oXYZDisplay->GetProperty, COLOR_TABLE=ColorTable
    self.oPalette = obj_new('IDLgrPalette')
    if self.InvertTable then $
        ColorTable=reverse(Colortable,1)
    self.oPalette -> SetProperty, BLUE=ColorTable[*,2], $
        GREEN=ColorTable[*,1], $
        RED=ColorTable[*,0]
    if ~(self->ConstructWidgets(GroupLeader)) then $
        return, 0

    return, 1

end



;------------------------------------------------------------------------------
;+
; This method is responsible for initializing the standard graphics tree
; to be displayed in the draw widget.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_ColorTableDialog::InitializeDisplay

    widget_control, widget_info(self.tlb, FIND_BY_UNAME='Draw'), $
        GET_VALUE=oWindow
    self.oPalette -> GetProperty, BLUE=blue, $
        GREEN=green, $
        RED=red
    oPalette = obj_new('IDLgrPalette')
    oPalette -> SetProperty, BLUE=blue, $
        GREEN=green, $
        RED=red
    oWindow -> GetProperty, DIMENSIONS=winDims
    oImage = obj_new('IDLgrImage', bindgen(256,winDims[1]), $
        NAME='Image', $
        PALETTE=oPalette)
    oModel = obj_new('IDLgrModel', NAME='Model')
    oView = obj_new('IDLgrView', $
        NAME='View', $
        VIEWPLANE_RECT=[0,0,256,winDims[1]])
    oModel -> Add, oImage
    oView -> Add, oModel
    oWindow -> SetProperty, GRAPHICS_TREE=oView
    oWindow -> Draw

    return, 1

end


;------------------------------------------------------------------------------
;+
; This method handles events from list widgets
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from a list on the PALM_ColorTableDialog widget.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_ColorTableDialog::ListEvent, event

    widget_control, event.id, GET_UVALUE=uval
    case uval of
        'ColorTableSelect': begin
            self.oPalette -> LoadCT, event.index
            self -> UpdatePalette
            self -> UpdateDisplay
        end
        else:
    endcase

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
Pro PALM_ColorTableDialog::Notify_Realize, tlb

    compile_opt StrictArr
    on_error, 2

    void = self -> InitializeDisplay()
    widget_control, tlb, MAP = 1
    xmanager, 'GenericClass', tlb, $
        EVENT_HANDLER = 'GenericClassEvent'

End


;------------------------------------------------------------------------------
;+
; This method handles events from button widgets
;
; @param
;   event {in}{type=IDL event structure}{required}
;     An IDL widget event from a slider on the PALM_ColorTableDialog widget.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_ColorTableDialog::SliderEvent, event

    compile_opt idl2
    on_error, 2

    widget_control, event.id, GET_UVALUE=uval
    case uval of
        'Stretch': begin
            widget_control, self->Get('STRETCH_BASE'), GET_UVALUE = sliderID
            widget_control, sliderID[0], GET_VALUE = vBottom
            widget_control, sliderID[1], GET_VALUE = vTop

            if event.id eq sliderID[2] then begin
                widget_control, sliderID[2], GET_VALUE = vGamma
                self.gamma = 10^((vGamma/50.) - 1)
                sliderLabel = widget_info(self.tlb, FIND_BY_UNAME='SliderLabel')
                widget_control, sliderLabel, SET_VALUE = string(self.gamma, FORMAT='(f6.3)')
            endif

            k = 255./100
            vBottom = vBottom*k
            vTop = vTop*k
            k = vBottom eq vTop ? 1.0 : 255./(vTop-vBottom)
            self.cRamp = (self.gamma eq 1.0 ? round(findgen(256) * k + (-k*vBottom) > 0.0) : $
                ((findgen(256) * k/256. + (-k*vBottom/256.) > 0.0)^self.gamma) * 256.) < 255L

            self -> UpdatePalette
            self -> UpdateDisplay
        end
        else:
    endcase

end


;------------------------------------------------------------------------------
;+
; This method updates the palette used in the main application
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_ColorTableDialog::UpdatePalette

    compile_opt idl2
    on_error, 2

    selected=widget_info(self->Get('onColorTableSelect'), /LIST_SELECT)
    if selected ge 0 then begin
        self.oPalette -> LoadCT, selected
        self.oPalette -> GetProperty, BLUE=blue, $
            GREEN=green, $
            RED=red
    endif else $
        self.oPalette -> GetProperty, BLUE=blue, $
            GREEN=green, $
            RED=red
    self -> GetPalette, PALETTE=oPalette
    oPalette->SetProperty, BLUE=blue[self.cRamp], $
        GREEN=green[self.cRamp], $
        RED=red[self.cRamp]

end

;------------------------------------------------------------------------------
;+
; This method updates the colorbar display
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_ColorTableDialog::UpdateDisplay

    widget_control, self->Get('Draw'), GET_VALUE=oWindow
    oWindow -> Draw

end


;------------------------------------------------------------------------------
;+
; Define the member variables of the PALM_ColorTableDialog class, along with
; various internal structures.
;
; @field
;   cRamp
;     A 256-element byte array defining the color table ramp
; @field
;   gamma
;     A floating point scalar specifying the gamma value
; @field
;   invertTable
;     A boolean field indicating whether the main application's color table
;     is inverted
; @field
;   o3DDisplay
;     A reference to an instance of the PALM_3DDisplay object
; @field
;   oXYZDisplay
;     A reference to an instance of the PALM_XYZDisplay object
; @field
;   oPalette
;     A reference to an IDLgrPalette
; @field
;   tlb
;     The widget ID for the dialog's top level base
; @field
;   verbose
;      A boolean value that if set, will result in error messages being displayed
;      in the IDL ouput log
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Global Services Group
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_ColorTableDialog__Define

    compile_opt idl2

    class = {PALM_ColorTableDialog,       $
             cRamp       : fltarr(256),   $
             gamma       : 0.0,           $
             invertTable : 0B, $
             o3DDisplay  : obj_new(),     $
             oXYZDisplay : obj_new(),     $
             oPalette    : obj_new(),     $
             tlb         : 0L,            $
             verbose     : 0B $
            }
end
