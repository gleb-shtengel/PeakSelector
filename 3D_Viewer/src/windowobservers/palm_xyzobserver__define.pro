;------------------------------------------------------------------------------
;+
; This lifecycle method is called during the destruction of the
; PALM_XYZObserver  object via OBJ_DESTROY.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::Cleanup

end


;------------------------------------------------------------------------------
;+
; This method retrieves properties (usually member variable
; values) of this class.
;
; @Keyword
;   Cursor_Mode {out}{optional}
;       Set this keyword to a named variable to retrieve the
;       describing the current mouse drag event mode.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::GetProperty, $
    CURSOR_MODE = cursorMode

    compile_opt StrictArr
    on_error, 2

    if (arg_present(cursorMode)) then $
        cursorMode = self.Cursor_Mode

end


;------------------------------------------------------------------------------
;+
; This method is responsible for creating a new object when invoked via
; OBJ_NEW().
;
; @Param
;   oDisplay {in}{required}{type=objref}
;     Reference to the display object.
; @Keyword
;   WIDGET_DRAW {in}{optional}{type=widget ID}
;     ID of the draw widget associated with the draw window.
; @Keyword
;   _Ref_extra {in}{optional}
;     Any extra keywords are passed transparently to the
;     superclass init.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_XYZObserver::Init, oMainGUI,$
    _REF_EXTRA=_ref_extra

    compile_opt StrictArr
    on_error, 2

    if (n_params() ne 1) then begin
        message, 'One parameter is required.', /TRACEBACK
    endif
    if (~Obj_Valid(oMainGUI) || ~Obj_IsA(oMainGUI, 'PALM_MainGUI')) then begin
        message, 'oMainGUI object reference is invalid.', /TRACEBACK
    endif

    catch, errorNumber
    if (errorNumber ne 0) then begin
        catch, /CANCEL
        help, /LAST
        return, 0
    endif

    self.oMainGUI = oMainGUI
    self.oDisplay = self.oMainGUI -> GetObjectByName('XYZDisplay')

    return, 1

end


;------------------------------------------------------------------------------
;+
; This method handles mouse down (press) events
;
; @Param
;   oWindow {in}{required}{type=objref}
;     Reference to the window object.
; @Param
;   x {in}{required}{type=long}
;     X mouse position in device coordinates.
; @Param
;   y {in}{required}{type=long}
;     Y mouse position in device coordinates.
; @Param
;   buttonmask {in}{required}{type=byte}
;     Indicates which mouse button was pressed.
; @Param
;   modifiers {in}{required}{type=byte}
;     Modifier keys, if any.
; @Param
;   numclicks {in}{required}{type=byte}
;     number of mouse clicks.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::OnMouseDown, oWindow, X, Y, ButtonMask, Modifiers, NumClicks
    compile_opt StrictArr
    on_error, 2

    case ButtonMask of
        1 : begin
            self->OnLeftButtonDown, oWindow, X, Y, Modifiers, NumClicks
            end
        2 : begin
            self->OnMiddleButtonDown, oWindow, X, Y, Modifiers, NumClicks
            end
        4 : begin
            self->OnRightButtonDown, oWindow, X, Y, Modifiers, NumClicks
            end
        else :
    endcase

end


;------------------------------------------------------------------------------
;+
; This method is responsible for decomposing
; a button release event in the draw widget
; into a left, middle, or right event type and dispatching
; the event to a more granular event handling method.
;
; @Param
;   oWindow {in}{required}{type=objref}
;     Reference to the window object.
; @Param
;   x {in}{required}{type=long}
;     X mouse position in device coordinates.
; @Param
;   y {in}{required}{type=long}
;     Y mouse position in device coordinates.
; @Param
;   buttonmask {in}{required}{type=byte}
;     Indicates which mouse button was pressed.
; @Param
;   modifiers {in}{required}{type=byte}
;     Modifier keys, if any.
; @Param
;   numclicks {in}{required}{type=byte}
;     number of mouse clicks.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::OnMouseUp, oWindow, X, Y, ButtonMask, Modifiers, NumClicks
    compile_opt StrictArr
    on_error, 2

    case ButtonMask of
        1 : begin
            self->OnLeftButtonUp, oWindow, X, Y
            end
        2 : begin
            self->OnMiddleButtonUp, oWindow, X, Y
            end
        4 : begin
            self->OnRightButtonUp, oWindow, X, Y
            end
        else :
    endcase
end


;------------------------------------------------------------------------------
;+
; This method is responsible for handling keyboard
; events in the draw widget.
;
; @Param
;   oWindow {in}{required}{type=objref}
;     Reference to the window object.
; @Param
;   x {in}{required}{type=long}
;     X mouse position in device coordinates.
; @Param
;   y {in}{required}{type=long}
;     Y mouse position in device coordinates.
; @Param
;   modifiers {in}{required}{type=long}
;     Modifier keys, if any.
; @Param
;   press {in}{required}{type=long}
;     key press.
; @Param
;   release {in}{required}{type=long}
;     key release.
; @Param
;   keysymbol {in}{required}{type=long}
;     Integer that indicates special key
;     (e.g., 1=shift).
; @Param
;   character {in}{required}{type=byte}
;     byte value of ASCII character for key.
; @Param
;   isASCII {in}{required}{type=byte}
;     True if ASCII key was pressed.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::OnKeyboard, oWindow, IsASCII, Character, KeySymbol, X, Y, Press, Release, Modifiers
    compile_opt StrictArr
    on_error, 2

    case KeySymbol of
    1:  begin
        ;
        ; Shift
        ;
    end
    2: begin
        ;
        ; Control
        ;
    end
    4: begin  ;Ctrl-alt

    end
    else: begin
        case Character of
        27 : begin ;escape

        end
        else :
        endcase
    end
    endcase
end


;------------------------------------------------------------------------------
;+
; This method is responsible for handling
; a left button press event in the draw widget.
;
; @Param
;   x {in}{required}{type=long}
;     X mouse position in device coordinates.
; @Param
;   y {in}{required}{type=long}
;     Y mouse position in device coordinates.
; @Param
;   modifiers {in}{required}{type=byte}
;     Modifier keys, if any.
; @Param
;   numclicks {in}{required}{type=byte}
;     number of mouse clicks.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::OnLeftButtonDown, oWindow, X, Y, Modifiers, NumClicks
    compile_opt StrictArr
    on_error, 2

    if numClicks EQ 2 then begin
        self.oDisplay -> AxisHide
        return
    endif

    self.oDisplay -> SetSelect, oWindow, [X,Y]
    self.XY = [x,y]
    case self.Cursor_Mode of
        'Pan':self.oDisplay -> Pan, oWindow, [X, Y], /INITIAL_POSITION
        'Zoom': self.oDisplay -> Zoom, oWIndow, [X, Y], /INITIAL_POSITION
        else:
    endcase
    self.HoldLeft=1B

end


;------------------------------------------------------------------------------
;+
; This method is responsible for handling
; a left button release event in the draw widget.
;
; @Param
;   x {in}{required}{type=long}
;     X mouse position in device coordinates.
; @Param
;   y {in}{required}{type=long}
;     Y mouse position in device coordinates.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::OnLeftButtonUp, oWindow, X, Y

    compile_opt StrictArr
    on_error, 2

    case self.Cursor_Mode of
        'Pan': begin
            self.oDisplay->SetViews, /ADJUST_SLIDERS, /NO_LOCATION_ADJUST
        end
        'SliceScroll': begin
            self.oDisplay->UpdateVolumeForManip, $
                NO_DRAW=array_equal(self.XY, [x,y]), $
                /RELEASE
        end
        'Zoom': begin
            self.oDisplay->SetViews, /ADJUST_SLIDERS, /NO_LOCATION_ADJUST
        end
        else: print, self.Cursor_Mode
    endcase

    self.HoldLeft=0B

end


;------------------------------------------------------------------------------
;+
; This method is responsible for handling
; a middle button press event in the draw widget.
;
; @Param
;   x {in}{required}{type=long}
;     X mouse position in device coordinates.
; @Param
;   y {in}{required}{type=long}
;     Y mouse position in device coordinates.
; @Param
;   modifiers {in}{required}{type=byte}
;     Modifier keys, if any.
; @Param
;   numclicks {in}{required}{type=byte}
;     number of mouse clicks.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::OnMiddleButtonDown, oWindow, X, Y, Modifiers, NumClicks
    compile_opt StrictArr
    on_error, 2

    self.HoldMiddle = 1b

end


;------------------------------------------------------------------------------
;+
; This method is responsible for handling
; a middle button release event in the draw widget.
;
; @Param
;   x {in}{required}{type=long}
;     X mouse position in device coordinates.
; @Param
;   y {in}{required}{type=long}
;     Y mouse position in device coordinates.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::OnMiddleButtonUp, oWindow, X, Y
    compile_opt StrictArr
    on_error, 2

    self.HoldMiddle=0B

end


;------------------------------------------------------------------------------
;+
; This method is responsible for handling
; a mouse motion event in the draw widget.
;
; @Param
;   oWindow {in}{required}{type=objref}
;     Reference to the window object.
; @Param
;   x {in}{required}{type=long}
;     X mouse position in device coordinates.
; @Param
;   y {in}{required}{type=long}
;     Y mouse position in device coordinates.
; @Param
;   buttonmask {in}{required}{type=byte}
;     Indicates which mouse button was pressed.
; @Param
;   modifiers {in}{required}{type=byte}
;     Modifier keys, if any.
; @Param
;   numclicks {in}{required}{type=byte}
;     number of mouse clicks.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::OnMouseMotion, oWindow, X, Y, ButtonMask, Modifiers, NumClicks

    compile_opt StrictArr
    on_error, 2

    if ~self.HoldLeft then $
        return

    case self.Cursor_Mode of
        'WindowLevel': self.oDisplay -> WindowLevel, [X,Y]
        'SliceScroll': begin
            if ~array_equal(self.xy, [x,y]) then $
                self.oDisplay -> Move, oWindow, [X,Y]
        end
        'Pan':self.oDisplay -> Pan, oWindow, [X, Y]
        'Zoom': self.oDisplay -> Zoom, oWindow, [X, Y]
        else :
    endcase

end


;------------------------------------------------------------------------------
;+
; This method is responsible for handling
; a right button press event in the draw widget.
;
; @Param
;   x {in}{required}{type=long}
;     X mouse position in device coordinates.
; @Param
;   y {in}{required}{type=long}
;     Y mouse position in device coordinates.
; @Param
;   modifiers {in}{required}{type=byte}
;     Modifier keys, if any.
; @Param
;   numclicks {in}{required}{type=byte}
;     number of mouse clicks.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::OnRightButtonDown, oWindow, X, Y, Modifiers, NumClicks

    compile_opt StrictArr
    on_error, 2

; get the draw widget's ID
    oWindow -> GetProperty, NAME=wName
    wDraw = self.oMainGUI -> Get('PALM_'+wName+'Window')
    if ~widget_info(wDraw, /VALID) then $
        return

    self.HoldRight = 1B
    wContextMenu = widget_info(wDraw, /CHILD)
    widget_displaycontextmenu, wDraw, X, Y, wContextMenu

end


;------------------------------------------------------------------------------
;+
; This method is responsible for further decomposing
; a right button release event in the draw widget.
;
; @Param
;   x {in}{required}{type=long}
;     X mouse position in device coordinates.
; @Param
;   y {in}{required}{type=long}
;     Y mouse position in device coordinates.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::OnRightButtonUp, oWindow, X, Y
    compile_opt StrictArr
    on_error, 2

    self.HoldRight=0B

end


;------------------------------------------------------------------------------
;+
; This method handles mouse wheel events
;
; @param
;   oWindow {in}{type=objref}{required}
;     Reference to the window object.
; @param
;   x {in}{type=long}{required}
;     X mouse position in device coordinates.
; @param
;   y {in}{type=long}{required}
;     Y mouse position in device coordinates.
; @param
;   delta {in}{type=long}{required}
;     A long integer giving the direction and distance that the
;     wheel was rolled.
; @param
;   modifiers {in}{type=byte}{required}
;     Modifier keys, if any.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::OnWheel, oWindow, X, Y, Delta, Modifiers
    compile_opt StrictArr
    on_error, 2

end


;------------------------------------------------------------------------------
;+
; This procedure sets properties of the class
;
; @Keyword
;   CURSOR_MODE {in}{optional}{type=string}
;       Set this keyword to the string describing the cursor mode
;       associated with the left mouse button.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver::SetProperty, $
    CURSOR_MODE = cursorMode

    compile_opt StrictArr
    on_error, 2

    if (n_elements(cursorMode) NE 0) then $
        self.Cursor_Mode = cursorMode

end


;------------------------------------------------------------------------------
;+
; Define the member variables of the PALM_XYZObserver class, along with various
; internal structures.
;
; @File_comments
;  This file defines the PALM_SelectMode class for the framework of the
;  application.
;
; @field
;   CursorMode
;     A string that specifies the current cursor mode
; @field
;   holdLeft
;     A boolean field that if set indicates the left mouse button is being
;     pressed
; @field
;   holdMiddle
;     A boolean field that if set indicates the middle mouse button is being
;     pressed
; @field
;   holdRight
;     A boolean field that if set indicates the right mouse button is being
;     pressed
; @field
;   oDisplay
;     An object reference to the PALM_3DDisplay object
; @field
;   wDraw
;     A draw widget ID for the context menu
; @field
;   XY
;     A 2-element vector for storing the cursor position during mouse events
; @field
;   verbose
;     A boolean field that when set will have error messages printed to
;     the IDL output log.
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Professional Services Group
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZObserver__define
   void = {PALM_XYZObserver,          $
            Cursor_Mode : '',         $
            HoldLeft    : 0B,         $
            HoldMiddle  : 0b,         $
            HoldRight   : 0B,         $
            oDisplay    : obj_new(),  $
            oMainGUI    : obj_new(),  $
            XY          : intarr(2)   $
          }
end
