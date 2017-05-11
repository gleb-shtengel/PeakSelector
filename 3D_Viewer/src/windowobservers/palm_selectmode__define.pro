;------------------------------------------------------------------------------
;+
; This lifecycle method is called during the destruction of the PALM_SelectMode
; object via OBJ_DESTROY.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_SelectMode::Cleanup

end


;------------------------------------------------------------------------------
;+
; This method initializes the object
;
; @param
;   oDisplay {in}{type=object reference}{required}
;     A reference to an instance of the PALM_3DDisplay object
;
; @keyword
;    VERBOSE {in}{type=boolean}{optional}
;      Set this keyword to have the object display error messages
;      in the IDL output log.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_SelectMode::Init, oDisplay, $
    VERBOSE=verbose

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST
        return, 0
    endif

    self.verbose = keyword_set(verbose)

    if ~obj_valid(oDisplay) then begin
        errMsg = 'input argument oDisplay is not a valid object'
        catch, /CANCEL
        return, 0
    endif
    if ~obj_isa(oDisplay, 'PALM_3DDisplay') then begin
        errMsg = 'agrument oDisplay is not of the type PALM_3DDisplay'
        catch, /CANCEL
        return, 0
    endif

    self.oDisplay = oDisplay
    return, 1

end


;------------------------------------------------------------------------------
;+
; This method is for handling keyboard events
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
;   modifiers {in}{type=long}{required}
;     Modifier keys, if any.
; @param
;   press {in}{type=long}{required}
;     key press.
; @param
;   release {in}{type=long}{required}
;     key release.
; @param
;   keysymbol {in}{type=long}{required}
;     Integer that indicates special key
;     (e.g., 1=shift).
; @param
;   character {in}{type=byte}{required}
;     byte value of ASCII character for key.
; @param
;   isASCII {in}{type=byte}{required}
;     True if ASCII key was pressed.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_SelectMode::OnKeyboard, oWindow, IsASCII, Character, KeySymbol, x, y, Press, Release, Modifiers

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

end


;------------------------------------------------------------------------------
;+
; this method handles press events from the left mouse button
;
; @param
;   x {in}{type=long}{required}
;     X mouse position in device coordinates.
; @param
;   y {in}{type=long}{required}
;     Y mouse position in device coordinates.
; @param
;   modifiers {in}{type=byte}{required}
;     Modifier keys, if any.
; @param
;   numclicks {in}{type=byte}{required}
;     number of mouse clicks.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_SelectMode::OnLeftButtonDown, x, y, Modifiers, NumClicks

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

    self.holdLeft = 1B

end


;------------------------------------------------------------------------------
;+
; This method handles release events from the left mouse button
;
; @param
;   x {in}{type=long}{required}
;     X mouse position in device coordinates.
; @param
;   y {in}{type=long}{required}
;     Y mouse position in device coordinates.
; @param
;   modifiers {in}{type=byte}{required}
;     Modifier keys, if any.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_SelectMode::OnLeftButtonUp, x, y, Modifiers

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

    self.holdLeft = 0B

end


;------------------------------------------------------------------------------
;+
; this method handles press events from the middle mouse button
;
; @param
;   x {in}{type=long}{required}
;     X mouse position in device coordinates.
; @param
;   y {in}{type=long}{required}
;     Y mouse position in device coordinates.
; @param
;   modifiers {in}{type=byte}{required}
;     Modifier keys, if any.
; @param
;   numclicks {in}{type=byte}{required}
;     number of mouse clicks.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_SelectMode::OnMiddleButtonDown, x, y, Modifiers, NumClicks

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

    self.holdMiddle = 1B

end


;------------------------------------------------------------------------------
;+
; This method handles release events from the middle mouse button
;
; @param
;   x {in}{type=long}{required}
;     X mouse position in device coordinates.
; @param
;   y {in}{type=long}{required}
;     Y mouse position in device coordinates.
; @param
;   modifiers {in}{type=byte}{required}
;     Modifier keys, if any.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_SelectMode::OnMiddleButtonUp, x, y, Modifiers

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

    self.holdMiddle = 0B

end


;------------------------------------------------------------------------------
;+
; This method handles mouse down events
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
;   buttonmask {in}{type=byte}{required}
;     Indicates which mouse button was pressed.
; @param
;   modifiers {in}{type=byte}{required}
;     Modifier keys, if any.
; @param
;   numclicks {in}{type=byte}{required}
;     number of mouse clicks.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_SelectMode::OnMouseDown, oWindow, x, y, ButtonMask, Modifiers, NumClicks

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

    case ButtonMask of
        1: self -> OnLeftButtonDown, x, y, Modifiers, NumClicks
        2: self -> OnMiddleButtonDown, x, y, Modifiers, NumClicks
        4: self -> OnRightButtonDown, x, y, Modifiers, NumClicks
        else:
    endcase

end


;------------------------------------------------------------------------------
;+
; This method handles mouse motion events
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
;   buttonmask {in}{type=byte}{required}
;     Indicates which mouse button was pressed.
; @param
;   modifiers {in}{type=byte}{required}
;     Modifier keys, if any.
; @param
;   numclicks {in}{type=byte}{required}
;     Number of mouse clicks.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_SelectMode::OnMouseMotion, oWindow, x, y, ButtonMask, Modifiers, NumClicks

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

    if self.holdLeft then begin
        return
    endif

    if self.holdMiddle then begin
        return
    endif

    if self.holdRight then begin
        return
    endif

; No buttons pressed
    self.oDisplay -> GetProperty, $
        MODEL=oModel, $
        WINDOW=oWindow
    if oWindow -> PickData(self.oDisplay, oModel, [x,y], xyz) then begin
        self.oDisplay -> GetProperty, $
            REFERENCE_RANGE=refRange, $
            XRANGE=xRange, $
            YRANGE=yRange, $
            ZRANGE=zRange
    endif

end


;------------------------------------------------------------------------------
;+
; Thie method handles mouse up events
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
;   buttonmask {in}{type=byte}{required}
;     Indicates which mouse button was pressed.
; @param
;   modifiers {in}{type=byte}{required}
;     Modifier keys, if any.
; @param
;   numclicks {in}{type=byte}{required}
;     Number of mouse clicks.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_SelectMode::OnMouseUp, oWindow, x, y, ButtonMask, Modifiers, NumClicks

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

    case ButtonMask of
        1: self -> OnLeftButtonUp, x, y, Modifiers
        2: self -> OnMiddleButtonUp, x, y, Modifiers
        4: self -> OnRightButtonUp, x, y, Modifiers
        else:
    endcase

end


;------------------------------------------------------------------------------
;+
; this method handles press events from the right mouse button
;
; @param
;   x {in}{type=long}{required}
;     X mouse position in device coordinates.
; @param
;   y {in}{type=long}{required}
;     Y mouse position in device coordinates.
; @param
;   modifiers {in}{type=byte}{required}
;     Modifier keys, if any.
; @param
;   numclicks {in}{type=byte}{required}
;     number of mouse clicks.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_SelectMode::OnRightButtonDown, x, y, Modifiers, NumClicks

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

    self.holdRight = 1B

end


;------------------------------------------------------------------------------
;+
; This method handles release events from the right mouse button
;
; @param
;   x {in}{type=long}{required}
;     X mouse position in device coordinates.
; @param
;   y {in}{type=long}{required}
;     Y mouse position in device coordinates.
; @param
;   modifiers {in}{type=byte}{required}
;     Modifier keys, if any.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_SelectMode::OnRightButtonUp, x, y, Modifiers

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

    self.holdRight = 0B

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
pro PALM_SelectMode::OnWheel, oWindow, x, y, Delta, Modifiers

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

end


;------------------------------------------------------------------------------
;+
; Define the member variables of the PALM_SelectMode class, along with various
; internal structures.
;
; @File_comments
;  This file defines the PALM_SelectMode class for the framework of the
;  application.
;
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
pro PALM_SelectMode__Define

    void = {PALM_SelectMode, $
            holdLeft   : 0B, $
            holdMiddle : 0B, $
            holdRight  : 0B, $
            oDisplay   : obj_new(), $ ; Do not destroy
            verbose    : 0B $
           }

end
