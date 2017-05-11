;------------------------------------------------------------------------------
;+
; This lifecycle method is called during the destruction of the PALM_3DObserver
; object via OBJ_DESTROY.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DObserver::Cleanup

    if obj_valid(self.oMode) then $
        obj_destroy, self.oMode

    ptr_free, self.pModes

end


;------------------------------------------------------------------------------
;+
; This function returns the current manipulation mode.
;
; @returns
;   A string with the current manipulation mode
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_3DObserver::GetWindowMode

    compile_opt idl2
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if self.verbose then $
            help, /LAST
        return, ''
    endif

    return, self.mode
end


;------------------------------------------------------------------------------
;+
; This method initializes the object.
;
; @returns
;    1 for success and 0 for failure
;
; @param
;   oMainGUI {in}{type=object reference}{required}
;     A reference to an instance of the PALM_MainGUI object
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
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_3DObserver::Init, oMainGUI, $
    ERROR_MESSAGE=errMsg, $
    VERBOSE=verbose

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
        return,  0
    endif

    self.verbose = keyword_set(verbose)

    if ~obj_valid(oMainGUI) then begin
        errMsg = 'input argument oDisplay is not a valid object'
        if self.verbose then $
            print, errMsg
        return, 0
    endif
    if ~obj_isa(oMainGUI, 'PALM_MainGUI') then begin
        errMsg = 'agrument oMainGUI is not of the type PALM_MainGUI'
        if self.verbose then $
            print, errMsg
        return, 0
    endif
    self.oMainGUI = oMainGUI
    
    oDisplay = self.oMainGUI -> GetObjectByName('3DDisplay')
    if ~obj_valid(oDisplay) then begin
        errMsg = 'input argument oDisplay is not a valid object'
        if self.verbose then $
            print, errMsg
        return, 0
    endif
    self.oDisplay = oDisplay

    modes = ['Select', $
             'Pan', $
             'Rotate', $
             'Zoom']
    self.pModes = ptr_new(modes, /NO_COPY)
    self -> SetWindowMode, (*self.pModes)[0]

    return, 1

end


;------------------------------------------------------------------------------
;+
;  Pass a keyboard event to the proper handler for the current mode.
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
pro PALM_3DObserver::OnKeyboard, oWindow, IsASCII, Character, KeySymbol, X, Y, Press, Release, Modifiers

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

    self.oMode -> OnKeyboard, oWindow, IsASCII, Character, KeySymbol, X, Y, Press, Release, Modifiers

end


;------------------------------------------------------------------------------
;+
;  Pass a mouse down event to the proper handler for the current mode.
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
pro PALM_3DObserver::OnMouseDown, oWindow, X, Y, ButtonMask, Modifiers, NumClicks

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
self.verbose = 1
  oWindow->GetProperty, DIMENSIONS=dims
  
  if (x ge 2*dims[0]/3 && x lt dims[0] && y ge 0 && y lt dims[1]/3) then begin
    sEvent = { widget_draw }
    sEvent.press = buttonMask
    sEvent.modifiers = modifiers
    sEvent.clicks = numClicks
    sEvent.x = x
    sEvent.y = y
    
    
    name = 'MainModel/VolumeModel/Translation/Rotation/CutPlaneVis'
    oCut = self.oDisplay->GetByName(name)
    if ~obj_valid(oCut) then begin
      print, 'no cut'
    endif else begin
      oCut->Event, sEvent, MSG=msg
      if (msg eq 'rotate_begin') then begin
        oCut->SetProperty, UVALUE=self.oMode
        self.oMode = oCut
        self.oDisplay->RenderScene
      endif
      if (msg ne '') then return
    endelse
  endif

    case ButtonMask of
        1: self.oMode -> OnMouseDown, oWindow, X, Y, ButtonMask, Modifiers, NumClicks
        4: begin
        ; get the draw widget's ID
            oWindow -> GetProperty, NAME=wName
            wDraw = self.oMainGUI -> Get('PALM_3DWindow')
            if ~widget_info(wDraw, /VALID) then $
                return
            wContextMenu = widget_info(wDraw, /CHILD)
            widget_displaycontextmenu, wDraw, X, Y, wContextMenu
        end
        else:
    endcase

end


;------------------------------------------------------------------------------
;+
;  Pass a mouse motion event to the proper handler for the current mode.
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
pro PALM_3DObserver::OnMouseMotion, oWindow, X, Y, ButtonMask, Modifiers, NumClicks

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

    self.oMode -> OnMouseMotion, oWindow, X, Y, ButtonMask, Modifiers, NumClicks

    name = 'MainModel/VolumeModel/Translation/Rotation/CutPlaneVis'
    ;oCut = self.oDisplay->GetByName(name)
    ;oCut->UpdateDot
end


;------------------------------------------------------------------------------
;+
;  Pass a mouse up event to the proper handler for the current mode.
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
pro PALM_3DObserver::OnMouseUp, oWindow, X, Y, ButtonMask, Modifiers, NumClicks

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

    self.oMode -> OnMouseUp, oWindow, X, Y, ButtonMask, Modifiers, NumClicks
    if (obj_isa(self.oMode, 'palm_cutplane_vis')) then begin
      self.oMode->GetProperty, MESSAGE=msg
      if (msg eq 'rotate_end') then begin
        self.oMode->GetProperty, UVALUE=obj
        self.oMode = obj
      endif
    endif
end


;------------------------------------------------------------------------------
;+
; Pass a mouse up event to the proper handler for the current mode.
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
pro PALM_3DObserver::OnWheel, oWindow, X, Y, Delta, Modifiers

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

    self.oMode -> OnWheel, oWindow, X, Y, Delta, Modifiers

end


;------------------------------------------------------------------------------
;+
;  Set up the proper event handlers for the given window mode.
;
;  @param mode {in}{type=string}{required}
;    The mode to set.  This is the prefix string for the event handler
;    method names.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS PSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DObserver::SetWindowMode, mode

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

    if obj_valid(self.oMode) then $
        obj_destroy, self.oMode

    index = where(strupcase(*self.pModes) EQ strupcase(mode[0]), count)
    if count EQ 0 then $
        mode = (*self.pModes)[0]
    self.oMode = obj_new('PALM_'+mode+'Mode', self.oDisplay)
    self.mode = mode

end


;------------------------------------------------------------------------------
;+
; Define the member variables of the PALM_3DObserver class, along with various
; internal structures.
;
; @File_comments
;   This file defines the PALM_3DObserver class for the framework of the
;   application.
;
; @field
;   mode
;     A string specifying the current mode
; @field
;   pModes
;     A pointer to a string array defining the accepted cursor modes
; @field
;   oDisplay
;     An object reference to the PALM_3DDisplay object
; @field
;   oMode
;     An object reference to one of the PALM_mode objects
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
pro PALM_3DObserver__Define

    void = {PALM_3DObserver, $
            mode     : '', $
            pModes   : ptr_new(), $
            oDisplay : obj_new(), $ ; Do not destroy
            oMainGUI : obj_new(), $ ; Do not destroy
            oMode    : obj_new(), $
            verbose  : 0B $
           }

end
