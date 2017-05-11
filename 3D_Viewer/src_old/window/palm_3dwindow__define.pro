;------------------------------------------------------------------------------
;+
; This lifecycle method is called during the destruction of the PALM_3DWindow
; object via OBJ_DESTROY.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DWindow::Cleanup

    self -> IDLitWindow::Cleanup

end


;------------------------------------------------------------------------------
;+
; This method initializes the object
;
; @returns
;   1 for success and 0 for failure
;
; @keyword
;    VERBOSE {in}{type=boolean}{optional}
;      Set this keyword to have the object display error messages
;      in the IDL output log.
;
; @keyword
;    _REF_EXTRA {in}{optional}
;      Any additional keywords will be passed to the inherited
;      IDLitWindow object.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_3DWindow::Init, $
    VERBOSE=verbose, $
    _REF_EXTRA=_ref_extra

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

    ;; Init superclass
    if(self->IDLitWindow::Init(_EXTRA=_ref_extra) eq 0)then $
        return, 0

    return, 1

end


;------------------------------------------------------------------------------
;+
; This method is for setting the current cursor mode
;
; @param
;   mode {in}{type=string}{required}
;     A string specifying the cursor mode to be used
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DWindow::SetCurrentCursor, mode

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

    if n_elements(mode) EQ 0  then $
        return

    self -> IDLitWindow::SetCurrentCursor, mode[0]
end


;------------------------------------------------------------------------------
;+
; This method sets class properties
;
; @keyword
;   VERBOSE {in}{type=boolean}{optional}
;     Set this keyword to have the object print error messages
;     to the IDL ouput log.
; @keyword
;    _EXTRA {in}{optional}
;      All additional keywords will be passed to the inherited
;      IDLitWindow
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DWindow::SetProperty, $
    VERBOSE=verbose, $
    _EXTRA=_extra

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

    if n_elements(verbose) GT 0 then $
        self.verbose = verbose

    if n_elements(_extra) GT 0 then begin
        self -> IDLitWindow::SetProperty, _EXTRA=_extra
    endif

end


;------------------------------------------------------------------------------
;+
; Define the member variables of the PALM_3DWindow class, along with various internal
;  structures.
; @File_comments
; This file defines the PALM_3DWindow class for the framework of the application.
;
; @field
;   verbose
;     A boolean field that when set will have error messages printed to
;     the IDL output log.
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Global Services Group
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_3DWindow__Define

    void = {PALM_3DWindow, $
            inherits IDLitWindow, $
            verbose : 0B $
           }

end
