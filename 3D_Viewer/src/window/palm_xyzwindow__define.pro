;------------------------------------------------------------------------------
;+
; This lifecycle method is called during the destruction of the PALM_XYZWindow
; object via OBJ_DESTROY.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZWindow::Cleanup
    compile_opt StrictArr

    self->IDLitWindow::Cleanup

end


;------------------------------------------------------------------------------
;+
; This method is responsible for creating a new
; object when invoked via OBJ_NEW().
;
; @Keyword
;   _Ref_extra {in}{optional}
;     Any extra keywords are passed transparently to the
;     superclass init.
;
; @Returns
;   This function returns 1 for success or 0 for failure.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_XYZWindow::Init, $
    _REF_EXTRA=_ref_extra
    compile_opt StrictArr
    on_error, 2

    ;; Init superclass
    if(self->IDLitWindow::Init($
        _EXTRA=_ref_extra) eq 0)then $
        return, 0

    return, 1
end


;------------------------------------------------------------------------------
;+
; This routine sets properties of the class.
;
; @Keyword
;   _Extra {inout}{optional}
;       All other keywords are ignored.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZWindow::SetProperty, $
    SPLASH = wSplash,  $
    DECOMPOSED=wasDecomposed, $
    _EXTRA=_extra
    compile_opt Strictarr
    on_error, 2

    if n_elements(_extra) gt 0 then begin
        self->IDLitWindow::SetProperty, _EXTRA=_extra
    endif
end


;------------------------------------------------------------------------------
;+
; This method is responsible for handling an expose event.
;
; @Param
;   x {in}{required}{type=float}
;     Not used.
; @Param
;   y {in}{required}{type=float}
;     Not used.
; @Param
;   width {in}{required}{type=float}
;     Not used.
; @Param
;   height {in}{required}{type=float}
;     Not used.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZWindow::OnExpose, X, Y, Width, Height
    compile_opt StrictArr
    on_error, 2

    self -> Draw

end


;------------------------------------------------------------------------------
;+
; Define the member variables of the class.
;
; @File_Comments
;  This file defines the PALM_XYZWindow class.
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Global Services Group
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZWindow__define

    void = {PALM_XYZWindow, $

            inherits IDLitWindow $

           }
end
