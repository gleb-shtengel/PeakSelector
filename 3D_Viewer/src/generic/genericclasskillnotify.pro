;+
; @file_comments
; This routine is called when any object-based
; TLB is destroyed, either via the system menu
; or via WIDGET_CONTROL, /DESTROY.  In general it
; will not be called directly by the user's code
; but will be specified via the KILL_NOTIFY keyword
; to WIDGET_CONTROL.
;
; <p>
; The ::Destruct method is called on the object reference.</p>
;
; @Examples <pre>
;   WIDGET_CONTROL, TLB, SET_UVALUE = self
;   WIDGET_CONTROL, TLB, KILL_NOTIFY = 'GenericClassKillNotify'
;   WIDGET_CONTROL, TLB, EVENT_PRO = 'GenericClassEvent'
;   XMANAGER, 'Generic_Class', TLB </pre>
;
; @Param
;   wID {in}{required}{type=long}
;     The widget ID of the top-level base containing
;     the "self" reference.
;
; @Uses
;   objref->Destruct
;
; @History
;   February, 2002 : JLP, RSI
;-
pro GenericClassKillNotify, wID
    compile_opt StrictArr
    ON_ERROR, 2

    WIDGET_CONTROL, wID, GET_UVALUE = self
    if (N_ELEMENTS(self) eq 1) then begin
       if (OBJ_VALID(self)) then begin
;
; Any class that uses this routine must
; have a method named "::DESTRUCT".
;
         self->Destruct
       endif
    endif
end