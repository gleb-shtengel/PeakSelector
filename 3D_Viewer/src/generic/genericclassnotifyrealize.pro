;+
; @file_comments
; This routine is called when the TLB widget of any object-based TLB
; is realized. In turn, the ::NOTIFY_REALIZE method of the class whose
; object reference is stored in the specified ID is called.
;
; @Param
;   wID {in}{required}{type=structure}
;     The widget ID of the widget containing the self reference
;     in its UVALUE at the time of widget realization.
;
; @Examples <pre>
;   TLB = WIDGET_BASE(CWParent, UVALUE = self)
;   wUValueBase = WIDGET_BASE(TLB, UNAME = 'UValueBase', $
;     UVALUE = self, $
;     EVENT_PRO = 'GenericClassEvent', $
;     KILL_NOTIFY = 'GenericClassKillNotify', $
;     NOTIFY_REALIZE = 'GenericClassNotifyRealize')
;   WIDGET_CONTROL, TLB, /REALIZE </pre>
;
; @Uses
;   objref->Notify_Realize
;
; @History
;   February, 2003 : JLP, RSI
;-
pro GenericClassNotifyRealize, wID
    compile_opt StrictArr
    ON_ERROR, 2
    WIDGET_CONTROL, wID, GET_UVALUE = self
    if (N_ELEMENTS(self) ne 0) then begin
       if (OBJ_VALID(self)) then begin
         self->Notify_Realize, wID
       endif
    endif
end