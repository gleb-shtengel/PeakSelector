;+
; @file_comments
; This routine is called when an event occurs in any object-based TLB.
; In turn, the ::EVENT method of the class whose object reference is
; stored in the EVENT.HANDLER's UVALUE is called.
;
; @Param
;   sEvent {in}{required}{type=structure}
;     Any sort of IDL GUI event
;
; @Examples <pre>
;   WIDGET_CONTROL, TLB, SET_UVALUE = self, $
;     EVENT_PRO = 'GenericClassEvent'
;   XMANAGER, 'Generic_Class', TLB </pre>
;
; @History
;   June, 2001 : JLP, RSI
;-
pro GenericClassEvent, sEvent
    compile_opt StrictArr

; Note:  in case object has been destroyed in the meantime, check widget ID validity.
    if ~WIDGET_INFO(sEvent.Handler, /VALID_ID) then return

    WIDGET_CONTROL, sEvent.Handler, GET_UVALUE = oSelf

    if (N_ELEMENTS(oSelf) eq 1) then begin
       if (OBJ_VALID(oSelf)) then begin
;
; A class that uses this routine must have a method
; named "::EVENT".
;
         if obj_valid(oSelf) then oSelf->Event, sEvent
       endif
    endif
end