;------------------------------------------------------------------------------
;+
; This lifecycle method is called during the destruction of the PALM_StatusBar
; object via OBJ_DESTROY.
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
;------------------------------------------------------------------------------
pro PALM_StatusBar::Cleanup

    widget_control, widget_info(self.tlb, FIND_BY_UNAME='Draw'), $
        GET_VALUE=oWindow
    obj_destroy, oWindow
    widget_control, self.tlb, /DESTROY

end


;------------------------------------------------------------------------------
;+
; This method constructs the widgets used by the status bar
;
; @Rerturns
;   1 for success and 0 otherwise
;
; @Keyword
;   TITLE {in}{optional}{type=string}
;     A string specifying the title of the status bar
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_StatusBar::ConstructWidgets, $
    TITLE=title, CANCEL_BUTTON_PRESENT = cancel_button_present

    ss = get_screen_size()
    drawDims = [200,30]
    self.tlb = widget_base(/ROW, $
        MAP=0, $
        TITLE=title, $
        TLB_FRAME_ATTR=9)
    draw = widget_draw(self.tlb, $
        GRAPHICS_LEVEL=2, $
        UNAME='Draw', $
        XSIZE=drawDims[0], $
        YSIZE=drawDims[1])
    if cancel_button_present then self.cancelID = widget_button(self.tlb, VALUE='Cancel')
    geom = widget_info(self.tlb, /GEOMETRY)
    widget_control, self.tlb, $
        /MAP, $
        /REALIZE, $
        XOFF=(ss[0]-geom.scr_xSize)/2, $
        YOFF=(ss[1]-geom.scr_ySize)/2

    return, 1
end
;
;-------------------------------------------------------------------------------
;
FUNCTION PALM_StatusBar::CheckCancel, $
    MESSAGE=message, $
    RESPOND=respond


    ; Standard error handling.
    Catch, theError
    IF theError NE 0 THEN BEGIN
        Catch, /CANCEL
        PRINT, 'Error index: ', theError
        PRINT, 'Error message: ', !ERROR_STATE.MSG
        RETURN, 0
    ENDIF

        ; Assume no cancel operation.
    cancelFlag = 0

   ; Check for a Cancel event, if there is one.
   IF Widget_Info(self.cancelID, /Valid_ID) THEN BEGIN
      event = Widget_Event(self.cancelID, /NoWait)
      name = Tag_Names(event, /Structure_Name)
      IF name EQ 'WIDGET_BUTTON' THEN cancelFlag = 1
   ENDIF

   ; Need a response to the cancel flag?
   IF cancelFlag && Keyword_Set(respond) THEN BEGIN
       self -> Destroy
       IF N_Elements(message) EQ 0 THEN message = "Current Operation Cancelled by User"
       void = Dialog_Message(message)
   ENDIF

   RETURN, cancelFlag
END

;------------------------------------------------------------------------------
;+
; This method initializes the status bar object
;
; @Returns
;   1 for success and 0 otherwise
;
; @Keyword
;   COLOR {in}{optional}{type=bytarr}
;     A 3-element byte array specifying the color of the color bar
; @Keyword
;   TEXT_COLOR {in}{optional}{type=bytarr}
;     A 3-element byte array specifying the color of the text
;   TEXT_COLOR {in}{optional}{type=bytarr}
;
; @Keyword
;     cancel_button_present {in}{optional} defines whether there is or there is NOt cancel button
; default is NOT - GES
;
; @Keyword
;   TITLE {in}{optional}{type=string}
;     A string specifying the title of the status bar
;
; @keyword TOP_LEVEL_BASE {out}{type=widget id}{optional}
;     Status bar top level base
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;   03/10/2017 GShtengekl, added optional Cancel button and cleanup for it
;-
;------------------------------------------------------------------------------
function PALM_StatusBar::Init, $
    COLOR=inColor, $
    TEXT_COLOR=inTextColor, $
    CANCEL_BUTTON_PRESENT = in_cancel_button_present, $
    TITLE=title, $
    TOP_LEVEL_BASE=tlb

    Color = n_elements(inColor) LT 3 ? [255,0,0] : inColor[0:2]
    TextColor = n_elements(inTextColor) LT 3 ? [255,255,255] : inTextColor[0:2]
    ;print,'n_elements(CANCEL_BUTTON_PRESENT)=', n_elements(in_cancel_button_present)
    ;if  n_elements(in_cancel_button_present) then print,'CANCEL_BUTTON_PRESENT=', in_cancel_button_present
	cbp = n_elements(in_cancel_button_present) gt 0 ?   in_cancel_button_present	: 0

    if ~self->ConstructWidgets(TITLE=title, CANCEL_BUTTON_PRESENT = cbp) then $
        return, 0
    if ~self->InitializeDisplay(COLOR=color, TEXT_COLOR=TextColor) then $
        return, 0

    tlb = self.tlb

    return, 1

end


;------------------------------------------------------------------------------
;+
; This method initializes the graphics object used by the status bar
;
; @Returns
;   1 for success and 0 otherwise
;
; @Keyword
;   COLOR {in}{optional}{type=bytarr}
;     A 3-element byte array specifying the color of the color bar
; @Keyword
;   TEXT_COLOR {in}{optional}{type=bytarr}
;     A 3-element byte array specifying the color of the text
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
function PALM_StatusBar::InitializeDisplay, $
    COLOR=color, $
    TEXT_COLOR=TextColor

    oPoly = obj_new('IDLgrPolygon', $
        COLOR=color, $
        NAME='Polygon', $
        STYLE=2)
    oText = obj_new('IDLgrText', ' 0%', $
        ALIGNMENT=0.5, $
        COLOR=TextColor, $
        LOCATION=[0.5,0.3,0.5], $
        NAME='Text')
    oModel = obj_new('IDLgrModel', $
        NAME='Model')
    oView = obj_new('IDLgrView', $
        COLOR=[0,0,0], $
        VIEWPLANE_RECT=[0,0,1,1])
    oModel -> Add, oText
    oModel -> Add, oPoly
    oView -> Add, oModel
    widget_control, widget_info(self.tlb, FIND_BY_UNAME='Draw'), $
        GET_VALUE=oWindow
    oWindow -> SetProperty, GRAPHICS_TREE=oView
    oWindow -> Draw
    return, 1

end


;------------------------------------------------------------------------------
;+
; This method updates the status bar
;
; @Param
;   inPercent {in}{required}{type=float}
;     A floating point scalar (0<=scalar<=1) specifying the percent complete
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_StatusBar::UpdateStatus, inPercent

    if n_elements(inPercent) EQ 0 then $
        return

    widget_control, widget_info(self.tlb, FIND_BY_UNAME='Draw'), $
        GET_VALUE=oWindow
    oWindow -> GetProperty, GRAPHICS_TREE=oView
    oPoly = oView -> GetByName('Model/Polygon')

    percent = (inPercent>0)<1

    oPoly -> SetProperty, DATA=[[0,0], $
                                [percent,0], $
                                [percent,1], $
                                [0,1], $
                                [0,0]]
    oText = oView -> GetByName('Model/Text')
    oText -> SetProperty, STRINGS=string(fix(percent*100), FORMAT='(i2,"%")')
    oWindow -> Draw

end
;
;------------------------------------------------------------------------------
;+
; Define the member variables of the PALM_StatusBar class, along with various
; internal structures.
;
; @field
;   tlb
;     The widget ID of the status bar's top level base
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Global Services Group
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_StatusBar__Define

    void = {PALM_StatusBar, $
            tlb:0L, $
            cancelID: 0L $
           }

end
