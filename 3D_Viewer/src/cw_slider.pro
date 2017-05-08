; $Id: //depot/gsg/HHMI/Phase2/src/cw_slider.pro#1 $
;
; Copyright (c) 2006, Research Systems, Inc.  All rights reserved.
;       Unauthorized reproduction prohibited.
;+
; NAME:
;   cw_slider
;
; PURPOSE:
;   A compound widget that displays a smaller slider than the standard
;   widget_slider.  Also allows for other options, e.g., dual thumbs and
;   floating point values.
;
; CATEGORY:
;   Widgets
;
; CALLING SEQUENCE:
;   wID = cw_slider(parent)
;
; PARAMETERS:
;   PARENT - The widget ID of the parent widget.
;
; KEYWORDS:
;   BACKGROUND - If set, the background of the slider can be replaced
;                with an image.  The slider bar and the thumbs still
;                appear normally.  BACKGROUND can be any of the
;                following: a 3xMxN true colour image, a MxN gray
;                scale image, a MxN black and white array (only
;                consisting of 0s and 1s), or a 3 element colour
;                vector (RGB triplet) to which the entire background
;                will be set.
;
;   COLOR - The RGB colour triplet to use to colour the thumb.  Set to
;           a scalar to disable user colour and use default colours.
;           If DUAL is set this needs to be either a scalar to disable
;           colour or a 2x3 (or 3x2) array consisting of two colour
;           triplets.
;
;   DISPLAY_TICK_VALUES - A vector defining the values on which to
;                         display ticks.  These values may be
;                         different than the values to which the
;                         slider will adhere.  Setting this keyword
;                         overrides the display property of the
;                         TICK_VALUES keyword.
;
;   DOUBLE - If set show use double values in the slider.  The default
;            is to use long integer values.  This keyword overrides
;            the FLOAT keyword.
;
;   DRAG - Set this keyword to cause events to be generated
;          continuously while the slider is being dragged by the
;          user.  Normally the widget generates position events only
;          when the slider comes to rest at its final position.
;
;   DUAL - If set the slider will have two non-crossing thumbs and two
;          value labels.
;
;   EDITABLE - Set this keyword to allow direct user editing of the
;              text widget contents. Normally, the text in text
;              widgets is uneditable.
;
;   FLOAT - If set show use floating point values in the slider.  The
;           default is to use long integer values.  This keyword is
;           ignored if the DOUBLE keyword is set.
;
;   FORMAT - A string format code to be used when displaying the
;            value(s) of the slider.
;
;   HIDE_BAR - If set, do not display the center horizontal bar.
;
;   IMAGE_EVENTS - If set clicking on the left/right image will
;                  decrease/increase the slider by one increment.
;
;   INCREMENT - The increment to be used, starting with MINIMUM, to
;               determine the values to which the slider must snap.
;               This keyword is ignored if the TICK_VALUES keyword is
;               set.
;
;   JUMP - The amount to jump when the user clicks on the bar but off
;          of the thumb.
;
;   LABEL - The text for the label on the left side of the slider.
;
;   LEFT_IMAGE - A true colour [3, M, N] image to display between the
;                left side of the slider and any left side text box.
;
;   MAP - Unless set to 0, map the widget.  The default is map=1.
;
;   MAXIMUM - The maximum value of the range encompassed by the
;             slider. If this keyword is not supplied, a default of
;             100 is used.
;
;   MINIMUM - The minimum value of the range encompassed by the
;             slider.  If this keyword is not supplied a default of 0
;             is used.
;
;   NARROW_THUMB - If set a single pixel wide line will be drawn
;                  through the entire height of the slider area
;                  instead of the default 5 pixel wide short thumb.
;
;   NOUPDATE - If set, the slider and values will not automatically
;              update the display when moved by the mouse.  The
;              display will be updated via the next call to
;              cw_slider_set or cw_slider_set_value.  This is
;              useful if processing is needed before visual updates
;              should occur.
;
;   OVERLAP - If set, and DUAL is set, then the two thumbs can both
;             range from MINIMUM to MAXIMUM.  The default is that the
;             two thumbs cannot cross, e.g., the left thumb can only
;             take values ranging from MINIMUM to the current value of
;             the right thumb.  This keyword is ignored if DUAL is not
;             set.  If this keyword is set the shift-lock option is
;             disabled.
;
;   RIGHT_IMAGE - A true colour [3, M, N] image to display between the
;                 right side of the slider and any right side text box.
;
;   SHOW_TICKS - If set visible marks will be placed on the slider on
;                the values to which the slider will snap.
;
;   SLIDERPOS - The number of pixels from the left edge of the widget
;               to the left edge of the slider itself.  This can be
;               used for aligning sliders with different sized icons
;               or labels.
;
;   SUPPRESS_VALUE - If set, do not display the value label(s).
;
;   TICK_VALUES - A vector defining the values to which the slider
;                 must adhere.  Values outside the range will be
;                 clipped.  Setting TICK_VALUES overrides the
;                 INCREMENT keyword.
;
;   TITLE - The text for the label on the left side of the slider.
;           This is identical to LABEL and is added as a convenience
;           to match the keyword in widget_slider.
;
;   TOOLTIP - If set a non null string the string will will appear as
;             the tooltip on the slider.
;
;   VALUE - The inital value setting of the widget.  If DUAL is set
;           this needs be a two element vector.
;
;   XLABELSIZE - The size of the label portion of the widget in
;                pixels.  This keyword is ignored if LABEL is not
;                set.
;
;   XSIZE - The width of the widget in pixels.  This keyword is
;           ignored if XLABELSIZE, XVALUESIZE, and XSLIDERSIZE are all
;           set.
;
;   XSLIDERSIZE - The size of the slider bar in pixels.  If this
;                 keyword is not supplied, a default of 200 is used.
;
;   XVALUESIZE - The size of the value area(s) in pixels.  This
;                keyword is ignored if SUPPRESS_VALUE is set.
;
;   YSIZE - The height of the widget in pixels.
;
;   YTHUMBSIZE - The height of the thumbs in pixels.  The default is
;                11.
;
; OUTPUTS:
;   The widget ID of the newly created compound widget.
;
; EVENT STRUCTURE:
;   The event structure returned from this widget has the following form:
;   struct = {ID:0l, TOP:0l, HANDLER:0l, $
;             DRAG:0b, INDEX:0b, DUAL:0b, LINKED:0b, $
;             VALUE:0* $
;            }
;   where:
;     ID - The widget ID of this widget.
;     TOP - The tlb of this widget.
;     HANDLER - The widget ID of the widget receiving the event.
;     DRAG - Set if drag events are set and the event was generated
;            whilst the mouse was down.
;     INDEX - Reports which thumb/value was changed, left-0, right-1.
;             This is always 0 if only one thumb is present.
;     DUAL - Set if dual thumbs exist.
;     LINKED - Set if the shift key was held down to cause dual thumbs
;              to move together (only set if dual thumbs exist).
;     VALUE - Either a single value (if one thumb) or a two element
;             vector (if two thumbs exist) giving the new value(s) of
;             the thumbs.  The datatype of VALUE depends on the
;             datatype of the slider.
;
;-

;;----------------------------------------------------------------------------
;; cw_slider_event
;;
;; Purpose:
;;   Event handler.
;;
;; Parameters:
;;   EVENT - A widget event structure.
;;
;; Keywords:
;;   None
;;
function cw_slider_event, event
  compile_opt idl2, hidden

  ;; get state structure
  Widget_Control, $
    Widget_Info(Widget_Info(Widget_Info(event.id,/PARENT),/PARENT),/CHILD), $
    GET_UVALUE=pState

  eventName = Tag_Names(event, /STRUCTURE_NAME)
  if (eventName eq 'WIDGET_TRACKING') then begin
    ;;must set !d.window to something other than -1 in order to
    ;;change the direct graphics cursor without the system
    ;;creating a new window
    Widget_Control, event.id, GET_VALUE=wID
    win = !d.window
    Wset, wID
    ;;change cursor to arrow on track_in, back to original on
    ;;track_out.  This will affect other direct graphics windows
    ;;but that is all part of the DG world.
    ;;use "Arrow" (Windows) or "XC_center_ptr" (X)
    if (event.enter) then $
      Device, CURSOR_STANDARD= $
              (!version.os_family ne 'Windows' ? 22 : 32512) $
    else Device, /CURSOR_CROSSHAIR
    Wset, win
    return, 0
  endif

  ;; set up return structure
  struct = {ID:(*pState).tlb, TOP:event.top, HANDLER:0l, $
            DRAG:0b, INDEX:0b, DUAL:(*pState).dual, LINKED:0b, $
            VALUE:((*pState).dual ? [(*pState).value2,(*pState).value1] : $
                   (*pState).value1)}

  ;; value event
  if ((eventName eq 'WIDGET_KBRD_FOCUS') || $
      (eventName eq 'WIDGET_TEXT_CH')) then begin
    ;; Ignore enter events
    if ((eventName eq 'WIDGET_KBRD_FOCUS') && event.enter) then return, 0
    ;; Get new value
    Widget_Control, event.id, GET_VALUE=value, GET_UVALUE=oldvalue
    ;; Return if nothing has changed
    if (value eq oldvalue) then return, 0
    ;; Get uname to determine which value was changed
    uname = Widget_Info(event.id, /UNAME)

    ;; Convert input to proper numeric type
    on_ioerror, bad_input
    flag = 1

    case (*pState).type of
      ;; Convert to proper value using given format
      'FLOAT' : value = Float(value)
      'DOUBLE' : value = Double(value)
      else : value = Fix(value)
    endcase
    ;; Constrain to valid range
    value >= (*pState).min
    value <= (*pState).max
    flag = 0

    ;; Handle type conversion errors
    if (flag) then begin
      bad_input:
      ;; If a bad value was entered put in a zero
      value = 0 > (*pState).min < (*pState).max
    endif

    if (uname eq 'right_value') then begin
      ;; If non overlapping dual then ensure value is not less than
      ;; the other value
      if ((*pState).dual && ~(*pState).overlap) then $
        value >= (*pState).value2
      (*pState).value1 = value
      ;; Update value to nearest allowable value
      cw_slider_set_posindex, pState, 0, /VALUE
      ;; Set value on return event structure
      struct.index = struct.dual
      struct.value[struct.index] = (*pState).value1
    endif else begin
      ;; If non overlapping then ensure value is not more than the
      ;; other value
      if (~(*pState).overlap) then $
        value <= (*pState).value1
      (*pState).value2 = value
      ;; Update value to nearest allowable value
      cw_slider_set_posindex, pState, 1, /VALUE
      ;; Set value on return event structure
      struct.value = (*pState).value2
    endelse
    ;; Update displays
    Widget_Control, event.id, $
                    SET_VALUE=String(value, FORMAT=(*pState).format), $
                    SET_UVALUE=String(value, FORMAT=(*pState).format)
    cw_slider_display_slider, event.id
    cw_slider_display_value, event.id
    ;; Return event structure
    return, struct
  endif

  ;; mouse button down
  if ((event.type eq 0) && (event.press eq 1)) then begin
    ;; determine what was hit
    ;; assume click was not on a thumb
    index = -1
    ;; check thumb 2
    if ((*pState).dual && (event.x ge (*pState).pos2) && $
        (event.x-(*pState).pos2 le 4)) then $
      index = 1
    ;; second check for narrow thumb
    if ((*pState).narrowthumb && (*pState).dual && $
        (Abs(event.x-(*pState).pos2) le 3)) then $
          index = 1
    ;; check thumb 1
    if ((event.x ge (*pState).pos1) && (event.x-(*pState).pos1 le 4)) then $
      index = 0
    ;; second check for narrow thumb
    if ((*pState).narrowthumb && (Abs(event.x-(*pState).pos1) le 3)) then $
      index = 0

    case index of
      ;; not on a thumb
      -1 : begin
        ;; if dual and not on a thumb then return
        if ((*pState).dual) then return, 0

        ;; either jump to point or advance to next value
        previndex = (*pState).posindex1
        ;; add offset so as never to have down equal 0
        (*pState).down = (*pState).posindex1 + 1
        (*pState).followMouse = 0b
        (*pState).thumb = 0b
        if (~(*pState).jump) then begin
          ;; jump to current mouse position
          (*pState).pos1 = event.x
          (*pState).cur = (*pState).pos1
          cw_slider_set_posindex, pState, 0
        endif else begin
          ;; move one increment in direction toward mouse
          direction = event.x gt (*pState).pos1
          if (direction) then begin
            (*pState).posindex1 += (*pState).jump
            (*pState).posindex1 <= N_Elements((*pState).tickpos)-1
          endif else begin
            (*pState).posindex1 -= (*pState).jump
            (*pState).posindex1 >= 0
          endelse
          ;; update position
          (*pState).pos1 = (*pState).tickpos[(*pState).posindex1]
          ;; update value
          (*pState).value1 = (*pState).tickvals[(*pState).posindex1]
        endelse
        if (previndex ne (*pState).posindex1) then begin
          ;; if position has changed then update display
          if (~(*pState).noupdate) then begin
            cw_slider_display_slider, event.id
            if ((*pState).showValue) then cw_slider_display_value, event.id
          endif
        endif
      end

      ;; on right or only thumb
      0 : begin
        (*pState).pos1 = event.x
        (*pState).cur = (*pState).pos1
        (*pState).down = (*pState).posindex1 + 1
        (*pState).followMouse = 1b
        (*pState).thumb = 0b
        cw_slider_set_posindex, pState, 0
      end

      ;; on left thumb
      1 : begin
        (*pState).pos2 = event.x
        (*pState).cur = (*pState).pos2
        (*pState).down = (*pState).posindex2 + 1
        (*pState).followMouse = 1b
        (*pState).thumb = 1b
        cw_slider_set_posindex, pState, 0
      end
      else :
    endcase

    return, 0
  end

  ;; mouse button up
  if ((event.type eq 1) && (*pState).down) then begin
    ;; save where mouse button went down
    down = (*pState).down
    ;; reset down flag to indicate mouse button is up
    (*pState).down = 0
    ;; update right or only thumb
    (*pState).pos1 = (*pState).tickpos[(*pState).posindex1]
    ;; update left thumb
    if ((*pState).dual) then $
      (*pState).pos2 = (*pState).tickpos[(*pState).posindex2]
    ;; if position has changed then update display and send event
    if (down-1 ne ([(*pState).posindex1, $
                    (*pState).posindex2])[(*pState).thumb]) then begin
      if (~(*pState).noupdate) then begin
        cw_slider_display_slider, event.id
        if ((*pState).showValue) then cw_slider_display_value, event.id
      endif
      struct.value = (*pState).dual ? [(*pState).value2,(*pState).value1] : $
        (*pState).value1
      if ((*pState).dual) then struct.index = ~((*pState).thumb eq 1)
      ;; update event structure
      struct.linked = (*pState).shiftdown || (*pState).linked
      ;; reset linked status
      (*pState).linked = 0
      return, struct
    endif
  end

  ;; mouse motion
  if ((event.type eq 2) && (*pState).down && (*pState).followMouse) then begin
    ;; save distance between thumbs
    if ((*pState).dual && (*pState).shiftdown) then $
      separation = (*pState).posindex1 - (*pState).posindex2
    if ((*pState).thumb) then begin
      ;; thumb 2
      if ((*pState).shiftdown) then begin
        ;; the shift key is down, move thumbs in tandem

        ;; save old position
        oldposindex = (*pState).posindex2
        ;; move position, constrained by location of other thumb
        (*pState).pos2 = event.x > 0 < $
                         ((*pState).slidersize - (*pState).thumbwidth)
        ;; get slider value based on thumb position.  this may force
        ;; the thumb to move back to an allowable location.
        cw_slider_set_posindex, pState, 1
        ;; update position indexes
        (*pState).posindex2 <= N_Elements((*pState).tickvals)-1-separation
        (*pState).posindex1 += (*pState).posindex2 - oldposindex
        ;; set new position for the other thumb
        (*pState).pos1 = (*pState).tickpos[(*pState).posindex1]
        ;; update event structure to indicate linked movement
        struct.linked = 1b
        (*pState).linked = 1b
      endif else begin
        if ((*pState).overlap) then begin
          ;; allow thumb access to full range of slider
          (*pState).pos2 = event.x > 0 < $
                           ((*pState).slidersize - (*pState).thumbwidth)
        endif else begin
          ;; constrain thumb to between edge and other thumb
          (*pState).pos2 = event.x > 0 < (*pState).pos1
        endelse
        ;; update position index
        cw_slider_set_posindex, pState, 1
      endelse
    endif else begin
      ;; thumb 1
      if ((*pState).dual && (*pState).shiftdown) then begin
        ;; the shift key is down, move thumbs in tandem

        ;; save old position
        oldposindex = (*pState).posindex1
        ;; move position, constrained by location of other thumb
        (*pState).pos1 = event.x > separation < $
                         ((*pState).slidersize - (*pState).thumbwidth)
        ;; get slider value based on thumb position.  this may force
        ;; the thumb to move back to an allowable location.
        cw_slider_set_posindex, pState, 0
        ;; update position indexes
        (*pState).posindex1 >= separation
        (*pState).posindex2 += (*pState).posindex1 - oldposindex
        ;; set new position for the other thumb
        (*pState).pos2 = (*pState).tickpos[(*pState).posindex2]
        ;; update event structure to indicate linked movement
        struct.linked = 1b
        (*pState).linked = 1b
      endif else begin
        if ((*pState).overlap) then begin
          ;; allow thumb access to full range of slider
          (*pState).pos1 = event.x > 0 < $
                           ((*pState).slidersize - (*pState).thumbwidth)
        endif else begin
          ;; constrain thumb to between edge and other thumb
          (*pState).pos1 = event.x > (*pState).pos2 < $
                           ((*pState).slidersize - (*pState).thumbwidth)
        endelse
        ;; update position index
        cw_slider_set_posindex, pState, 0
      endelse
    endelse
    ;; update display
    if (~(*pState).noupdate) then begin
      cw_slider_display_slider, event.id
      if ((*pState).showValue) then cw_slider_display_value, event.id
    endif
    ;; if drag events are requested then return an event structure
    if ((*pState).drag) then begin
      struct.drag = 1b
      struct.value = (*pState).dual ? [(*pState).value2,(*pState).value1] : $
        (*pState).value1
      if ((*pState).dual) then struct.index = ~((*pState).thumb eq 1)
      return, struct
    endif
  end

  ;; keyboard events
  if (event.type eq 6) then begin
    ;; record shift key
    if (*pState).dual && (event.key eq 1) && $
      ((*pState).shiftdown eq event.release) && (~(*pState).overlap) then $
        (*pState).shiftdown = ~event.release

    ;; arrow keys only work with a single thumb
    if (~event.release && ~(*pState).dual) then begin
      previndex = (*pState).posindex1
      if ((event.key eq 6) || (event.key eq 7)) then begin
        ;; up/right arrow
        ((*pState).posindex1 <= (N_Elements((*pState).tickpos)-2))++
        flag = 1
      endif
      if ((event.key eq 5) || (event.key eq 8)) then begin
        ;; down/left arrow
        ((*pState).posindex1 >= 1)--
        flag = 1
      endif
      if (event.key eq 9) then begin
        ;; Page up
        (*pState).posindex1 += (*pState).jump
        (*pState).posindex1 <= N_Elements((*pState).tickpos)-1
        flag = 1
      endif
      if (event.key eq 10) then begin
        ;; Page down
        (*pState).posindex1 -= (*pState).jump
        (*pState).posindex1 >= 0
        flag = 1
      endif
      if (event.key eq 11) then begin
        ;; Home
        (*pState).posindex1 = N_Elements((*pState).tickpos)-1
        flag = 1
      endif
      if (event.key eq 12) then begin
        ;; End
        (*pState).posindex1 = 0
        flag = 1
      endif
      if (N_Elements(flag) && (previndex ne (*pState).posindex1)) then begin
        ;; update slider position, display, and value
        (*pState).pos1 = (*pState).tickpos[(*pState).posindex1]
        if (~(*pState).noupdate) then begin
          cw_slider_display_slider, event.id
          if ((*pState).showValue) then cw_slider_display_value, event.id
        endif
        ;; update values
        (*pState).value1 = (*pState).tickvals[(*pState).posindex1]
        if ((*pState).dual) then $
          (*pState).value2 = (*pState).tickvals[(*pState).posindex2]
        ;; return event
        struct.value = (*pState).value1
        return, struct
      endif
    endif
  endif

  return, 0

end


;;----------------------------------------------------------------------------
;; cw_slider_image_event
;;
;; Purpose:
;;   Event handler.
;;
;; Parameters:
;;   EVENT - A widget event structure.
;;
;; Keywords:
;;   None
;;
function cw_slider_image_event, event
  compile_opt idl2, hidden

  ;; Tracking events
  if (Tag_Names(event, /STRUCTURE_NAME) eq 'WIDGET_TRACKING') then $
    return, cw_slider_event(event)

  ;; get state structure
  Widget_Control, $
    Widget_Info(Widget_Info(Widget_Info(event.id,/PARENT),/PARENT),/CHILD), $
    GET_UVALUE=pState

  ;; Main event handler expects event.id to be the slider draw widget
  ;; for redrawing purposes, make it so.
  id = event.id
  event.id = (*pState).wDraw

  ;; Mouse down event
  if (event.type eq 0l) then begin
    ;; Put in a fake huge left or right click position
    event.x = (Widget_Info(id, /UNAME) eq 'left') ? -100l : 10000l
    ;; Fake an event to jump thumb one value to the left
    jump = (*pState).jump
    (*pState).jump = 1
    void = cw_slider_event(event)
    (*pState).jump = jump
    return, 0
  endif

  ;; Mouse up event
  if (event.type eq 1l) then $
    return, cw_slider_event(event)

  ;; Keyboard event
  if (event.type eq 6l) then $
    return, cw_slider_event(event)

  return, 0

end


;;----------------------------------------------------------------------------
;; cw_slider_notify_realize
;;
;; Purpose:
;;   Realization event handler.
;;
;; Parameters:
;;   WID - A widget ID.
;;
;; Keywords:
;;   None
;;
pro cw_slider_notify_realize, wID
  compile_opt idl2, hidden

  ;; verify widget existance
  if ((size(wID, /TYPE) ne 3) || ~Widget_Info(wID, /VALID_ID) || $
    ~Widget_Info(Widget_Info(wID, /CHILD), /VALID_ID)) then return
  Widget_Control, Widget_Info(wID, /CHILD), GET_UVALUE=pState
  if (~Ptr_Valid(pState)) then return

  ;; set indexes for initial values
  ;; fake a shiftdown button so that a minimum value will be properly set
  (*pState).shiftdown = 1b
  cw_slider_set, (*pState).tlb, VALUE=[(*pState).value1, (*pState).value2], $
                   /FORCE_UPDATE
  (*pState).shiftdown = 0b

  ;; side images
  if ((*pState).leftImageFlag) then begin
    Widget_Control, (*pState).wLeftImage, GET_VALUE=wID
    Wset, wID
    Tv, (*pState).leftImage, /TRUE
  endif
  if ((*pState).rightImageFlag) then begin
    Widget_Control, (*pState).wRightImage, GET_VALUE=wID
    Wset, wID
    Tv, (*pState).rightImage, /TRUE
  endif

  ;; restore previous window setting.
  if ((*pState).oldWinID ne -2l) then begin
    Wset, (*pState).oldWinID
    (*pState).oldWinID = -2l
  endif

end


;;----------------------------------------------------------------------------
;; cw_slider_display_value
;;
;; Purpose:
;;   Updates the value label(s).
;;
;; Parameters:
;;   WID - A widget ID.
;;
;; Keywords:
;;   None
;;
pro cw_slider_display_value, wID
  compile_opt idl2, hidden

  if (~Widget_Info(wID, /VALID_ID)) then return

  ;; get state structure
  Widget_Control, $
    Widget_Info(Widget_Info(Widget_Info(wID,/PARENT),/PARENT),/CHILD), $
    GET_UVALUE=pState

  ;; minimum value
  if ((*pState).thumb || ((*pState).dual && (*pState).shiftdown)) then $
    Widget_Control, (*pState).wValue2, $
    SET_VALUE=String((*pState).tickvals[(*pState).posindex2], $
                     FORMAT=(*pState).format), $
    SET_UVALUE=String((*pState).tickvals[(*pState).posindex2], $
                     FORMAT=(*pState).format)

  ;; single or maximum value
  if (~(*pState).thumb || ((*pState).dual && (*pState).shiftdown)) then $
    Widget_Control, (*pState).wValue1, $
    SET_VALUE=String((*pState).tickvals[(*pState).posindex1], $
                     FORMAT=(*pState).format), $
    SET_UVALUE=String((*pState).tickvals[(*pState).posindex1], $
                     FORMAT=(*pState).format)

end


;;----------------------------------------------------------------------------
;; cw_slider_display_slider
;;
;; Purpose:
;;   Updates the slider.
;;
;; Parameters:
;;   WDRAW - A widget ID.
;;
;; Keywords:
;;   None
;;
pro cw_slider_display_slider, wDraw
  compile_opt idl2, hidden

  ;; Save previous DG state
  Tvlct, rr, gg, bb, /GET
  Device, GET_DECOMPOSED=dec
  win = !d.window

  ;; get state variable and some variables from it.
  Widget_Control, $
    Widget_Info(Widget_Info(Widget_Info(wdraw,/PARENT),/PARENT),/CHILD), $
    GET_UVALUE=pState
  xsize = (*pState).slidersize
  clrs = (*pState).clrs
  ;; set window to slider draw widget
  Widget_Control, (*pState).wDraw, GET_VALUE=wID
  ;; If widget is not yet realized then return, notify_realize will
  ;; handle displaying the slider later
  if (wID eq -1l) then return

  Wset, wID

  ;; Load colours needed for the slider bar
  Device, DECOMPOSED=0
  Tvlct, Transpose([[clrs.dark_shadow_3d],[clrs.face_3d], $
                    [clrs.light_edge_3d],[clrs.light_3d], $
                    [clrs.shadow_3d], $
                    [(*pState).color1], [(*pState).color2]])

  ;; display background
  Device, COPY=[0,0,xsize,(*pState).ysize,0,0,(*pState).pixmap]

  ;; display thumb(s).
  if (~(*pState).narrowthumb) then begin
    ;; Each arrow requires three tv commands so that the background does
    ;; not get overridden at the corners of the thumbs.

    ;; pointer 2 - down arrow
    if ((*pState).dual) then begin
      if ((*pState).useColor2) then begin
        ;; user defined colour
        Tv, (*pState).downArrowColour2[0:4,2:(*pState).ythumbSize-1], $
            (*pState).tickpos[(*pState).posindex2], $
            (*pState).thumbbottom+1
        Tv, (*pState).downArrowColour2[1:3,1], $
            (*pState).tickpos[(*pState).posindex2]+1, $
            (*pState).thumbbottom
        Tv, reform((*pState).downArrowColour2[2,0],1), $
            (*pState).tickpos[(*pState).posindex2]+2, $
            (*pState).thumbbottom-1
      endif else begin
        ;; default colours
        Tv, (*pState).downArrow[0:4,2:(*pState).ythumbSize-1], $
            (*pState).tickpos[(*pState).posindex2], $
            (*pState).thumbbottom+1
        Tv, (*pState).downArrow[1:3,1], $
            (*pState).tickpos[(*pState).posindex2]+1, $
            (*pState).thumbbottom
        Tv, reform((*pState).downArrow[2,0],1), $
            (*pState).tickpos[(*pState).posindex2]+2, $
            (*pState).thumbbottom-1
      endelse
    endif

    ;; pointer 1 - uparrow
    if ((*pState).useColor1) then begin
      ;; user defined colour
      Tv, (*pState).upArrowColour1[0:4,0:(*pState).ythumbSize-3], $
          (*pState).tickpos[(*pState).posindex1], $
          (*pState).thumbbottom
      Tv, (*pState).upArrowColour1[1:3,(*pState).ythumbSize-2], $
          (*pState).tickpos[(*pState).posindex1]+1, $
          (*pState).thumbbottom + (*pState).ythumbSize-2
      Tv, reform((*pState).upArrowColour1[2,(*pState).ythumbSize-1],1), $
          (*pState).tickpos[(*pState).posindex1]+2, $
          (*pState).thumbbottom + (*pState).ythumbSize-1
    endif else begin
      ;; default colours
      Tv, (*pState).upArrow[0:4,0:(*pState).ythumbSize-3], $
          (*pState).tickpos[(*pState).posindex1], $
          (*pState).thumbbottom
      Tv, (*pState).upArrow[1:3,(*pState).ythumbSize-2], $
          (*pState).tickpos[(*pState).posindex1]+1, $
          (*pState).thumbbottom + (*pState).ythumbSize-2
      Tv, reform((*pState).upArrow[2,(*pState).ythumbSize-1],1), $
          (*pState).tickpos[(*pState).posindex1]+2, $
          (*pState).thumbbottom + (*pState).ythumbSize-1
    endelse
  endif else begin
    ;; Define dashed bar
    y = indgen((*pState).ysize)
    x1 = fltarr((*pState).ysize)
    x1[Where(y mod 6 gt 2)] = !values.f_nan
    ;; pointer 2 - left line
    if ((*pState).dual) then begin
      x2 = fltarr((*pState).ysize)
      x2[Where(y mod 6 lt 3)] = !values.f_nan
      Plots, x2+(*pState).tickpos[(*pState).posindex2], y, /DEVICE, COLOR=0
    endif
    ;; pointer 1 - right line
    Plots, x1+(*pState).tickpos[(*pState).posindex1], y, /DEVICE, COLOR=0
  endelse

  ;; Reset everything to previous states
  Wset, win
  Device, DECOMPOSED=dec
  Tvlct, rr, gg, bb

end


;;----------------------------------------------------------------------------
;; cw_slider_kill
;;
;; Purpose:
;;   Kill routine, frees state pointer.
;;
;; Parameters:
;;   WID - A widget ID.
;;
;; Keywords:
;;   None
;;
pro cw_slider_kill, wId
  compile_opt idl2, hidden

  Widget_Control, wId, GET_UVALUE=pState
  Ptr_Free, pState

end


;;----------------------------------------------------------------------------
;; cw_slider_set_pixmap
;;
;; Purpose:
;;   Updates the pixmap which holds the background and slider bar.
;;
;; Parameters:
;;   XSIZE - The x size of the pixmap.
;;
;;   YSIZE - The y size of the pixmap.
;;
;;   BARBOTTOM - The distance from the bottom of the pixmap to the
;;               bottom of the slider bar, in pixels.
;;
;;   PIXMAP - The pixmap window ID.
;;
;;   CLRS - An array holding the colours to be used for the slider
;;          bar.
;;
;;   USERCOLOR1 - The colour to use for the first thumb, when
;;                requested.
;;
;;   USERCOLOR2 - The colour to use for the second thumb, when
;;                requested.
;;
;;   TICKPOS - If supplied, put tickmarks on the slider bar in the
;;             positions denoted by the TICKPOS array.
;;
;; Keywords:
;;   HIDEBAR - If set, do not display the horizontal bar across the
;;             slider.
;;
;;   IMAGE - Information for the background.  If not set the
;;           background defaults to the face_3d color.  Options for
;;           image are: a 3xMxN true colour image, a MxN gray scale
;;           image, a MxN black and white array (only consisting of 0s
;;           and 1s), or a 3 element colour vector (RGB triplet) to
;;           which the entire background will be set.
;;
;;   UPDATE - If the pstate is passed in create a new pixmap window.
;;            The default is to reuse the old pixmap window.
;;
pro cw_slider_set_pixmap, xsize, ysize, barbottom, pixmap, clrs, $
                          userColor1, userColor2, tickpos, $
                          UPDATE=pState, IMAGE=image, HIDEBAR=hidebar
  compile_opt idl2, hidden

  win = !d.window
  Device, GET_DECOMPOSED=dec
  Device, DECOMPOSED=0
  Tvlct, rr,gg,bb, /GET

  ;; destroy and create new pixmap if a resize is needed
  if (N_Elements(pState)) then begin
    Window, /PIXMAP, XSIZE=xsize, YSIZE=ysize
    pixmap = !d.window
    (*pState).pixmap = pixmap
  endif

  ;; load colours
  Tvlct, Transpose([[clrs.dark_shadow_3d],[clrs.face_3d], $
                    [clrs.light_edge_3d],[clrs.light_3d], $
                    [clrs.shadow_3d], $
                    [userColor1], [userColor2]])
  ;; switch to the pixmap
  Wset, pixmap

  ;; background
  if (N_Elements(image) gt 2) then begin
    if (Size(image, /N_DIMENSIONS) eq 3) then begin
      ;; true colour image
      true = (Where(Size(image, /DIMENSIONS) eq 3))[0]
      im = Color_Quan(image, true+1, rrr, ggg, bbb, COLORS=245)
      Tvlct, rrr, ggg, bbb, 7
      Tv, im+7b
    endif else begin
      if (N_Elements(image) eq 3) then begin
        ;; single colour
        Tvlct, image[0], image[1], image[2], 7
        Erase, 7
      endif else if (max(image) eq 1) then begin
        ;; black and white image
        Tvlct, [0,255], [0,255], [0,255] ,7
        Tv, image+7b
      endif else begin
        ;; gray scale image.  scale to 248 colours.
        im = Bytscl(image, TOP=248)+7b
        Tvlct, Findgen(248)/247.0*255, Findgen(248)/247.0*255, $
               Findgen(248)/247.0*255, 7
        Tv, im
      endelse
    endelse
  endif else begin
    ;; put face_3d as overall background colour
    Erase, 1
  endelse

  if (~Keyword_Set(hidebar)) then begin
    ;; background grove
    ;; light_edge_3d
    Tv, Bytarr(xsize)+2b, 0, barbottom
    ;; light_3d
    Tv, Bytarr(xsize)+3b, 0, barbottom+1
    ;; dark_shadow_3d
    Tv, Bytarr(xsize)+0b, 0, barbottom+2
    ;; shadow_3d
    Tv, Bytarr(xsize)+4b, 0, barbottom+3

    ;; put in end bars
    Tv, Transpose(Bytarr(3))+4b, 0, barbottom+1
    Tv, Transpose(Bytarr(2))+3b, xsize-2, barbottom+1
    Tv, Transpose(Bytarr(4))+2b, xsize-1, barbottom

    ;; add tick marks, if requested
    if (N_Elements(tickpos) gt 0) then begin
      for i=0,N_Elements(tickpos)-1 do begin
        Tv, Transpose([[0,0,0,3,0,0,0,0],[2,2,2,3,0,2,2,2]]), $
            tickpos[i]+1, barbottom-2
      endfor
    endif
  endif

  ;; reset things to previous state
  Device, DECOMPOSED=dec
  Tvlct, rr, gg, bb
  Wset, win

end


;;----------------------------------------------------------------------------
;; cw_slider_set_posindex
;;
;; Purpose:
;;   Sets the index in the position and value arrays for each thumb.
;;
;; Parameters:
;;   PSTATE - A pointer to the state structure.
;;
;;   INDEX - Determines which thumb to use.
;;           0 - right/only thumb, 1 - left thumb
;;
;; Keywords:
;;   VALUE - If set, determine index from the value, the default is to
;;           determine the index from the position.
;;
pro cw_slider_set_posindex, pState, index, $
                            VALUE=value
  compile_opt idl2, hidden

  ;; determine index of closest value

  if (Keyword_Set(value)) then begin
    if (Keyword_Set(index)) then begin
      ;; value 2
      distance = Abs((*pState).value2 - (*pState).tickvals)
      minD = Min(distance)
      (*pState).posindex2 = (Where(distance eq minD))[0]
    endif else begin
      ;; value 1
      distance = Abs((*pState).value1 - (*pState).tickvals)
      minD = Min(distance)
      (*pState).posindex1 = (Where(distance eq minD))[0]
    endelse
  endif else begin
    if (Keyword_Set(index)) then begin
      ;; value 2
      distance = Abs((*pState).pos2 - (*pState).tickpos)
      minD = Min(distance)
      (*pState).posindex2 = (Where(distance eq minD))[0]
    endif else begin
      ;; value 1
      distance = Abs((*pState).pos1 - (*pState).tickpos)
      minD = Min(distance)
      (*pState).posindex1 = (Where(distance eq minD))[0]
    endelse
  endelse
  ;; update values
  (*pState).value1 = (*pState).tickvals[(*pState).posindex1]
  if ((*pState).dual) then $
    (*pState).value2 = (*pState).tickvals[(*pState).posindex2]
  ;; update positions
  (*pState).pos1 = (*pState).tickpos[(*pState).posindex1]
  if ((*pState).dual) then $
    (*pState).pos2 = (*pState).tickpos[(*pState).posindex2]

end


;;----------------------------------------------------------------------------
;; cw_slider_get
;;
;; Purpose:
;;   Returns requested properties.
;;
;; Parameters:
;;   WID - A widget ID.
;;
;; Keywords:
;;   COLOR - The RGB colour triplet used to colour the thumb(s).
;;
;;   DOUBLE - True if the slider is using double precision values.
;;
;;   DUAL - True if dual thumbs are present.
;;
;;   FLOAT - True is the slider is using floating point precision
;;           values.
;;
;;   LABEL - The label on the left side of the slider, or '' if not
;;           set.
;;
;;   MAP - True is the widget is mapped.
;;
;;   MAXIMUM - The maximum value of the range encompassed by the
;;             slider.
;;
;;   MINIMUM - The minimum value of the range encompassed by the
;;             slider.
;;
;;   OVERLAP - True is dual thumbs are present and the thumbs are
;;             allowed to overlap.
;;
;;   TICK_VALUES - A vector defining the values to which the slider
;;                 must adhere.
;;
;;   VALUE - The current value of the thumb(s).
;;
;;   XLABELSIZE - The size of the label portion of the widget in
;;                pixels.
;;
;;   XSIZE - The width of the widget in pixels.
;;
;;   XSLIDERSIZE - The size of the slider bar in pixels.
;;
;;   XVALUESIZE - The size of the value area(s) in pixels.
;;
pro cw_slider_get, wID, $
                   VALUE=value, COLOR=color, DUAL=dual, $
                   DOUBLE=double, FLOAT=float, FORMAT=format, $
                   LABEL=label, MAP=map, MAXIMUM=maximum, $
                   MINIMUM=minimum, OVERLAP=overlap, $
                   TICK_VALUES=tick_values, UVALUE=uvalue, $
                   XLABELSIZE=xlabelsize, $
                   XSIZE=xsize, XSLIDERSIZE=xslidersize, $
                   XVALUESIZE=xvaluesize
  compile_opt idl2, hidden
  on_error, 2

  if (~Widget_Info(wID, /VALID_ID)) then return

  Widget_Control, Widget_Info(wId, /CHILD), GET_UVALUE=pState
  if (~Ptr_Valid(pState)) then return
  ;; Ensure this is a proper cw_slider structure
  wh = Where(Tag_Names(*pState) eq 'CW_SLIDER_ID_TAG', cnt)
  if (cnt eq 0) then return
  ;; uvalue
  if arg_present(uvalue) then $
    uvalue = (*pState).uvalue

  ;; value
  if (Arg_Present(value)) then begin
    value = (*pState).value1
    if ((*pState).dual) then value = [(*pState).value2, value]
  endif

  ;; color
  if (Arg_Present(color)) then begin
    color = (*pState).useColor1 ? (*pState).color1 : -1
    if ((*pState).dual && (*pState).useColor1) then $
      color = [[(*pState).color2], [color]]
  endif

  ;; dual
  if (Arg_Present(dual)) then dual = (*pState).dual

  ;; double
  if (Arg_Present(double)) then double = (*pState).type eq 'DOUBLE'

  ;; float
  if (Arg_Present(float)) then float = (*pState).type eq 'FLOAT'

  ;; format
  if (Arg_Present(format)) then format = (*pState).format

  ;; label
  if (Arg_Present(label)) then begin
    label = ''
    if ((*pState).wLabel ne 0) then $
      Widget_Control, (*pState).wLabel, GET_VALUE=label
  endif

  ;; map
  if (Arg_Present(map)) then map = Widget_Info((*pState).tlb, /MAP)

  ;; maximum
  if (Arg_Present(maximum)) then maximum = (*pState).max

  ;; minimum
  if (Arg_Present(minimum)) then minimum = (*pState).min

  ;; overlap
  if (Arg_Present(overlap)) then overlap = (*pState).overlap

  ;; tick_values
  if (Arg_Present(tick_values)) then tick_values = (*pState).tickvals

  ;; xlabelsize
  if (Arg_Present(xlabelsize)) then xlabelsize = (*pState).labelSize

  ;; xsize
  if (Arg_Present(xsize)) then begin
    geo = Widget_Info((*pState).tlb, /GEOMETRY)
    xsize = geo.xsize
  endif

  ;; xslidersize
  if (Arg_Present(xslidersize)) then xslidersize = (*pState).slidersize

  ;; xvaluesize
  if (Arg_Present(xvaluesize)) then xvaluesize = (*pState).valuesize

end


;;----------------------------------------------------------------------------
;; cw_slider_set
;;
;; Purpose:
;;   Sets requested properties.
;;
;; Parameters:
;;   WID - A widget ID.
;;
;; Keywords:
;;   BACKGROUND - If set, update the background of the slider.  If
;;                this value is a scalar the default background will
;;                be used.
;;
;;   COLOR - If set, set the colour (RGB triplet) of the thumbs.
;;           Setting this to a scalar sets the colour to the default.
;;
;;   DISPLAY_VALUE - If set, sets the strings displayed in the value
;;                   label(s).
;;
;;   FORCE_UPDATE - If set force an update of the widget.
;;
;;   INDEX - If dual thumbs are present INDEX can be used to specify
;;           which thumb/value to update.
;;           0-Left thumb/value, 1-Right thumb/value.
;;
;;   LABEL - The text for the label (if one exists) on the left side
;;           of the slider.
;;
;;   MAP - If 0 then unmap the widget, otherwise map the widget.
;;
;;   NOUPDATE - If set, the slider and values will not automatically
;;              update the display when moved by the mouse.  The
;;              display will be updated via the next call to
;;              cw_slider_set or cw_slider_set_value.
;;
;;   SLIDERPOS - The number of pixels from the left edge of the widget
;;               to the left edge of the slider itself.
;;
;;   TOOLTIP - The value of the tooltip to appear on the slider.
;;
;;   VALUE - If set, update the value of the widget.
;;
;;   XLABELSIZE - The size of the label portion of the widget in
;;                pixels.
;;
;;   XSLIDERSIZE - The size of the slider bar in pixels.
;;
;;   XVALUESIZE - The size of the value area(s) in pixels.
;;
pro cw_slider_set, wID, $
                   VALUE=value, COLOR=color, DISPLAY_VALUE=strval, $
                   BACKGROUND=background, FORCE_UPDATE=forceupdate, $
                   INDEX=index, XLABELSIZE=xlabelsize, $
                   XSLIDERSIZE=xslidersize, XVALUESIZE=xvaluesize, $
                   SLIDERPOS=sliderpos, TOOLTIP=tooltip, LABEL=label, $
                   MAP=map, NOUPDATE=noupdate
  compile_opt idl2, hidden
  on_error, 2

  if (~Widget_Info(wID, /VALID_ID)) then return

  Widget_Control, Widget_Info(wId, /CHILD), GET_UVALUE=pState
  if (~Ptr_Valid(pState)) then return
  ;; Ensure this is a proper cw_slider structure
  wh = Where(Tag_Names(*pState) eq 'CW_SLIDER_ID_TAG', cnt)
  if (cnt eq 0) then return

  updateSlider = 0
  updateValue = 0

  ;; background
  if (N_Elements(background) ne 0) then begin
    cw_slider_set_pixmap, (*pState).slidersize, (*pState).ysize, $
                          (*pState).barbottom, (*pState).pixmap, $
                          (*pState).clrs, (*pState).color1, $
                          (*pState).color2, IMAGE=background, $
                          hidebar=(*pState).hidebar
    updateSlider = 1
  endif

  ;; color
  if (N_Elements(color) ne 0) then begin
    if (N_Elements(color) lt 3) then begin
      (*pState).useColor1 = 0
      (*pState).useColor2 = 0
    endif
    if ((N_Elements(color) eq 3) && ~(*pState).dual) then begin
      (*pState).color1 = color
      (*pState).useColor1 = 1
    endif
    if ((N_Elements(color) eq 6) && (*pState).dual) then begin
      (*pState).color2 = color[0:2]
      (*pState).color1 = color[3:5]
      (*pState).useColor1 = 1
      (*pState).useColor2 = 1
    endif
    updateSlider = 1
  endif

  ;; value
  if (N_Elements(value) ne 0) then begin
    oldVal1 = (*pState).value1
    oldVal2 = (*pState).value2
    if (N_Elements(value) eq 2 and (*pState).dual) then begin
      (*pState).value1 = value[1]
      cw_slider_set_posindex, pState, 0, /VALUE
      (*pState).value2 = value[0]
      cw_slider_set_posindex, pState, 1, /VALUE
    endif else begin
      if (Keyword_Set(index)) then begin
        (*pState).value2 = value[0]
        cw_slider_set_posindex, pState, 1, /VALUE
      endif else begin
        (*pState).value1 = value[0]
        cw_slider_set_posindex, pState, 0, /VALUE
      endelse
    endelse
    if (Keyword_Set(forceupdate) || ((*pState).value1 ne oldVal1) || $
        ((*pState).value2 ne oldVal2)) then begin
      updateSlider = 1
      updateValue = 1
    endif
  endif

  ;; xlabelsize
  if (N_Elements(xlabelsize) ne 0) && ((*pState).wLabel ne 0) then begin
    Widget_Control, (*pState).wLabel, XSIZE=xlabelsize
  endif

  ;; xslidersize
  if (N_Elements(xslidersize) ne 0) then begin
    (*pState).slidersize = xslidersize
    cw_slider_set_pixmap, (*pState).slidersize, (*pState).ysize, $
                            (*pState).barbottom, (*pState).pixmap, $
                            (*pState).clrs, (*pState).color1, $
                            (*pState).color2, UPDATE=pState, $
                            hidebar=(*pState).hidebar
    updateSlider = 1
  endif

  ;; xvaluesize
  if (N_Elements(xvaluesize) ne 0) then begin
    Widget_Control, (*pState).wValue1, XSIZE=xvaluesize
    if ((*pState).dual) then $
      Widget_Control, (*pState).wValue2, XSIZE=xvaluesize
  endif

  ;; label
  if (N_Elements(label) ne 0) && ((*pState).wLabel ne 0) then begin
    Widget_Control, (*pState).wLabel, VALUE=label[0]
  endif

  ;; sliderpos.  This is done after the other sizes because it
  ;; overrides any other settings.
  if (N_Elements(sliderpos) ne 0) then begin
    valueSize = ((*pState).wValue2 ne 0l) ? $
                (Widget_Info((*pState).wValue2, /GEOMETRY)).scr_xsize : 0l
    Widget_Control, (*pState).wLabel, SCR_XSIZE=sliderPos-valueSize
    updateSlider = 1
  endif

  ;; map
  if (N_Elements(map) ne 0) then begin
    Widget_Control, (*pState).tlb, MAP=Keyword_Set(map)
  endif

  ;; noupdate
  if (N_Elements(noupdate) ne 0) then $
    (*pState).noupdate = Keyword_Set(noupdate)

  ;; tooltip
  if (N_Elements(tooltip) ne 0) then begin
    ;; tooltips cannot be nullstrings on unix.
    if (tooltip eq '') && (Strupcase(!version.os_family) ne 'WINDOWS') then $
      tooltip=' '
    Widget_Control, (*pState).wDraw, TOOLTIP=String(tooltip)
  endif

  if (updateSlider) then $
    cw_slider_display_slider, (*pState).wDraw
  if (updateValue) then begin
    ;; Fake a shiftdown so that both values update if slider is in
    ;; dual mode
    currentShiftDown = (*pState).shiftdown
    (*pState).shiftdown = 1b
    cw_slider_display_value, (*pState).wValue1
    (*pState).shiftdown = currentShiftDown
  endif

  ;; display_value
  ;; As the above call to display_value sets the labels, this must
  ;; happen after that to avoid overwriting.
  if (N_Elements(strval) ne 0) then begin
    if (N_Elements(strval) eq 2 and (*pState).dual) then begin
      Widget_Control, (*pState).wValue1, $
                      SET_VALUE=strval[1], SET_UVALUE=strval[1]
      Widget_Control, (*pState).wValue2, $
                      SET_VALUE=strval[0], SET_UVALUE=strval[0]
    endif else begin
      Widget_Control, (*pState).wValue1, $
                      SET_VALUE=strval[0], SET_UVALUE=strval[0]
    endelse
  endif

end


;;----------------------------------------------------------------------------
;; cw_slider_set_value
;;
;; Purpose:
;;   Allows a Widget_Control call to set the value of the widget.
;;
;; Parameters:
;;   WID - A widget ID.
;;
;;   VALUE - The value to which the widget should be set.
;;
;; Keywords:
;;   None
;;
pro cw_slider_set_value, wID, value
  compile_opt idl2, hidden

  cw_slider_set, wID, VALUE=value

end


;;----------------------------------------------------------------------------
;; cw_slider_get_value
;;
;; Purpose:
;;   Returns the current value of the widget.
;;
;; Parameters:
;;   WID - A widget ID.
;;
;; Keywords:
;;   None
;;
function cw_slider_get_value, wID
  compile_opt idl2, hidden

  cw_slider_get, wID, VALUE=value
  return, value

end


;;----------------------------------------------------------------------------
;; cw_slider
;;
;; Purpose:
;;   Main routine.  See top of file for inputs.
;;
function cw_slider, parent, $
                    BACKGROUND=background, $
                    COLOR=colorIn, $
                    DISPLAY_TICK_VALUES=dispTicks, $
                    DOUBLE=double, $
                    DRAG=drag, $
                    DUAL=dual, $
                    EDITABLE=editable, $
                    FLOAT=float, $
                    FORMAT=format, $
                    HIDE_BAR=hideBar, $
                    IMAGE_EVENTS=imageEvents, $
                    INCREMENT=incrementIn, $
                    JUMP=jumpIn, $
                    LABEL=labelIn, $
                    LEFT_IMAGE=leftImageIn, $
                    MAP=map, $
                    MAXIMUM=maxxIn, $
                    MINIMUM=minnIn, $
                    NARROW_THUMB=narrowThumb, $
                    NOUPDATE=noupdate, $
                    OVERLAP=overlap, $
                    RIGHT_IMAGE=rightImageIn, $
                    SHOW_TICKS=showTicks, $
                    SLIDERPOS=sliderpos, $
                    SUPPRESS_VALUE=noval, $
                    TICK_VALUES=tickvIn, $
                    TITLE=titleIn, $
                    TOOLTIP=tooltipIn, $
                    UVALUE=uvalue, $
                    UNAME=uname, $
                    VALUE=valueIn, $
                    XLABELSIZE=labelsizeIn, $
                    XSIZE=xsizeIn, $
                    XSLIDERSIZE=slidersizeIn, $
                    XVALUESIZE=valuesizeIn, $
                    YSIZE=ysizeIn, $
                    YTHUMBSIZE=ythumbSizeIn, $
                    _EXTRA=_extra
  compile_opt idl2, hidden

  on_error, 2

  if (~Widget_Info(parent, /VALID_ID)) then $
    Message, 'Parent must be a valid widget ID'

  ;; set up needed value
  thumb_width = Keyword_Set(narrowThumb) ? 1 : 5

  ;; check inputs
  showval = ~Keyword_Set(noval)
  if (N_Elements(map) ne 0) then map=Keyword_Set(map) else map=1
  ;; TITLE can be used to set the label value
  if (N_Elements(titleIn) eq 0) then label='' else label=titleIn[0]
  ;; But LABEL overrides
  if (N_Elements(labelIn) ne 0) then label=labelIn[0] else label=''
  if (N_Elements(minnIn) eq 0) then minn=0 else minn=minnIn[0]
  if (N_Elements(maxxIn) eq 0) then maxx=100 else maxx=maxxIn[0]
  if (N_Elements(ythumbSizeIn) eq 0) then $
    ythumbSize = 11 else ythumbSize=(Long(ythumbSizeIn[0]) > 5)
  if (N_Elements(ysizeIn) eq 0) then begin
    ysize = 12
    yflag = 0
  endif else begin
    ysize = (Long(ysizeIn[0]) > 6)
    yflag = 1
  endelse
  if (N_Elements(tooltipIn) ne 0) then $
    tooltip=String(tooltipIn[0]) $
  else tooltip=''

  leftImageFlag = 0b
  leftImage = 0b
  rightImageFlag = 0b
  rightImage = 0b
  if (N_Elements(leftImageIn) ne 0) then begin
    ;; verify true colour image
    sz = Size(leftImageIn, /DIMENSIONS)
    if ((N_Elements(sz) eq 3) && (sz[0] eq 3)) then begin
      leftImageFlag = 1b
      leftImage = leftImageIn
    endif
  endif
  if (N_Elements(rightImageIn) ne 0) then begin
    ;; verify true colour image
    sz = Size(rightImageIn, /DIMENSIONS)
    if ((N_Elements(sz) eq 3) && (sz[0] eq 3)) then begin
      rightImageFlag = 1b
      rightImage = rightImageIn
    endif
  endif

  ;; calculate distance from bottom of draw window to bottom of the
  ;; slider bar.
  barbottom = (ysize-12)/2+4
  ;; calculate size of thumb and location of thumb bottom
  ythumbSize <= ysize-1
  thumbbottom = barbottom - ((ythumbSize - 4) / 2)
  ;; check input sizes
  labelSize = N_Elements(labelsizeIn) ne 0 ? Long(labelsizeIn[0]) : 0
  valueSize = N_Elements(valuesizeIn) ne 0 ? Long(valuesizeIn[0]) : 0
  sliderSize = N_Elements(slidersizeIn) ne 0 ? Long(slidersizeIn[0]) : 0
  sliderPos = N_Elements(sliderpos) ne 0 ? Long(sliderpos[0]) : 0
  ;; check to ensure all three values have been entered
  xsize = labelSize + valueSize + sliderSize
  if (xsize eq 0) then slidersize=200

  if (N_Elements(tickvIn) ne 0) then begin
    ;; ensure increasing order
    tickvals = Double(tickvIn[sort(tickvIn)])
    ;; filter out values outside the min-max range
    tickvals = tickvals[Where(tickvals ge minn and tickvals le maxx)]
    ;; determine valid tick positions
    tickpos = Long((Double(tickvals)-minn)/(Double(maxx)-minn)* $
                   (slidersize-thumb_width))
    ;; set jump value
    jump = Keyword_Set(incrementIn)
  endif else if (N_Elements(incrementIn) ne 0) then begin
    ;; determine number of values that can fit on the slider
    nvals = Fix((Double(maxx)-minn)/incrementIn[0]+1) > 2
    tickvals = Dindgen(nvals)*incrementIn[0]+minn
    ;; determine value tick positions
    tickpos = Long((Double(tickvals)-minn)/(Double(maxx)-minn)* $
                   (slidersize-thumb_width))
    jump = 1
  endif else begin
    ;; if ticks or increment is not set then each pixel in the slider
    ;; bar is a 'tick'
    tickpos = Lindgen(slidersize-(thumb_width-1))
    tickvals = Double(tickpos)/(slidersize-(thumb_width-1)-1)* $
               (Double(maxx)-minn)+minn
    jump = 0
  endelse
  if (N_Elements(jumpIn) ne 0) then jump = Long(jumpIn[0])

  ;; set initial values.
  if (Keyword_Set(dual)) then begin
    if (N_Elements(valueIn) eq 2) then begin
      value1 = valueIn[0]
      value2 = valueIn[1]
    endif else begin
      value1 = minn
      value2 = maxx
    endelse
    value2 <= maxx
    value1 >= minn
    value1 <= value2
    if (N_Elements(colorIn) eq 6) then begin
      sz = Size(colorIn, /dimensions)
      if (Array_Equal(sz, [2l, 3l])) then begin
        color2 = Reform(colorIn[0,0:2])
        color1 = Reform(colorIn[1,0:2])
      endif else begin
        color2 = colorIn[0:2]
        color1 = colorIn[3:5]
      endelse
      useColor1 = 1
      useColor2 = 1
    endif else if (N_Elements(colorin) eq 3) then begin
      color2 = colorin
      color1 = [0,0,0]
      useColor1 = 1
      useColor2 = 0
    endif else begin
      color1 = [0,0,0]
      color2 = [0,0,0]
      useColor1 = 0
      useColor2 = 0
    endelse
  endif else begin
    value1 = N_Elements(valueIn) ne 0 ? valueIn[0] : ((maxx-minn)/2.0)+minn
    value1 >= minn
    value1 <= maxx
    value2 = 0b
    if (N_Elements(colorIn) ge 3) then begin
      color1 = colorIn[0:2]
      useColor1 = 1
    endif else begin
      color1 = [0,0,0]
      useColor1 = 0
    endelse
    color2 = [0,0,0]
    useColor2 = 0
  endelse

  type = Keyword_Set(double) ? 'DOUBLE' : $
    (Keyword_Set(float) ? 'FLOAT' : 'LONG')
  case type of
    'FLOAT' : begin
      tickvals = Float(tickvals)
      format = N_Elements(format) ne 0 ? format : '(F0.2)'
      value1 = Float(value1)
      value2 = Float(value2)
    end
    'LONG' : begin
      tickvals = Long(tickvals)
      format = N_Elements(format) ne 0 ? format : '(I0)'
      value1 = Long(value1)
      value2 = Long(value2)
    end
    'DOUBLE' : begin
      format = N_Elements(format) ne 0 ? format : '(D0.2)'
      value1 = Double(value1)
      value2 = Double(value2)
    end
    else :
  endcase

  ;; Verify format code
  catch, error
  if (error ne 0) then begin
    catch, /CANCEL
    Message, 'Invalid format code'
  endif else begin
    void = String(!pi, FORMAT=format)
  endelse

  ;; Set up default catch block
  catch, error
  if (error ne 0) then begin
    catch, /CANCEL
    Message, 'Invalid value'
  endif

  ;; save current window states
  win = !d.window

  ;; create a pixmap window to hold the background slider bar
  Window, /FREE, /PIXMAP, XSIZE=slidersize, YSIZE=ysize
  pixmap = !d.window
  ;; get system colours
  clrs = Widget_Info(Widget_Base(),/SYSTEM_COLORS)
  ;; Set a few colours to be the same across platforms
  clrs.dark_shadow_3d = [113, 111, 100]
  clrs.light_edge_3d = [255, 255, 255]
  clrs.light_3d = [241, 239, 226]
  clrs.shadow_3d = [157, 157, 161]
  if (Keyword_Set(showTicks)) then begin
    ;; use calculated tick positions
    tickDisplay = tickpos
    ;; if set, use display tick values
    if (N_Elements(dispTicks) ne 0) then begin
      dTicks = tickvals[Value_Locate(tickvals, [dispTicks])]
      tickDisplay = Long((Double(dTicks)-minn)/(Double(maxx)-minn)* $
                         (slidersize-thumb_width))
    endif
  endif
  ;; set the pixmap to hold the background slider bar.
  cw_slider_set_pixmap, slidersize, ysize, barbottom, pixmap, clrs, $
                        color1, color2, tickDisplay, IMAGE=background, $
                        HIDEBAR=Keyword_Set(hideBar)
  Wset, win

  ;; up pointing arrow, default colours
  upArrow = Reverse(Byte([[1,1,0,1,1], $
                          [1,2,4,0,1], $
                          [[2,3,1,4,0] # Replicate(1,ythumbSize-4)], $
                          [2,4,4,4,0], $
                          [0,0,0,0,0]]),2)
  ;; down pointing arrow, default colours
  downArrow = Byte([[1,1,0,1,1], $
                    [1,2,4,0,1], $
                    [[2,3,1,4,0] # Replicate(1,ythumbSize-4)], $
                    [2,3,3,4,0], $
                    [2,2,2,2,0]])
  ;; up pointing arrow, user colour 1
  upArrowColour1 = Reverse(Byte([[1,1,0,1,1], $
                                 [1,2,5,0,1], $
                                 [[2,5,5,5,0] # Replicate(1,ythumbSize-4)], $
                                 [2,5,5,5,0], $
                                 [0,0,0,0,0]]),2)
  ;; down pointing arrow, user colour 2
  downArrowColour2 = Byte([[1,1,0,1,1], $
                           [1,2,6,0,1], $
                           [[2,6,6,6,0] # Replicate(1,ythumbSize-4)], $
                           [2,6,6,6,0], $
                           [2,2,2,2,0]])

  wLabel = 0l
  wValue1 = 0l
  wValue2 = 0l

  ;; main widget base
  tlb = Widget_Base(parent, /ROW, MAP=0, $
                    SPACE=2, XPAD=0, YPAD=0, TAB_MODE=1, $
                    PRO_SET_VALUE='cw_slider_set_value', $
                    FUNC_GET_VALUE='cw_slider_get_value', $
                    UNAME=uname, _EXTRA=_extra)

  ;; label
  if ((label ne '') || (sliderPos ne 0)) then begin
    wBase = Widget_Base(tlb, /ROW, /ALIGN_CENTER, SPACE=0, XPAD=0, YPAD=0)
    if (labelSize ne 0) then begin
      wLabel = Widget_Label(wBase, XSIZE=labelSize, VALUE=label)
    endif else begin
      wLabel = Widget_Label(wBase, VALUE=label)
    endelse
  endif

  if (showval) then begin
    ;; determine max size needed to hold value string
    stringsize1 = Widget_Info(tlb, STRING_SIZE=String(minn, FORMAT=format))
    stringsize2 = Widget_Info(tlb, STRING_SIZE=String(maxx, FORMAT=format))
    sz = valueSize ne 0 ? valueSize : Max([Stringsize1[0], Stringsize2[0]])
    ;; Add room for widget_text window dressing
    sizeOffset = (Strupcase(!version.os_family) eq 'WINDOWS') ? 6 : 16
  endif

  if (showval && Keyword_Set(dual)) then begin
    ;; left value label
    wBase = Widget_Base(tlb, /ROW, /ALIGN_CENTER, SPACE=0, XPAD=0, YPAD=0)
    wValue2 = Widget_Text(wBase, EDITABLE=Keyword_Set(editable), $
                          /KBRD_FOCUS_EVENTS, $
                          SCR_YSIZE=stringsize2[1]+sizeOffset, $
                          SCR_XSIZE=sz+sizeOffset, $
                          UNAME='left_value', $
                          VALUE=String(minn, FORMAT=format))
  endif

  ;; Left image
  if (leftImageFlag) then begin
    imagesz = Size(leftImage, /DIMENSIONS)
    wBase = Widget_Base(tlb, SPACE=0, XPAD=0, YPAD=0, /ROW, /ALIGN_CENTER)
    wLeftImage = Widget_Draw(wBase, XSIZE=imagesz[1], YSIZE=imagesz[2], $
                             /ALIGN_CENTER, GRAPHICS_LEVEL=0, $
                             /TRACKING_EVENTS, UNAME='left', $
                             KEYBOARD_EVENTS=2, RETAIN=2, $
                             BUTTON_EVENTS=Keyword_Set(imageEvents))
    if (tooltip ne '') then Widget_Control, wLeftImage, TOOLTIP=tooltip
  endif else wLeftImage = 0l

  ;; draw widget.  extra wBase is needed so that the number of steps
  ;; from draw or label widgets to tlb is consistant.
  wBase = Widget_Base(tlb, SPACE=0, XPAD=0, YPAD=0, /ROW, /ALIGN_CENTER)
  wDraw = Widget_Draw(wBase, XSIZE=Slidersize, YSIZE=ysize, /EXPOSE_EVENTS, $
                      KEYBOARD_EVENTS=2, /MOTION_EVENTS, /BUTTON_EVENTS, $
                      /TRACKING_EVENTS, /ALIGN_CENTER, RETAIN=2, $
                      GRAPHICS_LEVEL=0)
  if (tooltip ne '') then Widget_Control, wDraw, TOOLTIP=tooltip

  ;; Right image
  if (rightImageFlag) then begin
    imagesz = Size(rightImage, /DIMENSIONS)
    wBase = Widget_Base(tlb, SPACE=0, XPAD=0, YPAD=0, /ROW, /ALIGN_CENTER)
    wRightImage = Widget_Draw(wBase, XSIZE=imagesz[1], YSIZE=imagesz[2], $
                              /ALIGN_CENTER, GRAPHICS_LEVEL=0, $
                              /TRACKING_EVENTS, UNAME='right', $
                              KEYBOARD_EVENTS=2, RETAIN=2, $
                              BUTTON_EVENTS=Keyword_Set(imageEvents))
    if (tooltip ne '') then Widget_Control, wRightImage, TOOLTIP=tooltip
  endif else wRightImage = 0l

  if (showval) then begin
    ;; right value label
    wBase = Widget_Base(tlb, /ROW, /ALIGN_CENTER, SPACE=0, XPAD=0, YPAD=0)
    wValue1 = Widget_Text(wBase, EDITABLE=Keyword_Set(editable), $
                          /KBRD_FOCUS_EVENTS, $
                          SCR_YSIZE=stringsize2[1]+sizeOffset, $
                          SCR_XSIZE=sz+sizeOffset, $
                          UNAME='right_value', $
                          VALUE=String(minn, FORMAT=format))
  endif

  ;; If sliderPos was passed in it overrides xLabelSize
  if (sliderPos ne 0) then begin
    valueSize = (wValue2 ne 0l) ? $
                (Widget_Info(wValue2, /GEOMETRY)).scr_xsize : 0l
    Widget_Control, wLabel, SCR_XSIZE=(sliderPos-valueSize) > 0
  endif

  ;; set up state structure
  state = {tlb:tlb, $
           cw_slider_id_tag:0b, $
           barbottom:barbottom, $
           clrs:clrs, $
           color1:color1, $
           color2:color2, $
           cur:Round((xsize-thumb_width)/2.), $
           down:0l, $
           downArrow:downArrow, $
           downArrowColour2:downArrowColour2, $
           drag:Byte(Keyword_Set(drag)), $
           dual:Byte(Keyword_Set(dual)), $
           editable:Byte(Keyword_Set(editable)), $
           followMouse:0b, $
           format:format, $
           hidebar:Byte(Keyword_Set(hidebar)), $
           jump:jump, $
           label:label, $
           labelSize:labelSize, $
           leftImageFlag:leftImageFlag, $
           leftImage:leftImage, $
           linked:0b, $
           max:maxx, $
           min:minn, $
           narrowthumb:Byte(Keyword_Set(narrowThumb)), $
           noupdate:Byte(Keyword_Set(noupdate)), $
           oldWinID:win, $
           overlap:Byte(Keyword_Set(overlap)), $
           pixmap:pixmap, $
           pos1:Round((xsize-thumb_width)/2.), $
           pos2:0l, $
           posindex1:0l, $
           posindex2:0l, $
           rightImageFlag:rightImageFlag, $
           rightImage:rightImage, $
           shiftdown:0b, $
           showticks:Byte(Keyword_Set(showTicks)), $
           showValue:showVal, $
           slidersize:slidersize, $
           thumb:0b, $
           thumbbottom:thumbbottom, $
           thumbwidth:thumb_width, $
           tickvals:tickvals, $
           tickpos:tickpos, $
           type:type, $
           upArrow:upArrow, $
           upArrowColour1:upArrowColour1, $
           useColor1:useColor1, $
           useColor2:useColor2, $
           value1:value1, $
           value2:value2, $
           valueSize:valueSize, $
           wDraw:wDraw, $
           wLabel:wLabel, $
           wLeftImage:wLeftImage, $
           wRightImage:wRightImage, $
           wValue1:wvalue1, $
           wValue2:wvalue2, $
           xsize:xsize, $
           ysize:ysize, $
           ythumbSize:ythumbSize $
          }
  pState = Ptr_New(state, /NO_COPY)
  ;; set uvalue of top base
  Widget_Control, Widget_Info(tlb, /CHILD), SET_UVALUE=pState, $
                  KILL_NOTIFY='cw_slider_kill'

  ;; "Hide" draw widget to prevent other DG calls from using it
  if (wDraw ne 0) then Xmanager, 'cw_slider', wDraw, /JUST_REG

  ;; the event func must be set after realization for it to stick
  Widget_Control, wDraw, EVENT_FUNC='cw_slider_event'
  if (wLeftImage ne 0l) then begin
    Xmanager, 'cw_slider', wLeftImage, /JUST_REG
    Widget_Control, wLeftImage, EVENT_FUNC='cw_slider_image_event'
  endif
  if (wRightImage ne 0l) then begin
    Xmanager, 'cw_slider', wRightImage, /JUST_REG
    Widget_Control, wRightImage, EVENT_FUNC='cw_slider_image_event'
  endif
  if ((*pState).editable && (*pState).showValue) then begin
    Widget_Control, wValue1, EVENT_FUNC='cw_slider_event'
    if (wValue2 ne 0l) then $
      Widget_Control, wValue2, EVENT_FUNC='cw_slider_event'
  endif

  ;; map widget
  if (map) then Widget_Control, tlb, /MAP
  ;; set window index to previous value
  Wset, win

  ;; If widget is being created in an already realized base the
  ;; notify_realize routine must be postponed until the state can be
  ;; cached.
  if (Widget_Info(tlb, /REALIZED)) then begin
    cw_slider_notify_realize, tlb
  endif else begin
    Widget_Control, tlb, NOTIFY_REALIZE='cw_slider_notify_realize'
  endelse

  return, tlb

end
