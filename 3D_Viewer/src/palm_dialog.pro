;--------------------------------------------------------------
;+
; A simple dialog
;
; @returns The top level base of the new dialog.
;
; @keyword TEXT {in}{type=string}{required}
;   The text of the dialog.
;
; @keyword TITLE {in}{type=string}{required}
;   The title for the dialog.
;-
;--------------------------------------------------------------
function PALM_Dialog, $
    TEXT=text, $
    TITLE=title

    tlb = widget_base( $
        /COL, $
        MAP=0, $
        TITLE=title, $
        XSIZE=200)
    wLabel = widget_label(tlb, VALUE=' ')
    for i = 0, n_elements(text)-1 do $
        wLabel = widget_label(tlb, $
            /ALIGN_CENTER, $
            VALUE=text[i])
    wLabel = widget_label(tlb, VALUE=' ')
    widget_control, tlb, /REALIZE
    ss = get_screen_size()
    geom = widget_info(tlb, /GEOM)
    widget_control, tlb, /MAP, $
        XOFFSET=(ss[0]-geom.scr_xSize)/2, $
        YOFFSET=(ss[1]-geom.scr_ySize)/2

    return, tlb

end
