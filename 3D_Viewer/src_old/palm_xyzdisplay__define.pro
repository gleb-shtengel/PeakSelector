;-------------------------------------------------------------------------------------
;+
; This method adds the images from the EM volume to the display
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::AddEMVolume, pVolume, $
    HIDE=hide, $
    OPACITY=opacity

    compile_opt strictarr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif
    if ~ptr_valid(pVolume) then begin
        message, /TRACEBACK, 'Invalid input for EM volume'
        return
    endif

    ptr_free, self.pVolumeEM
    dims = size(*self.pVolume, /DIMENSIONS)
    self->GetDisplayObject, $
        X_EM_IMAGE=oImageEmX, $
        Y_EM_IMAGE=oImageEmY, $
        Z_EM_IMAGE=oImageEmZ
    dims = [[dims[[2,1]]],[dims[[0,2]]],[dims[[0,1]]]]
    oImage = [oImageEmX,oImageEmY,oImageEmZ]
    for i = 0, n_elements(oImage)-1 do begin
        oImage[i]->SetProperty, $
            ALPHA_CHANNEL=opacity, $
            DIMENSIONS=dims[*,i], $
            HIDE=hide
    endfor
    self.pVolumeEM = pVolume
    self->UpdateImagesEM, /X, /Y, /Z
    self->RenderScene
    
end

;-------------------------------------------------------------------------------------
;+
; This method adds the images from the EM volume to the display
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::ClearEMVolume

    compile_opt strictarr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    self->GetDisplayObject, $
        X_EM_IMAGE=oImageX, $
        Y_EM_IMAGE=oImageY, $
        Z_EM_IMAGE=oImageZ
    oImageX->SetProperty, /HIDE
    oImageY->SetProperty, /HIDE
    oImageZ->SetProperty, /HIDE
    self->RenderScene

end

;-------------------------------------------------------------------------------------
;+
; This method toggles the axes on and off
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::AxisHide

    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    if ~ptr_valid(self.pVolume) then $
        return

    if n_elements(*self.pVolume) EQ 0 then $
        return

    self -> GetDisplayObject, Z_POLYLINE_HORIZONTAL=oPolyZHorizontal, $
                              Z_POLYLINE_VERTICAL=oPolyZVertical, $
                              Y_POLYLINE_HORIZONTAL=oPolyYHorizontal, $
                              Y_POLYLINE_VERTICAL=oPolyYVertical, $
                              X_POLYLINE_HORIZONTAL=oPolyXHorizontal, $
                              X_POLYLINE_VERTICAL=oPolyXVertical

    oPolyZHorizontal -> GetProperty, HIDE=hide
    hide = ~hide
    oPolyZHorizontal -> SetProperty, HIDE=hide
    oPolyZVertical -> SetProperty, HIDE=hide
    oPolyYHorizontal -> SetProperty, HIDE=hide
    oPolyYVertical -> SetProperty, HIDE=hide
    oPolyXHorizontal -> SetProperty, HIDE=hide
    oPolyXVertical -> SetProperty, HIDE=hide

    self -> RenderScene

end



;------------------------------------------------------------------------------
;+
; This lifecycle method is called during the destruction of the PALM_XYZDisplay
; object via OBJ_DESTROY.
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;------------------------------------------------------------------------------
pro PALM_XYZDisplay::Cleanup

    compile_opt StrictArr
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    ptr_free, self.oSelect
    obj_destroy, [self.oBuffer, self.oPalette, self.oPaletteEM]
    ptr_free, self.rvol
    ptr_free, self.gvol
    ptr_free, self.bvol
    ptr_free, self.pVolumeEM

end



;-------------------------------------------------------------------------------------
;+
; This method is clears the windows
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::Clear

    self -> GetDisplayObject, Z_IMAGE=oImageZ, $
                              Y_IMAGE=oImageY, $
                              X_IMAGE=oImageX

    if ~obj_valid(oImageZ) then $
        return

    oImageZ -> SetProperty, DATA=bytarr(2,2)
    oImageY -> SetProperty, DATA=bytarr(2,2)
    oImageX -> SetProperty, DATA=bytarr(2,2)

    self -> RenderScene

end


;-------------------------------------------------------------------------------------
;+
; This method exports the contents of a window to an image file
;
; @Parameter
;    windowID {in}{required}{type=string}
;      A string specifying the window from which the image will be captured. Acceptable
;      values are:
;        PALM_XWindow
;        PALM_YWindow
;        PALM_ZWindow
;
; @Keyword
;    BMP {in}{optional}{type=boolean}
;      Set this keyword to have the image written to a BMP file
;
; @Keyword
;    TIFF {in}{optional}{type=boolean}
;      Set this keyword to have the image written to a TIFF file
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::ExportImage, windowID, $
    BMP=BMP, $
    TIFF=TIFF

    compile_opt StrictArr
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    case windowID of
        'PALM_XWindow': index = 0
        'PALM_YWindow': index = 1
        'PALM_ZWindow': index = 2
        else: return
    endcase

; Get the image
    wait, 0.5 ; Pause to let the window clear
    image = self->GetImage(index)

    if keyword_set(BMP) then begin
        if self -> GetExportFile(EXTENSION='bmp', OUTPUT_FILE=file) then begin
            write_bmp, file, image, /RGB
        endif
    endif

    if keyword_set(TIFF) then begin
        if self -> GetExportFile(EXTENSION='tiff', OUTPUT_FILE=file) then begin
            write_tiff, file, reverse(image, 3)
        endif
    endif

end


;-------------------------------------------------------------------------------------
;+
; This method exports the contents of a window to a motion file
;
; @Parameter
;    windowID {in}{required}{type=string}
;      A string specifying the window from which the image will be captured. Acceptable
;      values are:
;        PALM_XWindow
;        PALM_YWindow
;        PALM_ZWindow
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::ExportMotion, windowID, $
    MPEG=mpeg

    compile_opt StrictArr
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        if obj_valid(oMPEG) then $
            obj_destroy, oMPEG
        help, /LAST
        return
    endif

    if ~(self->GetExportFile(EXTENSION='mpg', OUTPUT_FILE=file)) then $
        return

    widget_control, /HOURGLASS

    case windowID of
        'PALM_XWindow': index = 0
        'PALM_YWindow': index = 1
        'PALM_ZWindow': index = 2
        else: return
    endcase
    doX = index EQ 0
    doY = index EQ 1
    doZ = index EQ 2

    maxDots = 50
    addVal = self.volDims[index]/maxDots
    self.oMainGUI -> UpdateInformationLabel, 'Capturing MPEG Frames'

    oMPEG = obj_new('IDLgrMPEG', $
        FILENAME=file, $
        QUALITY=100)
    if keyword_set(mpeg) then begin
        indexCurrent = self.index[index]
        for i = 0L, self.VolDims[index]-1 do begin
            self.index[index] = i
            self -> UpdateImages, $
                INPUT_INDEX=i, $
                X=doX, Y=doY, Z=doZ
            self -> UpdateAxes, $
                X=doX, Y=doY, Z=doZ
            self -> RenderScene
            oMPEG -> Put, self->GetImage(index, REVERSE=3)
            if ~(i MOD addVal) then $
                self.oMainGUI -> UpdateInformationLabel, /APPEND, /REMOVE_PERCENT, $
                    '. <'+strtrim(100*i/self.VolDims[index],2)+'%>'
        endfor
        self.index[index] = indexCurrent
        self -> UpdateImages, X=doX, Y=doY, Z=doZ, $
            INPUT_INDEX=indexCurrent
        self -> UpdateAxes, X=doX, Y=doY, Z=doZ
        self -> RenderScene
    endif

    self.oMainGUI -> UpdateInformationLabel, 'Saving', /APPEND, /REMOVE_PERCENT
    oMPEG -> Save
    obj_destroy, oMPEG
    self.oMainGUI -> UpdateInformationLabel, ' '

    widget_control, HOURGLASS=0

end


;-------------------------------------------------------------------------------------
;+
; This method returns referneces to underlying graphics objects
;
; @keyword
;   X_IMAGE {out}{type=object reference}
;       Set this keyword to a named variable to retrieve a reference to
;       the X image object
;
; @keyword
;   X_POLYLINE_HORIZONTAL {out}{type=object reference}
;       Set this keyword to a named variable to retrieve a reference to
;       the horizontal polynomial in the X display
;
; @keyword
;   X_POLYLINE_VERTICAL {out}{type=object reference}
;       Set this keyword to a named variable to retrieve a reference to
;       the vertical polynomial in the X display
; @keyword
;   X_TEXT {out}{type=object reference}
;       Set this keyword to a named variable to retrieve a reference to
;       the text object in the X display
;
; @keyword
;   Y_IMAGE {out}{type=object reference}
;       Set this keyword to a named variable to retrieve a reference to
;       the Y image object
;
; @keyword
;   Y_POLYLINE_HORIZONTAL {out}{type=object reference}
;       Set this keyword to a named variable to retrieve a reference to
;       the horizontal polynomial in the Y display
;
; @keyword
;   Y_POLYLINE_VERTICAL {out}{type=object reference}
;       Set this keyword to a named variable to retrieve a reference to
;       the vertical polynomial in the Y display
;
; @keyword
;   Y_TEXT {out}{type=object reference}
;       Set this keyword to a named variable to retrieve a reference to
;       the text object in the Y display
;
; @keyword
;   Z_IMAGE {out}{type=object reference}
;       Set this keyword to a named variable to retrieve a reference to
;       the Z image object
;
; @keyword
;   Z_POLYLINE_HORIZONTAL {out}{type=object reference}
;       Set this keyword to a named variable to retrieve a reference to
;       the horizontal polynomial in the Z display
;
; @keyword
;   Z_POLYLINE_VERTICAL {out}{type=object reference}
;       Set this keyword to a named variable to retrieve a reference to
;       the vertical polynomial in the Z display
;
; @keyword
;   Z_TEXT {out}{type=object reference}
;       Set this keyword to a named variable to retrieve a reference to
;       the text object in the Z display
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::GetDisplayObject, $
    X_IMAGE=oImageX, $
    X_EM_IMAGE=oImageEmX, $
    X_POLYLINE_HORIZONTAL=oPolylineXHorizontal, $
    X_POLYLINE_VERTICAL=oPolylineXVertical, $
    X_TEXT=oTextX, $
    Y_IMAGE=oImageY, $
    Y_EM_IMAGE=oImageEmY, $
    Y_POLYLINE_HORIZONTAL=oPolylineYHorizontal, $
    Y_POLYLINE_VERTICAL=oPolylineYVertical, $
    Y_TEXT=oTextY, $
    Z_IMAGE=oImageZ, $
    Z_EM_IMAGE=oImageEmZ, $
    Z_POLYLINE_HORIZONTAL=oPolylineZHorizontal, $
    Z_POLYLINE_VERTICAL=oPolylineZVertical, $
    Z_TEXT=oTextZ

    compile_opt StrictArr
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    if ~obj_valid(self.oView[0])  then $
        return

; X
    if arg_present(oImageX) then $
        oImageX = self.oView[0] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL/IMAGE')
    if arg_present(oImageEmX) then $
        oImageEmX = self.oView[0] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL/IMAGE_EM')
    if arg_present(oPolylineXHorizontal) then $
        oPolylineXHorizontal = self.oView[0] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL/POLYLINE_X_HORIZONTAL')
    if arg_present(oPolylineXVertical) then $
        oPolylineXVertical = self.oView[0] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL/POLYLINE_X_VERTICAL')
    if arg_present(oTextX) then $
        oTextX = self.oView[0] -> GetByName('MODEL/TEXT')
; Y
    if arg_present(oImageY) then $
        oImageY = self.oView[1] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL/IMAGE')
    if arg_present(oImageEmY) then $
        oImageEmY = self.oView[1] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL/IMAGE_EM')
    if arg_present(oPolylineYHorizontal) then $
        oPolylineYHorizontal = self.oView[1] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL/POLYLINE_Y_HORIZONTAL')
    if arg_present(oPolylineYVertical) then $
        oPolylineYVertical = self.oView[1] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL/POLYLINE_Y_VERTICAL')
    if arg_present(oTextY) then $
        oTextY = self.oView[1] -> GetByName('MODEL/TEXT')
; Z
    if arg_present(oImageZ) then $
        oImageZ = self.oView[2] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL/IMAGE')
    if arg_present(oImageEmZ) then $
        oImageEmZ = self.oView[2] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL/IMAGE_EM')
    if arg_present(oPolylineZHorizontal) then $
        oPolylineZHorizontal = self.oView[2] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL/POLYLINE_Z_HORIZONTAL')
    if arg_present(oPolylineZVertical) then $
        oPolylineZVertical = self.oView[2] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL/POLYLINE_Z_VERTICAL')
    if arg_present(oTextZ) then $
        oTextZ = self.oView[2] -> GetByName('MODEL/TEXT')

end



;-------------------------------------------------------------------------------------
;+
; This method prompts the user for an output file
;
; @returns  True (1) if success, false (0) if user cancels
;
; @keyword EXTENSION {in}{type=string}{required}
;   The file extension to filter on
;
; @keyword OUTPUT_FILE {out}{type=string}{optional}
;   The output pathname selected.
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
function PALM_XYZDisplay::GetExportFile, $
    EXTENSION=extension, $
    OUTPUT_FILE=file

    compile_opt StrictArr
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return, 0
    endif

    file = dialog_pickfile( $
        DEFAULT_EXTENSION=extension, $
        FILTER='*.'+extension, $
        TITLE='Select an output file' $
        )
    if file EQ '' then $
        return, 0

    if file_test(file) then begin
        void = dialog_message(/QUESTION, $
            ['File already exists: '+file, $
             'Would you like to overwrite this file?'])
        if strupcase(void) EQ 'NO' then $
            return, 0
    endif

    return, 1

end


;-------------------------------------------------------------------------------------
;+
; This method captures an image from the specified window index
;
; @returns  The image currently displayed in the requested window.
;
; @param indexWindow {in}{type=number}{required}
;   X, Y, or Z window (0, 1, 2)
;
; @keyword REVERSE {in}{type=boolean}{optional}
;   If set, reverse the image
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
function PALM_XYZDisplay::GetImage, indexWindow, $
    REVERSE=reverse

    if ~obj_valid(self.oBuffer) then $
        self.oBuffer = obj_new('IDLgrBuffer')
    self.oWindow[indexWindow] -> GetProperty, DIMENSIONS=dims, $
        GRAPHICS_TREE=oView
    self.oBuffer -> SetProperty, DIMENSIONS=dims
    self.oBuffer -> Draw, oView
    oImage = self.oBuffer -> Read()
    oImage -> GetProperty, DATA=image
    obj_destroy, oImage

    if n_elements(reverse) GT 0 then $
        image = reverse(image, reverse[0])

    return, image

end



;-------------------------------------------------------------------------------------
;+
; This method is for accessing class properties
;
; @keyword COLOR_TABLE {out}{optional}
;   XYZ display color table
;
; @keyword Z_INDEX {out}{optional}
;   Current Z index displayed
;
; @keyword Z_LOCATION {out}{optional}
;   Z location 
;
; @keyword Z_SLICE {out}{optional}
;   Z slice
;
; @keyword Z_VPR {out}{optional}
;   Z VPR
;
; @keyword BYTESCALE_RANGE {out}{optional}
;   Min, max byte scale range
;
; @keyword WINDOW_DIMENSIONS {out}{optional}
;   Window dimensions
;
; @keyword X_COLOR {out}{optional}
;   X axis color
;
; @keyword Y_COLOR {out}{optional}
;   Y axis color
;
; @keyword Z_COLOR {out}{optional}
;   Z axis color
;
; @keyword BS_MAX {out}{optional}
;   Byte scale max
;
; @keyword BS_MIN {out}{optional}
;   Byte scale min
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::GetProperty, $
    COLOR_TABLE=ColorTable, $
    Z_INDEX=ZIndex, $
    Z_LOCATION=ZLocation, $
    Z_SLICE=ZSlice, $
    Z_VPR=ZVPR, $
    BYTESCALE_RANGE=BSRange, $
    WINDOW_DIMENSIONS=winDims, $
    X_COLOR=xColor, $
    Y_COLOR=yColor, $
    Z_COLOR=zColor, $
    BS_MAX=bsmax,  $
    BS_MIN=bsmin

    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    if arg_present(bsmax) then begin
        bsmax = self.BSRange[1]
    endif

    if arg_present(bsmin) then begin
        bsmin = self.BSRange[0]
    endif

    if arg_present(ColorTable) then begin
        self.oPalette -> GetProperty, BLUE=blue, $
            GREEN=green, $
            RED=red
        ColorTable = [[red],[green],[blue]]
    endif
; Z information
    if arg_present(ZIndex) then $
        ZIndex = self.index[2]
    getLocation = arg_present(ZLocation)
    getSlice = arg_present(ZSlice)
    getVPR = arg_present(ZVPR)
    if getLocation || getSlice || getVPR then $
        self -> GetDisplayObject, Z_IMAGE=oImage
    if getLocation then $
        oImage -> GetProperty, LOCATION=ZLocation
    if getSlice then $
        oImage -> GetProperty, DATA=ZSlice
    if getVPR then $
        self.oView[2] -> GetProperty, VIEWPLANE_RECT=ZVPR

    if arg_present(BSRange) then $
        BSRange = self.BSRange

    if arg_present(winDims) then $
        self.oView[2] -> GetProperty, DIMENSION=winDims
; Axes colors
    if arg_present(xColor) then $
        xColor = self.color[0,*]
    if arg_present(yColor) then $
        yColor = self.color[1,*]
    if arg_present(zColor) then $
        zColor = self.color[2,*]

end


;-------------------------------------------------------------------------------------
;+
; This method initializes the object.
;
;
; @returns
;   1 for success and 0 otherwise
;
; @keyword COLOR_TABLE {in}{required}
;   Display default color table.
;
; @keyword MAIN_GUI {in}{required}
;   Reference to the main GUI object.
;
; @keyword X_WINDOW {in}{required}
;   X graphics window.
;
; @keyword Y_WINDOW {in}{required}
;   Y graphics window.
;
; @keyword Z_WINDOW {in}{required}
;   Z graphics window.
;
; @keyword
;   _EXTRA {in}{optional}
;       Any extra keywords are passed to the inherited IDLgrScene object
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
function PALM_XYZDisplay::Init, $
    COLOR_TABLE=ColorTable, $
    MAIN_GUI = oMainGUI, $
    X_WINDOW=oWindowX, $
    Y_WINDOW=oWindowY, $
    Z_WINDOW=oWindowZ
    _EXTRA=_extra

    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return, 0
    endif


    if ~obj_valid(oWindowX) then $
        return, 0
    if ~obj_isa(oWindowX, 'IDLitWindow') then $
        return, 0
    if ~obj_valid(oWindowY) then $
        return, 0
    if ~obj_isa(oWindowY, 'IDLitWindow') then $
        return, 0
    if ~obj_valid(oWindowZ) then $
        return, 0
    if ~obj_isa(oWindowZ, 'IDLitWindow') then $
        return, 0

    self.oWindow = [oWindowX, oWindowY, oWindowZ]
    self.oMainGUI = obj_valid(oMainGUI) ? $
        (obj_isa(oMainGUI[0], 'PALM_MainGUI') ? oMainGUI[0] : obj_new()) : $
        obj_new()

    self.pVolume = ptr_new()
    self.oSelect = ptr_new(/ALLOCATE)

; Color settings
    self.AxisTransparency = 128B
    self.BSIncrement = 1
    self.BSRange = [0,255]
    self.BackgroundColor = [255B,255B,255B]
    self.color[0,*] = [255B,0B,0B]
    self.color[1,*] = [0B,255B,0B]
    self.color[2,*] = [0B,255B,255B]
    self.defaultWL = replicate(!values.d_nan, 2)

    if n_elements(wlCenter) GT 0 then $
        self -> SetProperty, DEFAULT_WL_CENTER=wlCenter

    if n_elements(wlWidth) GT 0 then $
        self -> SetProperty, DEFAULT_WL_WIDTH=wlWidth

    if ~self->InitializeDisplay() then $
        return, 0

    self -> SetProperty, COLOR_TABLE=ColorTable

    return, 1

end



;-------------------------------------------------------------------------------------
;+
; This method initializes the graphics oibjects used
;
;
; @returns
;   1 for success and 0 otherwise
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
function PALM_XYZDisplay::InitializeDisplay

    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return, 0
    endif

    self.oPalette = obj_new('IDLgrPalette')
    self.oPaletteEM = obj_new('IDLgrPalette')
    self.oPaletteEM->LoadCT, 0
    nameVars = ['X','Y','Z']
    for i = 0, n_elements(nameVars)-1 do begin
        self.oView[i] = obj_new('IDLgrView', $
            COLOR=self.BackgroundColor, $
            NAME=nameVars[i])
        oModel = obj_new('IDLgrModel', NAME='MODEL')
        self.oView[i] -> Add, oModel
        oTranslationModel = obj_new('IDLgrModel', $
            NAME='TRANSLATIONMODEL')
        oModel -> Add, oTranslationModel
        oScaleModel = obj_new('IDLgrModel', $
            NAME='SCALEMODEL')
        oTranslationModel -> Add, oScaleModel
        oImage = obj_new('IDLgrImage', NAME='IMAGE', $
            PALETTE=self.oPalette)
        oImageEM = obj_new('IDLgrImage', $
            ALPHA_CHANNEL=0.1, $
            BLEND_FUNCTION=[3,4], $
            NAME='IMAGE_EM', $
            PALETTE=self.oPaletteEM)
        oText = obj_new('IDLgrText', nameVars[i], $
            COLOR=reform(self.color[i,*]), $
            NAME='TEXT')
        oScaleModel -> Add, [oImage, oImageEM]
        oModel -> Add, oText
        oPolyline = obj_new('IDLgrPolyline', $ ; Z
            /HIDE, $
            NAME='POLYLINE_'+nameVars[i]+'_HORIZONTAL', $
            THICK=2, $
            VERT_COLOR=[[reform(self.color[((i LT 2)+1),*]),self.AxisTransparency], $
                        [reform(self.color[((i LT 2)+1),*]),self.AxisTransparency]])
        oScaleModel -> Add, oPolyline
        oPolyline = obj_new('IDLgrPolyline', $ ; Y
            /HIDE, $
            NAME='POLYLINE_'+nameVars[i]+'_VERTICAL', $
            THICK=2, $
            VERT_COLOR=[[reform(self.color[(i EQ 0),*]),self.AxisTransparency], $
                        [reform(self.color[(i EQ 0),*]),self.AxisTransparency]])
        oScaleModel -> Add, oPolyline

        self.oWindow[i] -> SetProperty, GRAPHICS_TREE=self.oView[i]

        ;  Add scale bars for the MPR views
        obj = obj_new('IDLgrPolyline', NAME='SCALE_'+nameVars[i]+'_X', THICK=2, $
                        COLOR=[255,255,255], ALPHA_CHANNEL=0.5)
        oModel->Add, obj
        obj = obj_new('IDLgrPolyline', NAME='SCALE_'+nameVars[i]+'_Y', THICK=2,  $
                        COLOR=[255,255,255], ALPHA_CHANNEL=0.5)
        oModel->Add, obj

        obj = obj_new('IDLgrText', NAME='TEXT_'+nameVars[i]+'_X',  $
                      COLOR=[255,255,255], ALPHA_CHANNEL=0.5)
        oModel->Add, obj
        obj = obj_new('IDLgrText', NAME='TEXT_'+nameVars[i]+'_Y',  $
                      COLOR=[255,255,255], ALPHA_CHANNEL=0.5)
        oModel->Add, obj
    endfor

    return, 1

end



;-------------------------------------------------------------------------------------
;+
; This method is called when moving either the slice in the current window or
; a polyline in a window
;
; @param oWindow {in}{type=window object}{required}
;   Window object for move.
;
; @param
;   XY {in}{type=integer}{required}
;       Set this keyword to a 2-element vector specifying a position in the
;       display
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Global Services Group
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::Move, oWindow, xy

    if n_elements(xy) LT 2 then $
        return
    if ~ptr_valid(self.pVolume) then $
        return
    if n_elements(*self.pVolume) EQ 0 then $
        return
    self -> UpdateVolumeForManip, /START, /NO_DRAW

    case n_elements(*self.oSelect) of
        0: self -> MoveSlice, oWindow, xy
        else: begin
            oWindow -> GetProperty, GRAPHICS_TREE=oView
            oModel = oView -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL')
            void = oWindow -> PickData(oView, oModel, xy, dataXY)
            self -> MoveAxis, dataXY
        end
    endcase

end



;-------------------------------------------------------------------------------------
;+
; This method updates the volume indices according to the input [x,y] value
;
;
; @param
;   XY {in}{type=integer}{required}
;       Set this keyword to a 2-element vector specifying a position in the
;       display
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::MoveAxis, xy

    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    if n_elements(xy) LT 2 then $
        return
    if ~ptr_valid(self.pVolume) then $
        return
    if n_elements(*self.pVolume) EQ 0 then $
        return

    if (xy[0] EQ -1) || (xy[1] EQ -1) then begin
    ;
    ; We've left the window...Clear the selected polylines
    ;
        ptr_free, self.oSelect
        self.oSelect = ptr_new(/ALLOCATE)
        return
    endif

    for i = 0, n_elements(*self.oSelect)-1 do begin

        (*self.oSelect)[i] -> GetProperty, DATA=verts, $
            NAME=polyName
        splitName = strtok(polyName, '_', /EXTRACT)
        case splitName[1] of
            'X': begin
                self -> GetDisplayObject, X_IMAGE=oImageX
                oImageX -> GetProperty, LOCATION=loc
                if splitName[2] EQ 'HORIZONTAL' then begin
                ; Z
                    self.index[2] = ((xy[0]-loc[0])>0)<(self.volDims[2]-1)
                    self -> UpdateAxes, /Z
                    self -> UpdateImages, /Z
                endif else begin
                ; Y
                    self.index[1] = ((xy[1]-loc[1])>0)<(self.volDims[1]-1)
                    self -> UpdateAxes, /Y
                    self -> UpdateImages, /Y
                endelse
            end
            'Y': begin
                self -> GetDisplayObject, Y_IMAGE=oImageY
                oImageY -> GetProperty, LOCATION=loc
                if splitName[2] EQ 'HORIZONTAL' then begin
                ; Z
                    self.index[2] = ((xy[1]-loc[1])>0)<(self.volDims[2]-1)
                    self -> UpdateAxes, /Z
                    self -> UpdateImages, /Z
                endif else begin
                ; X
                    self.index[0] = ((xy[0]-loc[0])>0)<(self.volDims[0]-1)
                    self -> UpdateAxes, /X
                    self -> UpdateImages, /X
                endelse
            end
            'Z': begin
                self -> GetDisplayObject, Z_IMAGE=oImageZ
                oImageZ -> GetProperty, LOCATION=loc
                if splitName[2] EQ 'HORIZONTAL' then begin
                ; Y
                    self.index[1] = ((xy[1]-loc[1])>0)<(self.volDims[1]-1)
                    self -> UpdateAxes, /Y
                    self -> UpdateImages, /Y
                endif else begin
                ; X
                    self.index[0] = ((xy[0]-loc[0])>0)<(self.volDims[0]-1)
                    self -> UpdateAxes, /X
                    self -> UpdateImages, /X
                endelse
            end
        endcase

    endfor

    self -> Update3DDisplay
    self -> RenderScene

end



;-------------------------------------------------------------------------------------
;+
; This method updates the slice index for the current view
;
; @param oWindow {in}{type=window object}{required}
;   The window for the slice display.
;
; @param
;   XY {in}{type=integer}{required}
;       Set this keyword to a 2-element vector specifying a position in the
;       display;
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Global Services Group
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::MoveSlice, oWindow, xy

    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    if n_elements(xy) LT 2 then $
        return
    if ~ptr_valid(self.pVolume) then $
        return
    if n_elements(*self.pVolume) EQ 0 then $
        return

    oWindow -> GetProperty, GRAPHICS_TREE=oView
    oView -> GetProperty, NAME=ViewName

    diff = self.xy[1] - xy[1]
    inc = diff GT 0 ? -1 : (diff EQ 0 ? 0 : 1)

    case ViewName of
        'X': begin
            self.index[0] = ((self.index[0]+inc)>0)<(self.volDims[0]-1)
            self -> UpdateAxes, /X
            self -> UpdateImages, /X
        end
        'Y': begin
            self.index[1] = ((self.index[1]+inc)>0)<(self.volDims[1]-1)
            self -> UpdateAxes, /Y
            self -> UpdateImages, /Y
        end
        'Z': begin
            self.index[2] = ((self.index[2]+inc)>0)<(self.volDims[2]-1)
            self -> UpdateAxes, /Z
            self -> UpdateImages, /Z
        end
        else: return
    endcase

    self -> Update3DDisplay
    self -> RenderScene
    self.xy = fix(xy[0:1])

end


;-------------------------------------------------------------------------------------
;+
; This method updates the slice index for the current view
;
; @param oWindow {in}{type=window object}{required}
;   The window to pan in.
;
; @param
;   XY {in}{type=integer}{required}
;       Set this keyword to a 2-element vector specifying a position in the
;       display;
;
; @keyword INITIAL_POSITION {in}{type=boolean}{required}
;   If set, this is the initial click for the pan.
;
; @keyword RELEASE {in}{type=boolean}{optional}
;   Unused.
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::Pan, oWindow, xy, $
    INITIAL_POSITION=InitialPosition, $
    RELEASE=doRelease

    if n_elements(xy) LT 2 then $
        return
    if ~ptr_valid(self.pVolume) then $
        return
    if n_elements(*self.pVolume) EQ 0 then $
        return
    if keyword_set(InitialPosition) then begin
       self.DragStart = xy
    endif
    if (array_equal(self.DragStart, [-1, -1])) then $
        return

    oWindow -> GetProperty, GRAPHICS_TREE=oView
    oView->GetProperty, NAME=name
    oModel = oView -> GetByName('MODEL/TRANSLATIONMODEL')
    valid = oWindow -> PickData(oView, oModel, self.DragStart, dataStartXY)
    if (valid eq -1) then begin
          return
    endif
    valid = oWindow -> PickData(oView, oModel, xy, dataEndXY)
    if (valid eq -1) then begin
        return
    endif
    delta = dataStartXY - dataEndXY
    oModel->Translate, -delta[0], -delta[1] , 0

    case name of
        'X': begin
            oModel = self.oView[2]->GetByName('MODEL/TRANSLATIONMODEL')
            oModel->Translate, 0, -delta[1], 0
        end
        'Y': begin
            oModel = self.oView[2]->GetByName('MODEL/TRANSLATIONMODEL')
            oModel->Translate, -delta[0], 0, 0
        end
        'Z': begin
            oModel = self.oView[0]->GetByName('MODEL/TRANSLATIONMODEL')
            oModel->Translate, 0, -delta[1], 0
            oModel = self.oView[1]->GetByName('MODEL/TRANSLATIONMODEL')
            oModel->Translate, -delta[0], 0, 0
        end
        else:
    endcase

;
; Store the current position as the start point for the next drag event
; when translating.
;
    self.DragStart = xy
    self->RenderScene

end


;-------------------------------------------------------------------------------------
;+
; This method resets the window level, pan and zoom
;
; @keyword NO_DRAW {in}{type=boolean}{optional}
;   If set, do not update graphics
;
; @keyword NO_WINDOWLEVEL {in}{type=boolean}{optional}
;   If set, do not apply window level
;
; @keyword NO_X {in}{type=boolean}{optional}
;   If set, do no update X
;
; @keyword NO_Y {in}{type=boolean}{optional}
;   If set, do no update Y
;
; @keyword NO_Z {in}{type=boolean}{optional}
;   If set, do no update Z
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::Reset, $
    NO_DRAW=noDraw, $
    NO_WINDOWLEVEL=noWindowLevel, $
    NO_X=noX, $
    NO_Y=noY, $
    NO_Z=noZ
;
; Window level
;
    doWindowLevel = ~keyword_set(noWindowLevel)
    if doWindowLevel then begin
        index = where(~finite(self.defaultWL), count)
        if count EQ 0 then $
            self.BSRange = self.defaultWL[0] + self.defaultWL[1]/[-2,2] $
        else begin
            DataRange = [min(*self.pVolume, MAX=maxVal), maxVal]
            self.BSRange = DataRange
        endelse
    endif
; Scale and translation
    doReset = ~[keyword_set(noX), keyword_set(noY), keyword_set(noZ)]
    for i = 0, 2 do begin
        if doReset[i] then begin
            (self.oView[i]->GetByName('MODEL/TRANSLATIONMODEL')) -> Reset
            (self.oView[i]->GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL')) -> Reset
        endif
    endfor

    self -> UpdateImages, ALL=doWindowLevel, X=doReset[0], Y=doReset[1], Z=doReset[2]
    self->SetViews, NO_DRAW=noDraw

end



;-------------------------------------------------------------------------------------
;+
; This method renders the current scene
;
; @keyword NO_X {in}{type=boolean}{optional}
;   If set, do not update X
;
; @keyword NO_Y {in}{type=boolean}{optional}
;   If set, do not update Y
;
; @keyword NO_Z {in}{type=boolean}{optional}
;   If set, do not update Z
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::RenderScene, $
    NO_X=noX, $
    NO_Y=noY, $
    NO_Z=noZ

    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    doUpdate = ~[keyword_set(noX), keyword_set(noY), keyword_set(noZ)]
    for i = 0, n_elements(self.oWindow)-1 do $
        if doUpdate[i] then $
            self.oWindow[i] -> Draw

end



;-------------------------------------------------------------------------------------
;+
; This methods sets class properties
;
; @keyword BACKGROUND_COLOR {in}{type=RGB vector}{optional}
;   Set the window background RGB value
;
; @keyword COLOR_TABLE {in}{type=256x3 matrix}{optional}
;   The color table for display
;
; @keyword HIDE {in}{type=boolean}{optional}
;   If set, hide the displays
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::SetProperty, $
    BACKGROUND_COLOR=BackgroundColor, $
    COLOR_TABLE=Colortable, $
    EM_COLOR_TABLE=emColorTable, $
    EM_HIDE=hideEM, $
    EM_OPACITY=emOpacity, $
    HIDE=hide, $
    MOLECULE_HIDE=hideMol, $
    _EXTRA=_extra

    compile_opt idl2
    on_error, 2

    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    if n_elements(BackgroundColor) GT 2 then begin
        self.oView[0] -> SetProperty, COLOR=BackgroundColor[0:2]
        self.oView[1] -> SetProperty, COLOR=BackgroundColor[0:2]
        self.oView[2] -> SetProperty, COLOR=BackgroundColor[0:2]
    endif

    if n_elements(ColorTable) EQ 768 then begin
        self.oPalette -> SetProperty, RED=ColorTable[*,0], $
                                      GREEN=ColorTable[*,1], $
                                      BLUE=ColorTable[*,2]

    endif
    if n_elements(emColorTable) EQ 768 then begin
        self.oPaletteEM->SetProperty, $
            RED=emColorTable[*,0], $
            GREEN=emColorTable[*,1], $
            BLUE=emColorTable[*,2]
    endif
    if n_elements(emOpacity) GT 0 then begin
        self->GetDisplayObject, $
            X_EM_IMAGE=oImageEmX, $
            Y_EM_IMAGE=oImageEmY, $
            Z_EM_IMAGE=oImageEmZ
        oImageEmX->SetProperty, ALPHA_CHANNEL=emOpacity
        oImageEmY->SetProperty, ALPHA_CHANNEL=emOpacity
        oImageEmZ->SetProperty, ALPHA_CHANNEL=emOpacity
    endif
    
    if n_elements(hide) GT 0 then begin
        for i = 0, n_elements(self.oView)-1 do begin
            (self.oView[i]->GetByName('MODEL')) -> SetProperty, HIDE=hide
        endfor
    endif

    if n_elements(hideEM) GT 0 then begin
        self->GetDisplayObject, $
            X_EM_IMAGE=oImageEmX, $
            Y_EM_IMAGE=oImageEmY, $
            Z_EM_IMAGE=oImageEmZ
        oImageEmX->SetProperty, HIDE=hideEM
        oImageEmY->SetProperty, HIDE=hideEM
        oImageEmZ->SetProperty, HIDE=hideEM
    endif

    if n_elements(hideMol) GT 0 then begin
        self->GetDisplayObject, $
            X_IMAGE=oImageX, $
            Y_IMAGE=oImageY, $
            Z_IMAGE=oImageZ
        oImageX->SetProperty, HIDE=hideMol
        oImageY->SetProperty, HIDE=hideMol
        oImageZ->SetProperty, HIDE=hideMol
    endif

end



;-------------------------------------------------------------------------------------
;+
; This object sets the currently selected polyline objects.
;
; @param
;   oWindow {in}{type=window object}{required}
;       The window to operate on.
;
; @param
;   XY {in}{type=integer}{required}
;       Set this keyword to a 2-element vector specifying a position in the
;       display
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::SetSelect, oWindow, xy

    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    if ~ptr_valid(self.pVolume) then $
        return
    if n_elements(*self.pVolume) EQ 0 then $
        return
    if n_elements(xy) LT 2 then $
        return

    self.xy = fix(xy[0:1])
    ptr_free, self.oSelect
    self.oSelect = ptr_new(/ALLOCATE)
    oWindow -> GetProperty, GRAPHICS_TREE=oView
    oPoly = oWindow -> Select(oView, xy, DIMENSION=[10,10])
    if ~obj_valid(oPoly[0]) then $
        return

    for i = 0, n_elements(oPoly)-1 do $
        if obj_isa(oPoly[i], 'IDLgrPolyline') then begin
            oPoly[i] -> GetProperty, NAME=polyName
            if polyName NE 'BORDER' then $
                *self.oSelect = n_elements(*self.oSelect) GT 0 ? [*self.oSelect, oPoly[i]] : [oPoly[i]]
        endif

end


;-------------------------------------------------------------------------------------
;+
; This method sets up the viewplanes and axes
;
; @keyword ADJUST_SLIDERS {in}{optional}{type=boolean}
;   Slider adjustment
;
; @keyword NO_DRAW {in}{optional}{type=boolean}
;   If set, no graphics update
;   
; @keyword NO_LOCATION_ADJUST {in}{optional}{type=boolean}
;   If set the image location will not adjust
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::SetViews, $
    ADJUST_SLIDERS=AdjustSliders, $
    NO_DRAW=noDraw, $
    NO_LOCATION_ADJUST=noLocationAdjust

    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    limits = self.limits
;
; Get the window dimensions and aspect ratios
;
    nDims = n_elements(self.oView)
    winDims = fltarr(nDims,2)
    winAspect = fltarr(nDims,2)
    for i = 0, nDims-1 do begin
        self.oWindow[i]->GetProperty, DIMENSIONS=dims
        winDims[i,*] = dims
        winAspect[i,*] = dims/min(dims)
    endfor
    dimMin = min(self.volDims, indexMinDim)
    vpr = fltarr(nDims,4)
    vpr[2,*] = [0,0,limits[[0,1]]*winAspect[2,*]]
;
; Calculate the YZ-plane's viewplane
;
    scale = [ $
        winAspect[2,1] * (winAspect[0,0]/winAspect[0,1]), $
        winAspect[2,1] $
        ]
    vpr[0,*] = [0,0,limits[[2,1]]*scale]
;
; Calculate the XZ-plane's viewplane
;
    scale = [ $
        winAspect[2,0], $
        winAspect[2,0] * (winAspect[1,1]/winAspect[1,0]) $
        ]
    vpr[1,*] = [0,0,limits[[0,2]]*scale]

    if indexMinDim NE 2 then begin
        oModel = self.oView[0]->GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL')
        ctm = oModel->GetCTM()
    ;
    ; If we have not zoomed in then adjust the viewplanes
    ;
        if (ctm[0,0] LE 1.0) then begin
            scale = 1
        ;
        ; Check YZ-plane
        ;
            locX = reform((vpr[0,[2,3]] - self.volDims[[2,1]])/2)
            if (locX[0] LT vpr[0,0]) then begin
                diff = 2*abs(vpr[0,0]-locX[0])
                scale = (vpr[0,2]+diff)/vpr[0,2]
            endif
        ;
        ; Check XZ-plane
        ;
            locY = reform((vpr[1,[2,3]] - self.volDims[[0,2]])/2)
            if locY[1] LT vpr[1,1] then begin
                diff = 2*abs(vpr[1,1]-locY[1])
                scale = scale*(vpr[1,3]+diff)/vpr[1,3]
            endif
        ;
        ; Recalculate XY and YZ planes
        ;
            vpr[0:2,2:3] *= scale
        endif
    endif
;
; Update the images/text
;
    for i = 0, nDims-1 do begin
        case[i] of
            0: begin
                vDims = self.volDims[[2,1]]
                nameX = 'MODEL/TEXT_X_X'
                nameY = 'MODEL/TEXT_X_Y'
                self -> GetDisplayObject, X_EM_IMAGE=oImageEM, $
                                          X_IMAGE=oImage, $
                                          X_POLYLINE_HORIZONTAL=oPolyHorizontal, $
                                          X_POLYLINE_VERTICAL=oPolyVertical, $
                                          X_TEXT=oText
                method = 'UpdateYZScaleBar'
            end
            1: begin
                vDims = self.volDims[[0,2]]
                nameX = 'MODEL/TEXT_Y_X'
                nameY = 'MODEL/TEXT_Y_Y'
                self -> GetDisplayObject, Y_EM_IMAGE=oImageEM, $
                                          Y_IMAGE=oImage, $
                                          Y_POLYLINE_HORIZONTAL=oPolyHorizontal, $
                                          Y_POLYLINE_VERTICAL=oPolyVertical, $
                                          Y_TEXT=oText
                method = 'UpdateXZScaleBar'
            end
            2: begin
                vDims = self.volDims[[0,1]]
                nameX = 'MODEL/TEXT_Z_X'
                nameY = 'MODEL/TEXT_Z_Y'
                self -> GetDisplayObject, Z_EM_IMAGE=oImageEM, $
                                          Z_IMAGE=oImage, $
                                          Z_POLYLINE_HORIZONTAL=oPolyHorizontal, $
                                          Z_POLYLINE_VERTICAL=oPolyVertical, $
                                          Z_TEXT=oText
                method = 'UpdateXYScaleBar'
            end
        endcase
        if ~keyword_set(noLocationAdjust) then begin
        ;
        ; Position the image
        ;
            ctm = (self.oView[i]->GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL'))->GetCTM()
            ctmWindow = (self.oView[i]->GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL'))->GetCTM(DESTINATION=self.oWindow[i])
            location = -(vpr[i,[2,3]]-vDims)/2*ctmWindow[3,[0,1]]/ctm[[0,1],[0,1]]
            oImage -> SetProperty, LOCATION=location
            oImageEM->SetProperty, LOCATION=location
        endif

        self.oView[i] -> SetProperty, VIEWPLANE_RECT=vpr[i,*]
    ; Size the text
        oText -> SetProperty, CHAR_DIMENSIONS=vpr[i,2:3]/winDims[i,*]*15
    ; Size of the axes labels
        oText = self.oView[i]->GetByName(nameX)
        oText->SetProperty, CHAR_DIMENSIONS=vpr[i,2:3]/winDims[i,*]*15
        oText = self.oView[i]->GetByName(nameY)
        oText->SetProperty, CHAR_DIMENSIONS=vpr[i,2:3]/winDims[i,*]*15
    ;
    ; Reset the polylines
    ;
        if i EQ 0 then begin
            oPolyVertical -> SetProperty, DATA= $
                [[0,0],[vpr[i,2],0]]
            oPolyHorizontal -> SetProperty, DATA= $
                [[0,0],[0,vpr[i,3]]]
        endif else begin
            oPolyVertical -> SetProperty, DATA= $
                [[0,0],[0,vpr[i,3]]]
            oPolyHorizontal -> SetProperty, DATA= $
                [[0,0],[vpr[i,2],0]]
        endelse

        call_method, method, self, vDims

    endfor

    self -> UpdateAxes, /ALL, APPLY_CTM=AdjustSliders
    if ~keyword_set(noDraw) then $
        self->RenderScene

end

;-------------------------------------------------------------------------------------
;+
; This method updates the 3D display according to the current slice positions
;
; @keyword NO_DRAW {in}{type=boolean}{optional}
;   If set, do not update graphics
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::Update3DDisplay, $
    NO_DRAW=noDraw

    if ~obj_valid(self.oMainGUI) then $
        return

    if ~obj_isa(self.oMainGUI, 'PALM_MainGUI') then $
        return

    self.oMainGUI -> GetProperty, DISPLAY_3D=o3DDisplay
    if ~obj_valid(o3DDisplay) then $
        return

    o3DDisplay -> GetProperty, $
        DRAG_QUALITY=DragQuality, $
        HIDE_SLICE_LINES=HideSliceLines, $
        HIDE_SLICE_PLANES=HideSlicePlanes

    o3DDisplay -> UpdateSlicePlanes, $
        X_LOCATION = self.index[0], $
        Y_LOCATION = self.index[1], $
        Z_LOCATION = self.index[2], $
        NO_RENDER=((HideSliceLines AND HideSlicePlanes) OR (DragQuality EQ 'MEDIUM') OR (keyword_set(noDraw)))

end



;-------------------------------------------------------------------------------------
;+
; This method updates the polylines in the display
;
;
; @keyword
;   ALL {in}{type=boolean}{optional}
;       Set this keyword to have all of the polylines updated
;
; @keyword
;   Z
;       Set this keyword to have polylines pertaining to the Z slice updated
;
; @keyword
;   Y
;       Set this keyword to have polylines pertaining to the Y slice updated
;
; @keyword
;   X
;       Set this keyword to have polylines pertaining to the X slice updated
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::UpdateAxes, $
    ALL=doAll, $
    APPLY_CTM=ApplyCTM, $
    Z=doZ, $
    Y=doY, $
    X=doX

    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    if ~ptr_valid(self.pVolume) then $
        return
    if n_elements(*self.pVolume) EQ 0 then $
        return
    if keyword_set(doAll) then $
        doZ = (doY = (doX = 1))
    ApplyCTM = keyword_set(ApplyCTM)

    self -> GetDisplayObject, Z_IMAGE = oImageZ, $
                              Z_POLYLINE_HORIZONTAL=oPolyZHorizontal, $
                              Z_POLYLINE_VERTICAL=oPolyZVertical, $
                              Y_IMAGE=oImageY, $
                              Y_POLYLINE_HORIZONTAL=oPolyYHorizontal, $
                              Y_POLYLINE_VERTICAL=oPolyYVertical, $
                              X_IMAGE=oImageX, $
                              X_POLYLINE_HORIZONTAL=oPolyXHorizontal, $
                              X_POLYLINE_VERTICAL=oPolyXVertical

    self.oView[1] -> GetProperty, VIEWPLANE_RECT=vprY

; Get the image dimensions and location
    oImageZ -> GetProperty, DIMENSIONS=dimZ, $
                            LOCATION = locZ
    oImageY -> GetProperty, DIMENSIONS=dimY, $
                            LOCATION=locY
    oImageX -> GetProperty, DIMENSIONS=dimX, $
                            LOCATION=locX

    if keyword_set(doX) then begin
    ;
    ; Z vertical
    ;
        oPolyZVertical -> GetProperty, DATA=verts
        verts[0,*] = locZ[0] + self.index[0]
        if ApplyCTM then begin
            ctm = (self.oView[2]->GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL'))->GetCTM()
            verts[1,*] = (verts[1,*]-ctm[3,1])/ctm[1,1]
        endif
        oPolyZVertical -> SetProperty, DATA=verts
    ;
    ; Y vertical
    ;
        oPolyYVertical -> GetProperty, DATA=verts
        verts[0,*] = locY[0] + self.index[0]
        if ApplyCTM then begin
            ctm = (self.oView[1]->GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL'))->GetCTM()
            verts[1,*] = (verts[1,*]-ctm[3,1])/ctm[1,1]
        endif
        oPolyYVertical -> SetProperty, DATA=verts
    endif

    if keyword_set(doY) then begin
    ;
    ; Z horizontal
    ;
        oPolyZHorizontal -> GetProperty, DATA=verts
        If ApplyCTM then begin
            ctm = (self.oView[2]->GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL'))->GetCTM()
            verts[0,*] = (verts[0,*]-ctm[3,0])/ctm[0,0]
        endif
        verts[1,*] = locZ[1] + self.index[1]
        oPolyZHorizontal -> SetProperty, DATA=verts
    ;
    ; X vertical
    ;
        oPolyXVertical -> GetProperty, DATA=verts
        if ApplyCTM then begin
            ctm = (self.oView[0]->GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL'))->GetCTM()
            verts[0,*] = (verts[0,*]-ctm[3,0])/ctm[0,0]
        endif
        verts[1,*] = locX[1] + self.index[1]
        oPolyXVertical -> SetProperty, DATA=verts
    endif

    if keyword_set(doZ) then begin
    ;
    ; Y horizontal
    ;
        oPolyYHorizontal -> GetProperty, DATA=verts
        if ApplyCTM then begin
            ctm = (self.oView[1]->GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL'))->GetCTM()
            verts[0,*] = (verts[0,*]-ctm[3,0])/ctm[0,0]
        endif
        verts[1,*] = locY[1] + self.index[2]
        oPolyYHorizontal -> SetProperty, DATA=verts
    ;
    ; X horizontal
    ;
        oPolyXHorizontal -> GetProperty, DATA=verts
        verts[0,*] = locX[0] + self.index[2]
        if ApplyCTM then begin
            ctm = (self.oView[0]->GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL'))->GetCTM()
            verts[1,*] = (verts[1,*]-ctm[3,1])/ctm[1,1]
        endif
        oPolyXHorizontal -> SetProperty, DATA=verts
    endif

end


;-------------------------------------------------------------------------------------
;+
;  Create a hue image for the XY plane
;
; @param oImage {in}{type=image object}{required}
;   The image object
;
; @param volDims {in}{type=vector}{required}
;   Volume dimensions
;
; @param index {in}{type=integer}{required}
;   Index along the Z axis
;-
pro PALM_XYZDisplay::XYHueImage, oImage, volDims, index
    compile_opt idl2

    if ~obj_valid(oImage) then $
        self->GetDisplayObject, Z_IMAGE=oImage
    if n_elements(volDims) EQ 0 then $
        volDims = size(*self.pVolume, /DIMENSION)
    if n_elements(index) EQ 0 then $
        index = self.index[2]

    ;  Use the volume as is
    img = bytarr(3,volDims[0],volDims[1])
    img[0,*,*] = reform((*self.rvol)[*,*,index])
    img[1,*,*] = reform((*self.gvol)[*,*,index])
    img[2,*,*] = reform((*self.bvol)[*,*,index])
    oImage->SetProperty, DATA=img
end


;-------------------------------------------------------------------------------------
;+
;  Create a hue image for the YZ plane
;
; @param oImage {in}{type=image object}{required}
;   The image object
;
; @param volDims {in}{type=vector}{required}
;   Volume dimensions
;
; @param index {in}{type=integer}{required}
;   Index along the X axis
;-
pro PALM_XYZDisplay::YZHueImage, oImage, volDims, index
    compile_opt idl2

    if ~obj_valid(oImage) then $
        self->GetDisplayObject, X_IMAGE=oImage
    if n_elements(volDims) EQ 0 then $
        volDims = size(*self.pVolume, /DIMENSION)
    if n_elements(index) EQ 0 then $
        index = self.index[0]

    ;  Extract the proper plane from the RGB volume data
    img = bytarr(3,volDims[2],volDims[1])
    im = transpose(reform((*self.rvol)[index,*,*]))
;    im[indgen(volDims[2]),*] = im[reverse(indgen(volDims[2])),*]
    img[0,*,*] = im
    im = transpose(reform((*self.gvol)[index,*,*]))
;    im[indgen(volDims[2]),*] = im[reverse(indgen(volDims[2])),*]
    img[1,*,*] = im
    im = transpose(reform((*self.bvol)[index,*,*]))
;    im[indgen(volDims[2]),*] = im[reverse(indgen(volDims[2])),*]
    img[2,*,*] = im
    oImage->SetProperty, DATA=img
end


;-------------------------------------------------------------------------------------
;+
;  Create a hue image for the XZ plane
;
; @param oImage {in}{type=image object}{required}
;   The image object
;
; @param volDims {in}{type=vector}{required}
;   Volume dimensions
;
; @param index {in}{type=integer}{required}
;   Index along the Y axis
;-
pro PALM_XYZDisplay::XZHueImage, oImage, volDims, index
    compile_opt idl2

    if ~obj_valid(oImage) then $
        self->GetDisplayObject, Y_IMAGE=oImage
    if n_elements(volDims) EQ 0 then $
        volDims = size(*self.pVolume, /DIMENSION)
    if n_elements(index) EQ 0 then $
        index = self.index[1]

    ;  Extract the proper plane from the RGB volume data
    img = bytarr(3,volDims[0],volDims[2])
    img[0,*,*] = reform((*self.rvol)[*,index,*])
    img[1,*,*] = reform((*self.gvol)[*,index,*])
    img[2,*,*] = reform((*self.bvol)[*,index,*])
    oImage->SetProperty, DATA=img
end


;-------------------------------------------------------------------------------------
;+
;  Update the XY scale bar
;
; @param imgdim {in}{type=vector}{required}
;   Image dimensions
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::UpdateXYScaleBar, imgdim
    compile_opt idl2, logical_predicate

    self.oMainGUI->GetProperty, DISPLAY_3D=o3DDisplay
    oModel = o3DDisplay->GetObjectByName('PALM')
    oModel->GetProperty, X_RANGE=xr, Y_RANGE=yr
    oXY = self.oView[2]->GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL')
    oXY->GetProperty, TRANSFORM=tr
    self.oView[2]->GetProperty, VIEWPLANE_RECT=vr

    lengths = [10, 20, 30, 50, 100, 200, 300, 400, 500, 1000] / 2

    d = xr[1] - xr[0]                ;  volume range in X (nm)
    void = min(abs(lengths-d), idx)
    ddx = lengths[idx]
    dx = imgdim[0] / d               ;  pixels / nm
    sx = ddx*dx*tr[0,0]              ;  scale bar length (nm) to pixels and X zoom factor

    d = yr[1] - yr[0]                ;  volume range in Y (nm)
    void = min(abs(lengths-d), idx)
    ddy = lengths[idx]
    dy = imgdim[1] / d               ;  pixels / nm
    sy = ddy*dy*tr[1,1]              ;  scale bar length (nm) to pixels and Y zoom factor

    ;  Draw the scale bar
    oBar = self.oView[2]->GetByName('MODEL/SCALE_Z_X')
    oBar->SetProperty, DATA=[[vr[2]-sx-10,20,0],[vr[2]-10,20,0]]
    oBar = self.oView[2]->GetByName('MODEL/SCALE_Z_Y')
    oBar->SetProperty, DATA=[[vr[2]-10,20,0],[vr[2]-10,20+sy,0]]

    ;  Add the text label
    oText = self.oView[2]->GetByName('MODEL/TEXT_Z_X')
    oText->GetProperty, CHAR_DIMENSIONS=cdim
    s = strtrim(ddx,2)+' nm'
    wd = strlen(s)*cdim[0] - 20*(400.0/vr[2])
    oText->SetProperty, STRING=s, LOCATION=[vr[2]-sx-wd,15,0]
    oText = self.oView[2]->GetByName('MODEL/TEXT_Z_Y')
    oText->GetProperty, CHAR_DIMENSIONS=cdim
    s = strtrim(ddy,2)+' nm'
    wd = strlen(s)*cdim[0] - 20*(400.0/vr[2])
    oText->SetProperty, STRING=s, LOCATION=[vr[2]-wd,23+sy,0]
end


;-------------------------------------------------------------------------------------
;+
;  Update the YZ scale bar
;
; @param imgdim {in}{type=vector}{required}
;   Image dimensions
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::UpdateYZScaleBar, imgdim
    compile_opt idl2, logical_predicate

    self.oMainGUI->GetProperty, DISPLAY_3D=o3DDisplay
    oModel = o3DDisplay->GetObjectByName('PALM')
    oModel->GetProperty, Y_RANGE=yr, Z_RANGE=zr
    oXY = self.oView[0]->GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL')
    oXY->GetProperty, TRANSFORM=tr
    self.oView[0]->GetProperty, VIEWPLANE_RECT=vr
    lengths = [10, 20, 30, 50, 100, 200, 300, 400, 500, 1000] / 2

    d = yr[1] - yr[0]                ;  volume range in Y (nm)
    void = min(abs(lengths-d), idx)
    ddy = lengths[idx]
    dy = imgdim[1] / d               ;  pixels / nm
    sy = ddy*dy*tr[0,0]              ;  scale bar length (nm) to pixels and Y zoom factor

    d = zr[1] - zr[0]                ;  volume range in Z (nm)
    void = min(abs(lengths-d), idx)
    ddz = lengths[idx]
    dz = imgdim[0] / d               ;  pixels / nm
    sz = ddz*dz*tr[1,1]              ;  scale bar length (nm) to pixels and Z zoom factor

    ;  Draw the scale bar
    oBar = self.oView[0]->GetByName('MODEL/SCALE_X_X')
    oBar->SetProperty, DATA=[[vr[2]-sz-10,20,0],[vr[2]-10,20,0]], HIDE=0
    oBar = self.oView[0]->GetByName('MODEL/SCALE_X_Y')
    oBar->SetProperty, DATA=[[vr[2]-10,20,0],[vr[2]-10,20+sy,0]], HIDE=0

    oText = self.oView[0]->GetByName('MODEL/TEXT_X_X')
    oText->GetProperty, CHAR_DIMENSIONS=cdim
    s = strtrim(ddz,2)+' nm'
    wd = strlen(s)*cdim[0] - 20*(400.0/vr[2])
    oText->SetProperty, STRING=s, LOCATION=[vr[2]-sz-wd,15,0], HIDE=0
    oText = self.oView[0]->GetByName('MODEL/TEXT_X_Y')
    oText->GetProperty, CHAR_DIMENSIONS=cdim
    s = strtrim(ddy,2)+' nm'
    wd = strlen(s)*cdim[0] - 20*(400.0/vr[2])
    oText->SetProperty, STRING=s, LOCATION=[vr[2]-wd,23+sy,0], HIDE=0
end


;-------------------------------------------------------------------------------------
;+
;  Update the XZ scale bar
;
; @param imgdim {in}{type=vector}{required}
;   Image dimensions
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::UpdateXZScaleBar, imgdim
    compile_opt idl2, logical_predicate

    self.oMainGUI->GetProperty, DISPLAY_3D=o3DDisplay
    oModel = o3DDisplay->GetObjectByName('PALM')
    oModel->GetProperty, X_RANGE=yr, Z_RANGE=zr
    oXY = self.oView[1]->GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL')
    oXY->GetProperty, TRANSFORM=tr
    self.oView[1]->GetProperty, VIEWPLANE_RECT=vr
    lengths = [10, 20, 30, 50, 100, 200, 300, 400, 500, 1000] / 2

    d = yr[1] - yr[0]                ;  volume range in Y (nm)
    void = min(abs(lengths-d), idx)
    ddy = lengths[idx]
    dy = imgdim[0] / d               ;  pixels / nm
    sy = ddy*dy*tr[0,0]              ;  scale bar length (nm) to pixels and Y zoom factor

    d = zr[1] - zr[0]                ;  volume range in Z (nm)
    void = min(abs(lengths-d), idx)
    ddz = lengths[idx]
    dz = imgdim[1] / d               ;  pixels / nm
    sz = ddz*dz*tr[1,1]              ;  scale bar length (nm) to pixels and Z zoom factor

    ;  Draw the scale bar
    oBar = self.oView[1]->GetByName('MODEL/SCALE_Y_X')
    oBar->SetProperty, DATA=[[vr[2]-sy-10,20,0],[vr[2]-10,20,0]], HIDE=0
    oBar = self.oView[1]->GetByName('MODEL/SCALE_Y_Y')
    oBar->SetProperty, DATA=[[vr[2]-10,20,0],[vr[2]-10,20+sz,0]], HIDE=0

    oText = self.oView[1]->GetByName('MODEL/TEXT_Y_X')
    oText->GetProperty, CHAR_DIMENSIONS=cdim
    s = strtrim(ddy,2)+' nm'
    wd = strlen(s)*cdim[0] - 20*(400.0/vr[2])
    oText->SetProperty, STRING=s, LOCATION=[vr[2]-sy-wd,15,0], HIDE=0
    oText = self.oView[1]->GetByName('MODEL/TEXT_Y_Y')
    oText->GetProperty, CHAR_DIMENSIONS=cdim
    s = strtrim(ddz,2)+' nm'
    wd = strlen(s)*cdim[0] - 20*(400.0/vr[2])
    oText->SetProperty, STRING=s, LOCATION=[vr[2]-wd,23+sz,0], HIDE=0
end


;-------------------------------------------------------------------------------------
;+
; This method updates the images
;
;
; @keyword
;   ALL {in}{type=boolean}{optional}
;       Set this keyword to have all of the images updated
;
; @keyword
;   Z
;       Set this keyword to have the Z image updated
;
; @keyword
;   Y
;       Set this keyword to have the Y image updated
;
; @keyword
;   X
;       Set this keyword to have the X image updated
;
; @keyword
;   INPUT_INDEX {in}{optional}{type=long}
;       Set this keyword to the specific index of the plane within the
;       volume to display.  The default action is to calculate the
;       index.
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::UpdateImages, $
    ALL=doAll, $
    Z=doZ, $
    Y=doY, $
    X=doX, $
    INPUT_INDEX = inputindex
    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif
    if ~ptr_valid(self.pVolume) then $
        return
    if n_elements(*self.pVolume) EQ 0 then $
        return

    if keyword_set(doAll) then $
        doZ = (doY = (doX = 1))
    self.oMainGUI->GetProperty, DISPLAY_3D=oDisplay3D
    oDisplay3D->GetProperty, HAVE_EM_VOLUME=haveEM
    useHue = ptr_valid(self.bVol) && n_elements(*self.bVol) GT 0 $
        && ptr_valid(self.gVol) && n_elements(*self.gVol) GT 0 $
        && ptr_valid(self.rVol) && n_elements(*self.rVol) GT 0
    volDims = size(*self.pVolume, /DIMENSION)
    if keyword_set(doX) then begin
        self -> GetDisplayObject, X_IMAGE=oImage, $
                                  X_TEXT=oText, $
                                  Z_IMAGE=oImageZ, $
                                  Z_POLYLINE_VERTICAL=oPoly
        oImageZ -> GetProperty, LOCATION=loc
        oPoly -> GetProperty, DATA=verts
        index = n_elements(inputindex) ne 0 ? $
            inputindex : $
            fix((verts[0,0]-loc[0])<(volDims[0]-1))
        if (widget_info(self.oMainGUI->Get('UseHue'), /BUTTON_SET)  &&  $
            widget_info(self.oMainGUI->Get('UseHueMPR'), /BUTTON_SET)) then begin
            self->YZHueImage, oImage, volDims, index
            dims = [volDims[2],volDims[1]]
        endif else begin
            image = transpose(reform(256*reform((*self.pVolume)[index,*,*])<255))
            oImage -> SetProperty, DATA=image
            dims = size(image, /DIM)
        endelse
        oText -> SetProperty, STRINGS='X='+strtrim(index,2)
        self->UpdateYZScaleBar, dims
    endif

    if keyword_set(doY) then begin
        self -> GetDisplayObject, Y_IMAGE=oImage, $
                                  Y_TEXT=oText, $
                                  Z_IMAGE=oImageZ, $
                                  Z_POLYLINE_HORIZONTAL=oPoly
        oImageZ -> GetProperty, LOCATION=loc
        oPoly -> GetProperty, DATA=verts
        index = n_elements(inputindex) ne 0 ? $
            inputindex : $
            fix((verts[1,0]-loc[1])<(volDims[1]-1))
        if (widget_info(self.oMainGUI->Get('UseHue'), /BUTTON_SET) &&  $
            widget_info(self.oMainGUI->Get('UseHueMPR'), /BUTTON_SET)) then begin
            self->XZHueImage, oImage, volDims, index
            dims = [volDims[0],volDims[2]]
        endif else begin
            image = reform(256*reform((*self.pVolume)[*,index,*])<255)
            dims = size(image, /DIM)
            oImage -> SetProperty, DATA=image
        endelse
        oText -> SetProperty, STRINGS='Y='+strtrim(index,2)
        self->UpdateXZScaleBar, dims
    endif

    if keyword_set(doZ) then begin
        self -> GetDisplayObject, Y_IMAGE=oImageY, $
                                  Z_TEXT=oText, $
                                  Y_POLYLINE_HORIZONTAL=oPoly, $
                                  Z_IMAGE=oImage
        oImageY -> GetProperty, LOCATION=loc
        oPoly -> GetProperty, DATA=verts
        index = n_elements(inputindex) ne 0 ? $
            inputindex : $
            fix((verts[1,0]-loc[1])<(volDims[2]-1))
        if (widget_info(self.oMainGUI->Get('UseHue'), /BUTTON_SET) &&  $
            widget_info(self.oMainGUI->Get('UseHueMPR'), /BUTTON_SET)) then begin
            self->XYHueImage, oImage, volDims, index
            dims = volDims[0:1]
        endif else begin
            image = reform(256*reform((*self.pVolume)[*,*,index])<255)
            oImage->SetProperty, DATA=image
            dims = size(image, /DIM)
        endelse
        oText -> SetProperty, STRINGS='Z='+strtrim(index,2)
        self->UpdateXYScaleBar, dims
    endif

    case check_math() of
        0:
        32: ; Do nothing on overflow error
        else: print, !error_state.Msg ; Print all other messages
    endcase
    self->UpdateImagesEM, $
        INPUT_INDEX=InputIndex, $
        X=doX, Y=doY, Z=doZ
    
end

;-------------------------------------------------------------------------------------
;+
; This method is called when a new volume has been loaded.  All of the underlying
; graphics objects are set accordingly or reset.
;
; @History
;   Nov, 2009 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::UpdateImagesEM, $
    INPUT_INDEX=InputIndex, $
    Z=doZ, $
    Y=doY, $
    X=doX

    self.oMainGUI->GetProperty, $
        DISPLAY_3D=o3DDisplay
    o3DDisplay->GetProperty, HAVE_EM_VOLUME=haveEM
    if ~haveEM then $
        return

    dimsVol = size(*self.pVolume, /DIMENSIONS)
    dimsEM = size(*self.pVolumeEM, /DIMENSIONS)
    haveIndex = n_elements(InputIndex) GT 0
    if keyword_set(doX) then begin
        index = round(float((haveIndex ? InputIndex : self.index[0]))/dimsVol[0]*dimsEM[0])
        image = transpose(reform(bytscl((*self.pVolumeEM)[index,*,*])))
        self->GetDisplayObject, X_EM_IMAGE=oImage
        oImage->SetProperty, DATA=temporary(image)
    endif
    if keyword_set(doY) then begin
        index = round(float((haveIndex ? InputIndex : self.index[1]))/dimsVol[1]*dimsEM[1])
        image = reform(bytscl((*self.pVolumeEM)[*,index,*]))
        self->GetDisplayObject, Y_EM_IMAGE=oImage
        oImage->SetProperty, DATA=temporary(image)
    endif
    if keyword_set(doZ) then begin
        index = round(float((haveIndex ? InputIndex : self.index[2]))/dimsVol[2]*dimsEM[2])
        image = reform(bytscl((*self.pVolumeEM)[*,*,index]))
        self->GetDisplayObject, Z_EM_IMAGE=oImage
        oImage->SetProperty, DATA=temporary(image)
    endif

end

;-------------------------------------------------------------------------------------
;+
; This method is called when a new volume has been loaded.  All of the underlying
; graphics objects are set accordingly or reset.
;
; @param oModel {in}{type=PALMgr3DModel object}{required}
;   The 3D model
;
; @keyword NO_DRAW {in}{type=boolean}{optional}
;   If set, do not update graphics
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::UpdateModel, oModel, $
    NO_DRAW=noDraw

    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    if ~obj_valid(oModel) then $
        return
    if ~obj_isa(oModel, 'PALMgr3DModel') then $
        return

    ptr_free, self.rvol
    ptr_free, self.gvol
    ptr_free, self.bvol
    oVol = oModel->GetVolObj()
    oVol->GetProperty, DATA0=r, DATA1=g, DATA2=b
    self.rvol = ptr_new(r,/NO_COPY)
    self.gvol = ptr_new(g,/NO_COPY)
    self.bvol = ptr_new(b,/NO_COPY)

    self.pVolume = oModel -> GetVolPtr()
    oModel -> GetProperty, VOLUME_XRANGE=xRange, $
                           VOLUME_YRANGE=yRange, $
                           VOLUME_ZRANGE=zRange, $
                           VOLUME_MAX=maxVolElem
    self.volDims = size(*self.pVolume, /DIM)
    self.maxVolumeElement = maxVolElem
    DataRange = [min(*self.pVolume, MAX=maxVal), maxVal]

    self.oView[2] -> GetProperty, DIMENSIONS=dims
    self.BSIncrement = (DataRange[1]-DataRange[0])/100
    self.index[2] = self.volDims[2]/2
    self.index[1] = self.volDims[1]/2
    self.index[0] = self.volDims[0]/2
    range = [xRange[1]-xRange[0], $
             yRange[1]-yRange[0], $
             zRange[1]-zRange[0]]

    pixelLength = range / self.volDims
    ratio = max(pixelLength) / pixelLength
    ratio = ratio / max(ratio)
    ;For use in sizing:
    self.limits = max(self.volDims)*ratio

    ;This will set up the viewplanes and axes:
    self->SetViews
    self.BSRange = [min(*self.pVolume, MAX=maxVal), maxVal]

    self -> UpdateImages, /ALL
    self -> Update3DDisplay, /NO_DRAW
    if ~keyword_set(noDraw) then $
        self -> RenderScene

end


;-------------------------------------------------------------------------------------
;+
; Update the volume for manipulation.
;
; @keyword NO_DRAW {in}{type=boolean}{optional}
;   If set, do not update graphics
;
; @keyword RELEASE {in}{required}
;   Set if mouse up.
;
; @keyword START {in}{required}
;   Set if mouse down (initial)
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
pro PALM_XYZDisplay::UpdateVolumeForManip, $
    NO_DRAW=noDraw, $
    RELEASE=release, $
    START=start

    if ~obj_valid(self.oMainGUI) then $
        return

    if ~obj_isa(self.oMainGUI, 'PALM_MainGUI') then $
        return

    self.oMainGUI -> GetProperty, DISPLAY_3D=o3DDisplay
    if ~obj_valid(o3DDisplay) then $
        return
; No need to update if no slice lines/planes are present
    o3DDisplay -> GetProperty, $
        HIDE_SLICE_LINES=HideSliceLines, $
        HIDE_SLICE_PLANES=HideSlicePlanes
    if HideSliceLines AND HideSlicePlanes then $
        return

; Update the display according to the drag quality setting
    o3DDisplay -> UpdateManip, NO_DRAW=noDraw, $
        RELEASE=release, $
        START=start

end


;-------------------------------------------------------------------------------------
;+
; This method adjusts the byte-scaling values
;
;
; @param
;   XY {in}{type=integer}{required}
;       Set this keyword to a 2-element vector specifying a position in the
;       display
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::WindowLevel, xy

    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    if n_elements(xy) LT 2 then $
        return
    if ~ptr_valid(self.pVolume) then $
        return
    if n_elements(*self.pVolume) EQ 0 then $
        return

    if xy[1] GT self.xy[1] then begin
        self.BSRange[1] = self.BSRange[1]-self.BSIncrement
        self.BSRange[0] = (self.BSrange[0]+self.BSIncrement) < self.BSRange[1]
    endif

    if xy[1] LT self.xy[1] then begin
        self.BSRange[1] = self.BSRange[1]+self.BSIncrement
        self.BSRange[0] = self.BSRange[0]-self.BSIncrement
    endif

    if xy[0] GT self.xy[0] then begin
        self.BSRange = self.BSRange + replicate(self.BSIncrement,2)
    endif

    if xy[0] LT self.xy[0] then begin
        self.BSRange = self.BSRange - replicate(self.BSIncrement,2)
    endif

    self.xy = xy
    self -> UpdateImages, /ALL
    self -> RenderScene

end


;-------------------------------------------------------------------------------------
;+
; Zoom a window
;
; @param oWindow {in}{type=window object}{required}
;   Which window to zoom
;
; @param
;   XY {in}{type=integer}{required}
;       Set this keyword to a 2-element vector specifying a position in the
;       display
;
; @keyword INITIAL_POSITION {in}{type=boolean}{optional}
;   If set, this is the initial zoom position
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay::Zoom, oWindow, xy, $
    INITIAL_POSITION=InitialPosition

    compile_opt StrictArr
    on_error, 2
    err = 0
    catch, err
    if err NE 0 then begin
        catch, /CANCEL
        help, /LAST
        return
    endif

    if n_elements(xy) LT 2 then $
        return
    if ~ptr_valid(self.pVolume) then $
        return
    if n_elements(*self.pVolume) EQ 0 then $
        return

    oWindow -> GetProperty, GRAPHICS_TREE=oView
    if keyword_set(initialPosition) then begin
       self.DragStart = xy
       for i = 0, n_elements(self.oView)-1 do begin
           oModel = self.oView[i] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL')
           oModel->GetProperty, TRANSFORM = initialTransform
           oModel->SetProperty, UVALUE = initialTransform
       endfor
       return
    endif

    if (array_equal(self.DragStart, [-1, -1])) then $
        return

    for i = 0, n_elements(self.oView)-1 do begin
        oImage = self.oView[i] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL/IMAGE')
        oImage->GetProperty, Dimensions = ImageDimensions
        oModel = self.oView[i] -> GetByName('MODEL/TRANSLATIONMODEL/SCALEMODEL')
        oTopModel = self.oView[i] ->GetByName('MODEL')
        oModel->GetProperty, UVALUE = initialTransform
        oModel->Reset
        oModel->SetProperty, TRANSFORM = initialTransform
    ;
    ; Grow or shrink the model depending on whether we're making a
    ; larger or smaller imaginary circle, relative to the center
    ; of the view.
    ;
        dy = xy[1] - self.DragStart[1]
        dR = 0.99^dy
    ;
    ; Move the center of the display to the origin, scale, then translate
    ; it back.  This ensures the center pixel stays in the middle.
    ;
    ; Since the viewplane rectangle doesn't match the dimensions of the
    ; view (I don't know why; I'm just fixing the zoom part), we have
    ; to take into account the scale difference between them.
    ;
        self.oWindow[i]->GetProperty, DIMENSIONS = dimensions
        self.oView[i]->GetProperty, Viewplane_Rect = vpr
        fy = vpr[3]/dimensions[1]
        fx = vpr[2]/dimensions[0]
        YOffset = dimensions[1]/2.
        oModel->Translate, -dimensions[0]/2.*fx, -YOffset*fy, 0.
        oModel->Scale, dR, dR , 1.
        oModel->Translate, dimensions[0]/2.*fx, YOffset*fy, 0

    endfor

    self->RenderScene
    self->UpdateImages, /ALL

end

;-------------------------------------------------------------------------------------
;+
;
;
; @field
;   AxisTransparency
;       The transparency value for the polylines. 0=transparent, 255=opaque
;
; @field
;   BSIncrement
;       The increment by which the byte-scaling values change
;
; @field
;   BSRange
;       The byte-scale range
;
; @field
;   color
;       A 3x3 byte array specifying the x, y and z colors
;
; @field
;   index
;       An array containing the x, y and z slice indices
;
; @field
;   oSelect
;       A pointer to  the currently selected polylines
;
; @field
;   oView
;       An object array containing the view object for the X, Y and Z display
;
; @field
;   oWindow
;       An object array containing references to the IDLitWindow objects
;
; @field
;   pVolume
;       A pointer to the current volume data
;
; @field
;   volDims
;       Holds the current volume dimensions
;
; @field
;   xy
;       recordds the last xy-position
; @field
;   DragStart
;       records the first xy-position of a drag operation (for
;       translation and scaling)
;
; @field rvol Local copy of hue red volume.
; @field gvol Local copy of hue green volume.
; @field bvol Local copy of hue blue volume.
;
; @Author
;   Daryl Atencio, ITT Visual Information Solutions Global Services Group
;
; @History
;   May, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;-
;-------------------------------------------------------------------------------------
pro PALM_XYZDisplay__Define

    void = {PALM_XYZDisplay, $

            AxisTransparency   : 0B, $
            BackgroundColor    : bytarr(3),   $
            BSIncrement        : 0.0,         $
            BSRange            : dblarr(2),   $
            color              : bytarr(3,3), $
            defaultWL          : dblarr(2),   $
            DragStart          : intarr(2),   $
            index              : intarr(3),   $
            limits             : dblarr(3),   $
            maxVolumeElement   : 0.0d,        $
            oBuffer            : obj_new(),   $
            oMainGUI           : obj_new(),   $
            oNotifier          : obj_new(),   $
            oPalette           : obj_new(),   $
            oPaletteEM         : obj_new(),   $
            oSelect            : ptr_new(),   $
            oView              : objarr(3),   $
            oWindow            : objarr(3),   $ ; Do not clean up
            oVolume            : obj_new(),   $ ; Do not clean up
            pVolume            : ptr_new(),   $ ; Do not clean up
            pVolumeEM          : ptr_new(),   $
            volDims            : intarr(3),   $
            xy                 : intarr(2),   $
            rvol               : ptr_new(),   $
            gvol               : ptr_new(),   $
            bvol               : ptr_new()    $
            }

end
