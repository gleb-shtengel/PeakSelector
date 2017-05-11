pro PALM
    addthispath
    oMain = obj_new('PALM_MainGUI', $
        ERROR_MESSAGE=errMsg, $
        /VERBOSE)
    oModel = obj_new('PALMgr3DModel')
    oMain -> UpdateModel, oModel
end