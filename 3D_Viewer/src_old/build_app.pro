.full_reset_session
cd, CURRENT=dirCurrent
dir = dialog_pickfile(/DIR, $
    TITLE='Select the "Phase2 directory')
if dir EQ '' then $
    stop
if ~file_test(dir+'src', /DIRECTORY) then $
    stop
dirDistrib = dir+'distrib'+path_sep()
file_delete, dir+'distrib', $
    /ALLOW_NONEXISTENT, $
    /QUIET, $
    /RECURSIVE
file_mkdir, dirDistrib
itResolve
.compile AnchorWid_eventcb.pro
.compile AnchorWid.pro
.compile ExtractMultiLabelWid_eventcb.pro
.compile ExtractMultiLabelWid.pro
.compile FittingInfoWid_eventcb.pro
.compile FittingInfoWid.pro
.compile gauss2dfithh.pro
.compile gauss2Rdfithh.pro
.compile GroupWid_eventcb.pro
.compile GroupWid.pro
.compile GuideStarWid_eventcb.pro
.compile GuideStarWid.pro
.compile PeakSelector_eventcb.pro
.compile PeakSelector.pro
.compile readdat.pro
.compile read_out_files.pro
.compile readrawloop6.pro
.compile read_sif.pro
.compile ReExtractMultiLabelWid_eventcb.pro
.compile ReExtractMultiLabelWid.pro
.compile Remove_XYZ_Tilt_eventcb.pro
.compile Remove_XYZ_Tilt.pro
.compile RotWid_eventcb.pro
.compile RotWid.pro
.compile SaveZslicesTIFF_eventcb.pro
.compile SaveZslicesTIFF.pro
.compile TransformRawSaveSaveSumWid_eventcb.pro
.compile TransformRawSaveSaveSumWid.pro
.compile ZoperationsWid_eventcb.pro
.compile ZoperationsWid.pro
.compile addthispath.pro
.compile angle3123.pro
.compile cw_slider.pro
.compile palm_3ddisplay__define.pro
.compile palm_3dviewer.pro
.compile palm_colortabledialog__define.pro
.compile palm_dialog.pro
.compile palm_maingui__define.pro
.compile palm_mj2player__define.pro
.compile palm_statusbar__define.pro
.compile palm_volumeopacitydialog__define.pro
.compile palmgr3dmodel__define.pro
.compile palm_xyzdisplay__define.pro
.compile sourcepath.pro
.compile sourceroot.pro
.compile genericclassevent.pro
.compile genericclasskillnotify.pro
.compile genericclassnotifyrealize.pro
.compile palm_3dwindow__define.pro
.compile palm_xyzwindow__define.pro
.compile palm_3dobserver__define.pro
.compile palm_panmode__define.pro
.compile palm_rotatemode__define.pro
.compile palm_selectmode__define.pro
.compile palm_xyzobserver__define.pro
.compile palm_zoommode__define.pro
.compile applytransform.pro
.compile dicomtransform.pro
.compile oblique.pro
.compile padd.pro
.compile palm_cutplane_vis__define.pro
.compile polylineplaneintersect.pro
.compile trackball__define.pro
.compile idlitnotifier__define.pro
.compile idlitgrscene__define.pro
resolve_all, CLASS=['PALM_StatusBar', 'Trackball', 'PALMgr3DModel', 'PALM_ColorTableDialog', 'PALM_MJ2Player', 'IDLitNotifier',  $
           'PALM_3DDisplay', 'PALM_XYZDisplay', 'PALM_VolumeOpacityDialog', 'PALM_MainGUI', 'PALM_3DObserver', 'PALM_XYZObserver', $
           'IDLitgrScene','orb'], /CONTINUE_ON_ERROR
save, /routines, FILE=dirDistrib+'peakselector.sav'
cd, dirCurrent