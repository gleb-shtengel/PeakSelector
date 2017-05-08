#ifndef ATSIFPROPERTIES_H
#define ATSIFPROPERTIES_H

typedef enum {
  // Using large numbers to force size to an integer
  ATSIF_AT_8         = 0x40000000,
  ATSIF_AT_U8        = 0x00000001,
  ATSIF_AT_32        = 0x40000002,
  ATSIF_AT_U32       = 0x40000003,
  ATSIF_Float        = 0x40000006,
  ATSIF_Double       = 0x40000007,
  ATSIF_String       = 0x40000008
} ATSIF_PropertyType;

  // Property Strings

#define  ATSIF_PROP_TYPE                       "Type"
#define  ATSIF_PROP_ACTIVE                     "Active"
#define  ATSIF_PROP_VERSION                    "Version"
#define  ATSIF_PROP_TIME                       "Time"
#define  ATSIF_PROP_FORMATTED_TIME             "FormattedTime"
#define  ATSIF_PROP_FILENAME                   "FileName"
#define  ATSIF_PROP_TEMPERATURE                "Temperature"
#define  ATSIF_PROP_UNSTABILIZEDTEMPERATURE    "UnstabalizedTemperature"
#define  ATSIF_PROP_HEAD                       "Head"
#define  ATSIF_PROP_HEADMODEL                  "HeadModel"
#define  ATSIF_PROP_STORETYPE                  "StoreType"
#define  ATSIF_PROP_DATATYPE                   "DataType"
#define  ATSIF_PROP_SIDISPLACEMENT             "SIDisplacement"
#define  ATSIF_PROP_SINUMBERSUBFRAMES          "SINumberSubFrames"
#define  ATSIF_PROP_PIXELREADOUTTIME           "PixelReadOutTime"
#define  ATSIF_PROP_TRACKHEIGHT                "TrackHeight"
#define  ATSIF_PROP_READPATTERN                "ReadPattern"
#define  ATSIF_PROP_READPATTERN_FULLNAME       "ReadPatternFullName"
#define  ATSIF_PROP_SHUTTERDELAY               "ShutterDelay"
#define  ATSIF_PROP_CENTREROW                  "CentreRow"
#define  ATSIF_PROP_ROWOFFSET                  "RowOffset"
#define  ATSIF_PROP_OPERATION                  "Operation"
#define  ATSIF_PROP_MODE                       "Mode"
#define  ATSIF_PROP_MODE_FULLNAME              "ModeFullName"
#define  ATSIF_PROP_TRIGGERSOURCE              "TriggerSource"
#define  ATSIF_PROP_TRIGGERSOURCE_FULLNAME     "TriggerSourceFullName"
#define  ATSIF_PROP_TRIGGERLEVEL               "TriggerLevel"
#define  ATSIF_PROP_EXPOSURETIME               "ExposureTime"
#define  ATSIF_PROP_DELAY                      "Delay"
#define  ATSIF_PROP_INTEGRATIONCYCLETIME       "IntegrationCycleTime"
#define  ATSIF_PROP_NUMBERINTEGRATIONS         "NumberIntegrations"
#define  ATSIF_PROP_KINETICCYCLETIME           "KineticCycleTime"
#define  ATSIF_PROP_FLIPX                      "FlipX"
#define  ATSIF_PROP_FLIPY                      "FlipY"
#define  ATSIF_PROP_CLOCK                      "Clock"
#define  ATSIF_PROP_ACLOCK                     "AClock"
#define  ATSIF_PROP_IOC                        "IOC"
#define  ATSIF_PROP_FREQUENCY                  "Frequency"
#define  ATSIF_PROP_NUMBERPULSES               "NumberPulses"
#define  ATSIF_PROP_FRAMETRANSFERACQMODE       "FrameTransferAcquisitionMode"
#define  ATSIF_PROP_BASELINECLAMP              "BaselineClamp"
#define  ATSIF_PROP_PRESCAN                    "PreScan"
#define  ATSIF_PROP_EMREALGAIN                 "EMRealGain"
#define  ATSIF_PROP_BASELINEOFFSET             "BaselineOffset"
#define  ATSIF_PROP_SWVERSION                  "SWVersion"
#define  ATSIF_PROP_SWVERSIONEX                "SWVersionEx"
#define  ATSIF_PROP_MCP                        "MCP"
#define  ATSIF_PROP_GAIN                       "Gain"
#define  ATSIF_PROP_VERTICALCLOCKAMP           "VerticalClockAmp"
#define  ATSIF_PROP_VERTICALSHIFTSPEED         "VerticalShiftSpeed"
#define  ATSIF_PROP_OUTPUTAMPLIFIER            "OutputAmplifier"
#define  ATSIF_PROP_PREAMPLIFIERGAIN           "PreAmplifierGain"
#define  ATSIF_PROP_SERIAL                     "Serial"
#define  ATSIF_PROP_DETECTORFORMATX            "DetectorFormatX"
#define  ATSIF_PROP_DETECTORFORMATZ            "DetectorFormatZ"
#define  ATSIF_PROP_NUMBERIMAGES               "NumberImages"

#endif

