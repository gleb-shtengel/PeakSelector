typedef int BYTE;

struct TUserText
{
  char *text;
};


struct TShutter
{
	char type;
  char mode;
  char custom_bg_mode;
  char custom_mode;
  float closing_time;
  float opening_time;
};


struct TShamrockSave
{
	int IsActive;
  int WavePresent;
  float Wave;
  int GratPresent;
  int GratIndex;
  float GratLines;
  char GratBlaze[32];
  int SlitPresent;
  float SlitWidth;
  int FlipperPresent;
  int FlipperPort;
  int FilterPresent;
  int FilterIndex;
  char FilterString[32];
  int AccessoryPresent;
  int Port1State;
  int Port2State;
  int Port3State;
  int Port4State;
  int OutputSlitPresent;
  float OutputSlitWidth;
};



struct TInstaImage
{
	BYTE head;                 // which head 1 2 etc
	BYTE store_type;           // single background reference source etc
	BYTE data_type;            // X,XY,XYZ do not know if this should be here
	BYTE mode;      				//realtime singlescan etc
	BYTE trigger_source;
	BYTE sync;              // Internal or external
	BYTE read_pattern;        // LIS MT or RANDOM
	BYTE shutter_delay;     // ON or OFF

	unsigned int type;                 // int long float
	unsigned int active;               // does it contain valid data
	unsigned int structure_version;
	int no_integrations;
	int no_points;         // must be tied to image_format
	int fast_track_height;
	int gain;              // should this be an index or actual gain
	int track_height;		// must be tied to TImage. Not valid in LIS or CI
	int series_length;         // must be tied to TImage class
	int operation_mode;         // InstaSpec II,IV,V
	int  mt_offset;
	int st_centre_row;
  int FlipX, FlipY, Clock, AClock, Gain, MCP, Prop, IOC;
  float Freq;

	char head_model[270];
	int detector_format_x;
	int detector_format_z;
	time_t timedate;
	char filename[270];			// MAXPATH for 32 bit is 260
	float temperature;
  float unstabilizedTemperature;
	float trigger_level;
	float exposure_time;
	float delay;
	float integration_cycle_time;
	float kinetic_cycle_time;
	float gate_delay;
	float gate_width;
	float GateStep;
	float pixel_readout_time;
 	struct TUserText user_text;
 	struct TShutter shutter;
  struct TShamrockSave shamrock_save;

  int VertClockAmp;
  float data_v_shift_speed;

  float PreAmpGain;
  int OutputAmp, Serial;

  int NumPulses;
  int mFrameTransferAcqMode;
  int mBaselineClamp;
  int mPreScan;
  int mEMRealGain;
  int mBaselineOffset;
  unsigned long mSWVersion;
};


struct TCalibImage
{
	BYTE x_type;
	BYTE x_unit;
	BYTE y_type;
	BYTE y_unit;
	BYTE z_type;
	BYTE z_unit;
	float x_cal[4];
	float y_cal[4];
	float z_cal[4];
	char* x_text;
	char* y_text;
	char* z_text;
	float rayleigh_wavelength;
	float pixel_length;
	float pixel_height;
};


 
struct LONG_RECT
{
	int left;
	int top;
	int right;
	int bottom;
};

struct TSubImage
{
	int left;
	int top;
	int right;
	int bottom;
	int vertical_bin;
	int horizontal_bin;
};


struct TImage
{
	struct TSubImage *position;
	struct LONG_RECT image_format;
	int no_subimages;     // per image
	int no_images;        // each image format must be identical
	unsigned long *subimage_offset;
	unsigned long *time_stamps;
	unsigned long image_length;
	unsigned long total_length;
};
