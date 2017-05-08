/***********************************************************************
CLASS:					NA
NAME:						Alan Murray
PROTECTION: 		NA
RETURN:       	NA
Version 				1.2
Last Modified		01/09/1999
DESCRIPTION:		This program will read in any Andor SIF file
								with the prupose of storing the structure and
                identifying what the structure is.
**********************************************************************/
#include <io.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <conio.h>
#include <dir.h>
#include "andorsif.h"

FILE *in, *out;
struct TImage Image;
struct TInstaImage InstaImage;
struct TCalibImage CalibImage;
#define HIWORD(l)   ((short) (((long) (l) >> 16) & 0xFFFF))
#define LOWORD(l)   ((short) (l))


/*****************************************************************************/
//Function: read_string
//Inputs: Terminating charachter
//Outputs: String
//The purpose of this function is to read in string to a buffer
// and return the string to the calling function
/*****************************************************************************/

void read_string(char terminator,char *buffer)
{
	char ch;
	int i;
	i=0;
	ch ='a';
  if(!buffer){
   printf("Buffer array was not created\n");
   exit(0);
  }
	while (ch != terminator){
	 	fscanf(in,"%c",&ch);
		buffer[i]= ch; 					 // Add the charachter to the buffer
		i++;
	}
  i--;											 // to remove termination charachter
	buffer[i] = NULL;
}



/*****************************************************************************/
//Function: read_int
//Inputs: Terminating charachter
//Outputs: A number of type 'long'
//The purpose of this function is to read in a number and
// return the number to the calling function
/*****************************************************************************/

long read_int(char terminator)
{
	long value;
	char ch,*int_buffer;
	int i;
	i=0;
	ch ='a';
  int_buffer = (char*)malloc(MAXPATH);
  if(!int_buffer){
  	printf("int_buffer array was not created\n");
    exit(0);
  }
  do { fscanf(in,"%c",&ch);} while(isspace(ch));// Skip leading spaces
  int_buffer[i]= ch;  												  //Captures first char
  i++;
	while (ch != terminator){
		fscanf(in,"%c",&ch);
		if (((!isalpha(ch)) && (!isspace(ch)))){       //Don't store spaces or letters
			int_buffer[i]= ch;
  		i++;
    }
	}
	int_buffer[i] = NULL;
	value = (atol(int_buffer));                  //convert value to a long integer
  free(int_buffer);
  return value;
}



/*****************************************************************************/
//Function: read_float
//Inputs: Terminating charachter
//Outputs: A number of type 'float'
//The purpose of this function is to read in a number and
// return the number to the calling function
/*****************************************************************************/

float read_float(char terminator)

{
	float value;
	char ch, *float_buffer;
	int i;
	i=0;
	ch ='a';
  float_buffer = (char*)malloc(MAXPATH);
  if (!float_buffer){
  	printf("float_buffer array was not created\n");
    exit(0);
  }
  do { fscanf(in,"%c",&ch);} while(isspace(ch));// Skip leading spaces
  float_buffer[i]= ch;                          //Store any numbers read in
  i++;
	while (ch != terminator){
		fscanf(in,"%c",&ch);
		float_buffer[i]= ch;                      // Try checking spaces and letters
		i++;
	}
  i--;
	float_buffer[i] = NULL;
  if (strcmp(float_buffer, "+INF") == 0) strcpy(float_buffer, "0");
	value = (atof(float_buffer));               // Convert value to a float
  free(float_buffer);
  return value;
}



/*****************************************************************************/
//Function: read_byte
//Inputs: none
//Outputs: A number of type 'int'
//The purpose of this function is to do a single read of a byte
//It does this by reading a charachter and outputting its integer value
//i.e. the Integer equivalent of the ascii value.
/*****************************************************************************/

int read_byte_and_skip_terminator(void)

{
	char ch,termin_ch;
  int i;
	ch ='a';
		fscanf(in,"%c",&ch);
    fscanf(in,"%c",&termin_ch);             //Skips the terminater
    i = (int)(ch);                          // gives integer value of ascii code
    return i;
}

/*****************************************************************************/
//Function: read_char
//Inputs: none
//Outputs: A character
//The purpose of this function is to read in a character and
//return the character to the calling function
//It is provided for values of type char
/*****************************************************************************/

char read_char_and_skip_terminator(void)

{
		char termin_ch,ch;
		ch ='a';
	 	fscanf(in,"%c",&ch);
    fscanf(in,"%c",&termin_ch);             //Skips the terminater
    return ch;
}


/*****************************************************************************/
//Function: read_len_chars
//Inputs: The length of the string
//Outputs: The string
//The purpose of this function is to read in a predefined number
//of charachters and return them as a string
/*****************************************************************************/

void read_len_chars(int string_length,char *len_chars_buffer)



{
	char ch;
	int i;
	ch ='a';
	if(!len_chars_buffer){
  	printf("lens_char array was not created");
    exit(0);
  }
		for(i=0;i<string_length;i++){
			fscanf(in,"%c",&ch);         // Reads in a string of length string_length
      len_chars_buffer[i] = ch;
    }
    len_chars_buffer[i] = NULL;
}







/*****************************************************************************/
//Function: read_image
//Inputs: The total number of values stored for the image i.e. image length
//Outputs: A buffer full of floating point numbers( The image values )
//The purpose of this function is to read in the image as
//a buffer of floating point numbers
/*****************************************************************************/
void read_image(long rd_image_length,float *image_buffer)
{
  fread(image_buffer,4,(size_t)rd_image_length,in);
  																			    //Reads the image 4 bytes at a time

}// End of read_image

/*****************************************************************************/
//Function: print_instaimage
//Inputs: An array for versions,a buffer to store usertext
//Outputs: none
//The purpose of this function is to print out the instaimage structure data
/*****************************************************************************/

void print_instaimage(long pversion[4],char *usertext_buffer)
{

  if(!usertext_buffer) printf("Cannot create usertext_buffer");
  fprintf(out,"unsigned int: version: %d: Terminated by 'Space'\n",pversion[0]);
  fprintf(out,"unsigned int: InstaImage.type: %d: Terminated by 'Space'\n",InstaImage.type);
	fprintf(out,"unsigned int: InstaImage.active: %d: Terminated by 'Space'\n",InstaImage.active);
	fprintf(out,"unsigned int: InstaImage.structure_version: %d: Terminated by 'Space'\n",InstaImage.structure_version);
  fprintf(out,"time_t: InstaImage.timedate: %d: Terminated by 'Space'\n",InstaImage.timedate);
  fprintf(out,"Float: Temperature: %f: Terminated by 'Space'\n",InstaImage.temperature);
  fprintf(out,"Byte: head: %d: Terminated by 'Space'\n",InstaImage.head);
  fprintf(out,"Byte: store_type: %d: Terminated by 'Space'\n",InstaImage.store_type);
  fprintf(out,"Byte: data_type: %d: Terminated by 'Space'\n",InstaImage.data_type);
  fprintf(out,"Byte: mode: %d: Terminated by 'Space'\n",InstaImage.mode);
  fprintf(out,"Byte: trigger_source: %d: Terminated by 'Space'\n",InstaImage.trigger_source);
  fprintf(out,"Float: trigger_level: %f: Terminated by 'Space'\n",InstaImage.trigger_level);
  fprintf(out,"Float: exposure_time: %f: Terminated by 'Space'\n",InstaImage.exposure_time);
  fprintf(out,"Float: delay: %f: Terminated by 'Space'\n",InstaImage.delay);
  fprintf(out,"Float: integration_cycle_time: %f: Terminated by 'Space'\n",InstaImage.integration_cycle_time);
  fprintf(out,"int: InstaImage.no_integrations: %d: Terminated by 'Space'\n",InstaImage.no_integrations);
  fprintf(out,"Byte: InstaImage.sync: %d: Terminated by 'Space'\n",InstaImage.sync);
  fprintf(out,"Float: InstaImage.kinetic_cycle_time: %f: Terminated by 'Space'\n",InstaImage.kinetic_cycle_time);
  fprintf(out,"Double: InstaImage.pixel_readout_time: %e: Terminated by 'Space'\n",InstaImage.pixel_readout_time);
  fprintf(out,"int: InstaImage.no_points: %d: Terminated by 'Space'\n",InstaImage.no_points);
  fprintf(out,"int: InstaImage.no_fast_track_height: %d: Terminated by 'Space'\n",InstaImage.fast_track_height);
  fprintf(out,"int: InstaImage.gain: %d: Terminated by 'Space'\n",InstaImage.gain);
  fprintf(out,"Float: InstaImage.gate_delay: %e: Terminated by 'Space'\n",InstaImage.gate_delay);
  fprintf(out,"Double: InstaImage.gate_width: %e: Terminated by 'Space'\n",InstaImage.gate_width);
  if( (HIWORD(pversion[0]) >= 1) && (LOWORD(pversion[0]) >=6) )
    fprintf(out,"Double: InstaImage.GateStep: %e: Terminated by 'Space'\n",InstaImage.GateStep);
  fprintf(out,"int: InstaImage.track_height: %d: Terminated by 'Space'\n",InstaImage.track_height);
  fprintf(out,"int: InstaImage.series_length: %d: Terminated by 'Space'\n",InstaImage.series_length);
  fprintf(out,"Byte: InstaImage.read_pattern: %d: Terminated by 'Space'\n",InstaImage.read_pattern);
  fprintf(out,"Byte: InstaImage.shutter_delay: %d: Terminated by 'Space'\n",InstaImage.shutter_delay);
  if( (HIWORD(pversion[0]) >= 1) && (LOWORD(pversion[0]) >=7) ) {
    fprintf(out,"int: InstaImage.st_centre_row: %d: Terminated by 'Space'\n",InstaImage.st_centre_row);
    fprintf(out,"int: InstaImage.mt_offset: %d: Terminated by 'Space'\n",InstaImage.mt_offset);
    fprintf(out,"int: InstaImage.operation_mode: %d: Terminated by 'Space'\n",InstaImage.operation_mode);
  }

  if( (HIWORD(pversion[0]) >= 1) && (LOWORD(pversion[0]) >=8) ) {
    fprintf(out,"int: InstaImage.FlipX: %d: Terminated by 'Space'\n",InstaImage.FlipX);
    fprintf(out,"int: InstaImage.FlipY: %d: Terminated by 'Space'\n",InstaImage.FlipY);
    fprintf(out,"int: InstaImage.Clock: %d: Terminated by 'Space'\n",InstaImage.Clock);
    fprintf(out,"int: InstaImage.AClock: %d: Terminated by 'Space'\n",InstaImage.AClock);
    fprintf(out,"int: InstaImage.MCP: %d: Terminated by 'Space'\n",InstaImage.MCP);
    fprintf(out,"int: InstaImage.Prop: %d: Terminated by 'Space'\n",InstaImage.Prop);
    fprintf(out,"int: InstaImage.IOC: %d: Terminated by 'Space'\n",InstaImage.IOC);
    fprintf(out,"Float: InstaImage.Freq: %e: Terminated by 'Space'\n",InstaImage.Freq);
  }

  if((HIWORD(pversion[0]) >= 1) && (LOWORD(pversion[0]) >=9)) {
    fprintf(out,"int: InstaImage.VertClockAmp: %d: Terminated by 'Space'\n",InstaImage.VertClockAmp);
    fprintf(out,"Float: InstaImage.data_v_shift_speed: %e: Terminated by 'Space'\n",InstaImage.data_v_shift_speed);
  }

  if((HIWORD(pversion[0]) >= 1) && (LOWORD(pversion[0]) >=10)) {
    fprintf(out,"int: InstaImage.OutputAmp: %d: Terminated by 'Space'\n",InstaImage.OutputAmp);
    fprintf(out,"Float: InstaImage.PreAmpGain: %e: Terminated by 'Space'\n",InstaImage.PreAmpGain);
  }

  if((HIWORD(pversion[0]) >= 1) && (LOWORD(pversion[0]) >=11))
    fprintf(out,"int: InstaImage.Serial: %d: Terminated by 'Space'\n",InstaImage.Serial);

  if((HIWORD(pversion[0]) >= 1) && (LOWORD(pversion[0]) >=13))
    fprintf(out,"int: InstaImage.NumPulses: %d: Terminated by 'Space'\n",InstaImage.NumPulses);

  if((HIWORD(pversion[0]) >= 1) && (LOWORD(pversion[0]) >=14))
    fprintf(out,"int: InstaImage.mFrameTransferAcqMode: %d: Terminated by 'Space'\n",InstaImage.mFrameTransferAcqMode);

  if((HIWORD(pversion[0]) >= 1) && (LOWORD(pversion[0]) >=15)) {
    fprintf(out,"float: InstaImage.unstabilizedTemperature: %d: Terminated by 'Space'\n",InstaImage.unstabilizedTemperature);
    fprintf(out,"int: InstaImage.mBaselineClamp: %d: Terminated by 'Space'\n",InstaImage.mBaselineClamp);
  }

  if((HIWORD(pversion[0]) >= 1) && (LOWORD(pversion[0]) >=16))
    fprintf(out,"int: InstaImage.mPreScan: %d: Terminated by 'Space'\n",InstaImage.mPreScan);

  if((HIWORD(pversion[0]) >= 1) && (LOWORD(pversion[0]) >=17))
    fprintf(out,"int: InstaImage.mEMRealGain: %d: Terminated by 'Space'\n",InstaImage.mEMRealGain);

  if((HIWORD(pversion[0]) >= 1) && (LOWORD(pversion[0]) >=18))
    fprintf(out,"int: InstaImage.mBaselineOffset: %d: Terminated by 'Space'\n",InstaImage.mBaselineOffset);

  if((HIWORD(pversion[0]) >= 1) && (LOWORD(pversion[0]) >=19))
    fprintf(out,"int: InstaImage.mSWVersion: %d: Terminated by 'Space'\n",InstaImage.mSWVersion);

  fprintf(out,"int: len: Terminated by 'Newline'\n");
  fprintf(out,"char*: InstaImage.head_model[270]: %s: Terminated by 'Space' + 'Newline' + 'Space'\n",InstaImage.head_model);
  fprintf(out,"int: InstaImage.detector_format_x: %d: Terminated by 'Space'\n",InstaImage.detector_format_x);
  fprintf(out,"int: InstaImage.detector_format_z: %d: Terminated by 'Space'\n",InstaImage.detector_format_z);
  fprintf(out,"int: len: Terminated by 'Newline'\n");
  fprintf(out,"char*: InstaImage.head_model[270]: %s: Terminated by 'Space' + 'Newline' + 'Space'\n",InstaImage.head_model);
  fprintf(out,"int: InstaImage.detector_format_x: %d: Terminated by 'Space'\n",InstaImage.detector_format_x);
  fprintf(out,"int: InstaImage.detector_format_z: %d: Terminated by 'Space'\n",InstaImage.detector_format_z);
  fprintf(out,"char*: InstaImage.head_model[270]: %s: Terminated by 'Space' + 'Newline' + 'Space'\n",InstaImage.head_model);
  fprintf(out,"int: InstaImage.detector_format_x: %d: Terminated by 'Space'\n",InstaImage.detector_format_x);
  fprintf(out,"int: InstaImage.detector_format_z: %d: Terminated by 'Space'\n",InstaImage.detector_format_z);
  fprintf(out,"int: len:  Terminated by 'Newline'\n");
  fprintf(out,"char: filename[270]: %s: Terminated by 'Space' + 'Newline'\n",InstaImage.filename);

  //Start of TUserText
  fprintf(out,"int: version: %d: Terminated by 'Space'\n",pversion[1]);
  fprintf(out,"int: len: Terminated by 'Space'\n");
  fprintf(out,"char*: InstaImage.user_text.text : %s: Terminated by 'Newline'\n",usertext_buffer);
  //End of TUserText

  //Start of TShutter
  fprintf(out,"int: version: %d: Terminated by 'Space' + 'Space'\n",pversion[2]);
  fprintf(out,"Char: InstaImage.shutter.type: %c: Terminated by 'Space' \n",InstaImage.shutter.type);
  fprintf(out,"Char: InstaImage.shutter.mode: %c: Terminated by 'Space' \n",InstaImage.shutter.mode);
  fprintf(out,"Char: InstaImage.shutter.custom_bg_mode: %c: Terminated by 'Space' \n",InstaImage.shutter.custom_bg_mode);
  fprintf(out,"Char: InstaImage.shutter.custom_mode: %c: Terminated by 'Space' \n",InstaImage.shutter.custom_mode);
  fprintf(out,"Float: InstaImage.closing_time: %f: Terminated by 'Space' + 'Space' \n",InstaImage.shutter.closing_time);
  fprintf(out,"Float: InstaImage.shutter.opening_time: %f: Terminated by 'Newline'\n",InstaImage.shutter.opening_time);
  // End of TShutter

  //Start of TShamrockSave
  fprintf(out,"int: version: %d: Terminated by 'Space' + 'Space'\n",pversion[3]);
  fprintf(out,"int: InstaImage.shamrock_save.IsActive: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.IsActive);
  fprintf(out,"int: InstaImage.shamrock_save.WavePresent: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.WavePresent);
  fprintf(out,"float: InstaImage.shamrock_save.Wave: %f: Terminated by 'Space'\n", InstaImage.shamrock_save.Wave);
  fprintf(out,"int: InstaImage.shamrock_save.GratPresent: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.GratPresent);
  fprintf(out,"int: InstaImage.shamrock_save.GratIndex: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.GratIndex);
  fprintf(out,"float: InstaImage.shamrock_save.GratLines: %f: Terminated by 'Space'\n", InstaImage.shamrock_save.GratLines);
  fprintf(out,"char: InstaImage.shamrock_save.GratBlaze[32]: %s: Terminated by 'NewLine'\n",InstaImage.shamrock_save.GratBlaze);
	fprintf(out,"int: InstaImage.shamrock_save.SlitPresent: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.SlitPresent);
  fprintf(out,"float: InstaImage.shamrock_save.SlitWidth: %f: Terminated by 'Space'\n", InstaImage.shamrock_save.SlitWidth);
  fprintf(out,"int: InstaImage.shamrock_save.FlipperPresent: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.FlipperPresent);
  fprintf(out,"int: InstaImage.shamrock_save.FlipperPort: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.FlipperPort);
  fprintf(out,"int: InstaImage.shamrock_save.FilterPresent: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.FilterPresent);
  fprintf(out,"int: InstaImage.shamrock_save.FilterIndex: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.FilterIndex);
  fprintf(out,"int: len:  Terminated by 'Space'\n");
  fprintf(out,"char: InstaImage.shamrock_save.FilterString[32]: %s: Terminated by 'Space'\n",InstaImage.shamrock_save.FilterString);
  fprintf(out,"int: InstaImage.shamrock_save.AccessoryPresent: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.AccessoryPresent);
  fprintf(out,"int: InstaImage.shamrock_save.Port1State: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.Port1State);
  fprintf(out,"int: InstaImage.shamrock_save.Port2State: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.Port2State);
  fprintf(out,"int: InstaImage.shamrock_save.Port3State: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.Port3State);
  fprintf(out,"int: InstaImage.shamrock_save.Port4State: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.Port4State);
  fprintf(out,"int: InstaImage.shamrock_save.OutputSlitPresent: %d: Terminated by 'Space'\n", InstaImage.shamrock_save.OutputSlitPresent);
  fprintf(out,"float: InstaImage.shamrock_save.OutputSlitWidth: %f: Terminated by 'Space' + 'Newline'\n", InstaImage.shamrock_save.OutputSlitWidth);





  //End of TShamrockSave

}// End of print_instaimage

/*****************************************************************************/
//Function: read_instaimage
//Inputs: none
//Outputs: none
//The purpose of this function is to read in the instaimage structure data
/*****************************************************************************/
void read_instaimage(void)
{
	char *instaimage_title;
	long version[4];
  long result;
  int len;
 	instaimage_title = (char*)malloc(MAXPATH);
  InstaImage.user_text.text = malloc(MAXPATH);
  version[0] = read_int(' '); //Version
  InstaImage.type = (unsigned int)read_int(' ');
  InstaImage.active = (unsigned int)read_int(' ');
  InstaImage.structure_version = (unsigned int)read_int(' ');
  InstaImage.timedate = read_int(' ');
  InstaImage.temperature = read_float(' ');
  InstaImage.head = read_byte_and_skip_terminator();
  InstaImage.store_type = read_byte_and_skip_terminator();
  InstaImage.data_type = read_byte_and_skip_terminator();
  InstaImage.mode = read_byte_and_skip_terminator();
  InstaImage.trigger_source = read_byte_and_skip_terminator();
  InstaImage.trigger_level = read_float(' ');
  InstaImage.exposure_time = read_float(' ');
  InstaImage.delay = read_float(' ');
  InstaImage.integration_cycle_time = read_float(' ');
  InstaImage.no_integrations = (int)read_int(' ');
  InstaImage.sync = read_byte_and_skip_terminator();
  InstaImage.kinetic_cycle_time = read_float(' ');
  InstaImage.pixel_readout_time = read_float(' ');
  InstaImage.no_points = (int)read_int(' ');
  InstaImage.fast_track_height = (int)read_int(' ');
	InstaImage.gain = (int)read_int(' ');
  InstaImage.gate_delay = read_float(' ');
  InstaImage.gate_width = read_float(' ');

  if( (HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=6) )
    InstaImage.GateStep = read_float(' ');

  InstaImage.track_height = (int)read_int(' ');
  InstaImage.series_length =(int) read_int(' ');
  InstaImage.read_pattern = read_byte_and_skip_terminator();
  InstaImage.shutter_delay = read_byte_and_skip_terminator();

  if( (HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=7) ) {
    InstaImage.st_centre_row = (int)read_int(' ');
    InstaImage.mt_offset = (int)read_int(' ');
    InstaImage.operation_mode = (int)read_int(' ');
  }

  if( (HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=8) ) {
    InstaImage.FlipX = (int)read_int(' ');
    InstaImage.FlipY = (int)read_int(' ');
    InstaImage.Clock = (int)read_int(' ');
    InstaImage.AClock = (int)read_int(' ');
    InstaImage.MCP = (int)read_int(' ');
    InstaImage.Prop = (int)read_int(' ');
    InstaImage.IOC = (int)read_int(' ');
    InstaImage.Freq = (int)read_int(' ');
  }

  if((HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=9)) {
    InstaImage.VertClockAmp = (int)read_int(' ');
    InstaImage.data_v_shift_speed = read_float(' ');
  }

  if((HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=10)) {
    InstaImage.OutputAmp = (int)read_int(' ');
    InstaImage.PreAmpGain = read_float(' ');
  }

  if((HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=11)) {
    InstaImage.Serial = (int)read_int(' ');
  }

  if((HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=13)) {
    InstaImage.NumPulses = (int)read_int(' ');
  }

  if((HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=14)) {
    InstaImage.mFrameTransferAcqMode = (int)read_int(' ');
  }

  if((HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=15)) {
    InstaImage.unstabilizedTemperature = (int)read_float(' ');
    InstaImage.mBaselineClamp = (int)read_int(' ');
  }

  if((HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=16))
    InstaImage.mPreScan = (int)read_int(' ');

  if((HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=17))
    InstaImage.mEMRealGain = (int)read_int(' ');

  if((HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=18))
    InstaImage.mBaselineOffset = (int)read_int(' ');

  if((HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=19))
    InstaImage.mSWVersion = (unsigned long)read_int(' ');

  if( (HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=5) ){
    read_int('\n');//
    read_string('\n',instaimage_title);
    strcpy(InstaImage.head_model,instaimage_title);
    InstaImage.detector_format_x = (int)read_int(' ');
    InstaImage.detector_format_z = (int)read_int(' ');
  }
  else if( (HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=3) ){
    unsigned long head_model = (int)read_int(' ');
    sprintf(InstaImage.head_model,"%lu",head_model);
    InstaImage.detector_format_x = (int)read_int(' ');
    InstaImage.detector_format_z = (int)read_int(' ');
  }
  else {
	 strcpy(InstaImage.head_model,"Unknown");
	 InstaImage.detector_format_x = 1024u;
	 InstaImage.detector_format_z = 256u;
  }

  read_int('\n');
  read_string('\n',instaimage_title);
  strcpy(InstaImage.filename,instaimage_title);

  //Start of TUserText
  version[1] = read_int(' ');
  result=read_int('\n');
  read_len_chars((int)result,InstaImage.user_text.text);
  //End of TUserText

  //Start of TShutter
  if( (HIWORD(version[0]) >= 1) && (LOWORD(version[0]) >=4) ) {
    version[2] = read_int(' ');
    InstaImage.shutter.type = read_char_and_skip_terminator();
    InstaImage.shutter.mode = read_char_and_skip_terminator();
    InstaImage.shutter.custom_bg_mode = read_char_and_skip_terminator();
    InstaImage.shutter.custom_mode = read_char_and_skip_terminator();
    InstaImage.shutter.closing_time = read_float(' ');
    InstaImage.shutter.opening_time = read_float('\n');
  }
  // End of TShutter

  //Start of TShamrockSave
  if( (HIWORD(version[0])>1) || ((HIWORD(version[0])==1) && (LOWORD(version[0]) >=12)) ){
    version[3] = read_int(' ');
    InstaImage.shamrock_save.IsActive = read_int(' ');
  	InstaImage.shamrock_save.WavePresent = read_int(' ');
  	InstaImage.shamrock_save.Wave = read_float(' ');
    InstaImage.shamrock_save.GratPresent = read_int(' ');
    InstaImage.shamrock_save.GratIndex = read_int(' ');
    InstaImage.shamrock_save.GratLines = read_float(' ');
    read_string('\n',InstaImage.shamrock_save.GratBlaze);
    InstaImage.shamrock_save.SlitPresent = read_int(' ');
    InstaImage.shamrock_save.SlitWidth = read_float(' ');
    InstaImage.shamrock_save.FlipperPresent = read_int(' ');
    InstaImage.shamrock_save.FlipperPort = read_int(' ');
    InstaImage.shamrock_save.FilterPresent = read_int(' ');
    InstaImage.shamrock_save.FilterIndex = read_int(' ');
    len = read_int(' ');
    read_len_chars(len, InstaImage.shamrock_save.FilterString);
    InstaImage.shamrock_save.AccessoryPresent = read_int(' ');
    InstaImage.shamrock_save.Port1State = read_int(' ');
    InstaImage.shamrock_save.Port2State = read_int(' ');
    InstaImage.shamrock_save.Port3State = read_int(' ');
    InstaImage.shamrock_save.Port4State = read_int(' ');
    InstaImage.shamrock_save.OutputSlitPresent = read_int(' ');
    InstaImage.shamrock_save.OutputSlitWidth = read_float(' ');
  }
  //End of TShamrockSave

  print_instaimage(version,InstaImage.user_text.text);
  free(instaimage_title);
  free(InstaImage.user_text.text);
}// End of read_instaimage

/*****************************************************************************/
//Function: print_calib_image
//Inputs: 3 buffers to store x_text,y_text and z_text
//Outputs: none
//The purpose of this function is to print out the calibimage structure data
/*****************************************************************************/
void print_calib_image(long pcversion,char *x_text_buffer,char *y_text_buffer
												,char *z_text_buffer)
{
  if(!x_text_buffer) printf("Cannot create x_text_buffer");
  if(!y_text_buffer) printf("Cannot create y_text_buffer");
  if(!z_text_buffer) printf("Cannot create z_text_buffer");
  fprintf(out,"int: version: %d: Terminated by 'Space'\n",pcversion);
  fprintf(out,"Byte: CalibImage.x_type: %d: Terminated by 'Space'\n"
  	,CalibImage.x_type);
  fprintf(out,"Byte: CalibImage.x_unit: %d: Terminated by 'Space'\n"
  	,CalibImage.x_unit);
  fprintf(out,"Byte: CalibImage.y_type: %d: Terminated by 'Space'\n"
  	,CalibImage.y_type);
  fprintf(out,"Byte: CalibImage.y_unit: %d: Terminated by 'Space'\n"
  	,CalibImage.y_unit);
  fprintf(out,"Byte: CalibImage.z_type: %d: Terminated by 'Space'\n"
  	,CalibImage.z_type);
  fprintf(out,"Byte: CalibImage.z_unit: %d: Terminated by 'Space' + 'Newline'\n"
  	,CalibImage.z_unit);
  fprintf(out,"Float: CalibImage.x_cal[0]: %f: Terminated by 'Space'\n"
  	,CalibImage.x_cal[0]);
  fprintf(out,"Float: CalibImage.x_cal[1]: %f: Terminated by 'Space'\n"
  	,CalibImage.x_cal[1]);
  fprintf(out,"Float: CalibImage.x_cal[2]: %f: Terminated by 'Space'\n"
  	,CalibImage.x_cal[2]);
  fprintf(out,"Float: CalibImage.x_cal[3]: %f: Terminated by 'Newline'\n"
  	,CalibImage.x_cal[3]);
  fprintf(out,"Float: CalibImage.y_cal[0]: %f: Terminated by 'Space'\n"
  	,CalibImage.y_cal[0]);
  fprintf(out,"Float: CalibImage.y_cal[1]: %f: Terminated by 'Space'\n"
  	,CalibImage.y_cal[1]);
  fprintf(out,"Float: CalibImage.y_cal[2]: %f: Terminated by 'Space'\n"
  	,CalibImage.y_cal[2]);
  fprintf(out,"Float: CalibImage.y_cal[3]: %f: Terminated by 'Newline'\n"
  	,CalibImage.y_cal[3]);
  fprintf(out,"Float: CalibImage.z_cal[0]: %f: Terminated by 'Space'\n"
  	,CalibImage.z_cal[0]);
  fprintf(out,"Float: CalibImage.z_cal[1]: %f: Terminated by 'Space'\n"
  	,CalibImage.z_cal[1]);
  fprintf(out,"Float: CalibImage.z_cal[2]: %f: Terminated by 'Space'\n"
  	,CalibImage.z_cal[2]);
  fprintf(out,"Float: CalibImage.z_cal[3]: %f: Terminated by 'Newline'\n"
  	,CalibImage.z_cal[3]);
  fprintf(out,"Float: CalibImage.rayleigh_wavelength: %f: Terminated by 'Newline'\n"
  	,CalibImage.rayleigh_wavelength);
  fprintf(out,"Float: CalibImage.pixel_length: %f: Terminated by 'Newline'\n"
  	,CalibImage.pixel_length);
  fprintf(out,"Float: CalibImage.pixel_height: %f: Terminated by 'Newline'\n"
  	,CalibImage.pixel_height);
  fprintf(out,"len");
  fprintf(out,"char*: CalibImage.x_text: string of 'len' chars long = %s\n"
  	,x_text_buffer);
  fprintf(out,"len");
  fprintf(out,"char*: CalibImage.y_text: string of 'len' chars long = %s\n"
  	,y_text_buffer);
  fprintf(out,"len");
  fprintf(out,"char*: Calib_Image.z_text: string of 'len' chars long = %s\n"
  	,z_text_buffer);
}


/*****************************************************************************/
//Function: read_calibimage
//Inputs: none
//Outputs: none
//The purpose of this function is to read in the calibimage structure data
/*****************************************************************************/

void read_calibimage(void)

{
	char *calibimage_title;
  long version;
  int len;
  calibimage_title = (char*)malloc(MAXPATH);
  CalibImage.x_text = malloc(MAXPATH);
  CalibImage.y_text = malloc(MAXPATH);
  CalibImage.z_text = malloc(MAXPATH);
  version = read_int(' ');
  CalibImage.x_type = read_byte_and_skip_terminator();
  CalibImage.x_unit = read_byte_and_skip_terminator();
  CalibImage.y_type = read_byte_and_skip_terminator();
  CalibImage.y_unit = read_byte_and_skip_terminator();
  CalibImage.z_type = read_byte_and_skip_terminator();
  CalibImage.z_unit = read_byte_and_skip_terminator();
  CalibImage.x_cal[0] = read_float(' ');
  CalibImage.x_cal[1] = read_float(' ');
  CalibImage.x_cal[2] = read_float(' ');
  CalibImage.x_cal[3] = read_float('\n');
  CalibImage.y_cal[0] = read_float(' ');
  CalibImage.y_cal[1] = read_float(' ');
  CalibImage.y_cal[2] = read_float(' ');
  CalibImage.y_cal[3] = read_float('\n');
  CalibImage.z_cal[0] = read_float(' ');
  CalibImage.z_cal[1] = read_float(' ');
  CalibImage.z_cal[2] = read_float(' ');
  CalibImage.z_cal[3] = read_float('\n');

  if( (HIWORD(version) >= 1) && (LOWORD(version) >=3) ){
    CalibImage.rayleigh_wavelength = read_float('\n');
    CalibImage.pixel_length = read_float('\n');
    CalibImage.pixel_height = read_float('\n');
  }

  len = (int)read_int('\n');
  read_len_chars(len,calibimage_title);
  strcpy(CalibImage.x_text,calibimage_title);
  len = (int)read_int('\n');
  read_len_chars(len,calibimage_title);
  strcpy(CalibImage.y_text,calibimage_title);
  len = (int)read_int('\n');
  read_len_chars(len,calibimage_title);
  strcpy(CalibImage.z_text,calibimage_title);
  print_calib_image(version,CalibImage.x_text,CalibImage.y_text,CalibImage.z_text);
  free(calibimage_title);
  free(CalibImage.x_text);
  free(CalibImage.y_text);
  free(CalibImage.z_text);
} //End of TCalibImage

/*****************************************************************************/
//Function: print_image_structure
//Inputs: 2 buffers to store image and version as well as a long for version
//Outputs: none
//The purpose of this function is to print out the image structure data
/*****************************************************************************/

void print_image_structure(long pversion1,float *print_image_buff
														,long *version2_buffer)
{
  int j,k,i;
  if(!version2_buffer) printf("Cannot create image2_buffer");
  if(!print_image_buff) printf("Cannot create print_image_buff");
  fprintf(out,"int: version: %d: Terminated by 'Space'\n",pversion1);
  fprintf(out,"int: Image.image_format.left: %d: Terminated by 'Space'\n"
		,Image.image_format.left);
  fprintf(out,"int: Image.image_format.top: %d: Terminated by 'Space'\n"
  	,Image.image_format.top);
  fprintf(out,"int: Image.image_format.right: %d: Terminated by 'Space'\n"
  	,Image.image_format.right);
  fprintf(out,"int: Image.image_format.bottom: %d: Terminated by 'Space'\n"
  	,Image.image_format.bottom);
  fprintf(out,"int: Image.no_images: %d: Terminated by 'Space'\n"
  	,Image.no_images);
  fprintf(out,"int: Image.no_subimages: %d: Terminated by 'Space'\n"
  	,Image.no_subimages);
  fprintf(out,"int: Image.total_length: %d: Terminated by 'Space'\n"
  	,Image.total_length);
  fprintf(out,"int: Image.image_length: %d: Terminated by 'Newline'\n"
  	,Image.image_length);
  for(j=0;j<Image.no_subimages;j++){ //Repeat no_subimages times
		fprintf(out,"int: Image.position[%d].left: %d: Terminated by 'Space'\n"
  	,j,Image.position[j].left);
    fprintf(out,"int: Image.position[%d].top: %d: Terminated by 'Space'\n"
  	,j,Image.position[j].top);
    fprintf(out,"int: Image.position[%d].right: %d: Terminated by 'Space'\n"
  	,j,Image.position[j].right);
    fprintf(out,"int: Image.position[%d].bottom: %d: Terminated by 'Space'\n"
  	,j,Image.position[j].bottom);
    fprintf(out,"int: Image.position[%d].vertical_bin: %d:Terminated by 'Space'\n"
    	,j,Image.position[j].vertical_bin);
    fprintf(out,"int: Image.position[%d].horizontal_bin: %d: Terminated by 'Space'\n"
    	,j,Image.position[j].horizontal_bin);
    fprintf(out,"int: Image.subimage_offset[j]: %d: Terminated by 'Newline'\n"
  		,Image.subimage_offset[j]);
  } // End of for(j=0;j<no_subimages;j++)
    for(k=0;k<Image.no_images;k++){
    fprintf(out,"int: Image.time_stamps[k]: %d: Terminated by 'Newline'\n"
  	,Image.time_stamps[k]);
  }


  for(i = 0; i<(long)Image.total_length;i++)// Print out floating point values
  	fprintf(out,"The float value for data value no.%d is %f\n",i+1
    	,print_image_buff[i]);

}

/*****************************************************************************/
//Function: read_image_structure
//Inputs: none
//Outputs: none
//The purpose of this function is to read the image structure data
/*****************************************************************************/

void read_image_structure(void)

{
	long version1,*version2;
  float *image_buff;
  int j;
  int k;
  version1 = read_int(' ');
  Image.image_format.left = (int)read_int(' ');
  Image.image_format.top = (int)read_int(' ');
  Image.image_format.right = (int)read_int(' ');
  Image.image_format.bottom = (int)read_int(' ');
  Image.no_images = (int)read_int(' ');
  Image.no_subimages = (int)read_int(' ');
  Image.total_length = read_int(' ');
  Image.image_length = read_int('\n');
  version2 = (long*)malloc((sizeof(long))*Image.no_subimages);
  if (!version2) printf("Cannot create version2 buffer array ");
  Image.position = malloc((sizeof(int))*Image.no_subimages*6);
  if (!Image.position) printf("Cannot create Image.position buffer array ");
  Image.subimage_offset = malloc((sizeof(unsigned long))*Image.no_subimages);
  if (!Image.subimage_offset) printf("Cannot create Image.subimage_offset buffer array ");
  for(j=0;j<Image.no_subimages;j++){ //Repeat no_subimages times
  	version2[j] = read_int(' ');
    Image.position[j].left= (int)read_int(' ');
    Image.position[j].top = (int)read_int(' ');
    Image.position[j].right = (int)read_int(' ');
    Image.position[j].bottom = (int)read_int(' ');
    Image.position[j].vertical_bin = (int)read_int(' ');
    Image.position[j].horizontal_bin = (int)read_int(' ');
    Image.subimage_offset[j] = read_int('\n');
  } // End of for(j=0;j<no_subimages;j++)


  Image.time_stamps = malloc((sizeof(unsigned long))*Image.no_images);
  for(k=0;k<Image.no_images;k++){
  	Image.time_stamps[k] = read_int('\n');
  }
  image_buff = (float*)malloc(4*(size_t)Image.total_length);
  read_image(Image.total_length,image_buff);
  print_image_structure(version1,image_buff,version2);
  free(version2);
  free(image_buff);
  free(Image.subimage_offset);
  free(Image.time_stamps);
  free(Image.position);
 }//read_image_structure;


/*****************************************************************************/
//Function: read_all_data
//Inputs: The number of images
//Outputs: Error
//The purpose of this function is to make all the function calls
//for reading in the data
/*****************************************************************************/

int read_all_data(void)
{
	char ch, *title;
	long version,is_present;
  int p;
  title = (char*)malloc(MAXPATH);   // Allocate dynamic memory for the string
  read_string('\n',title);
  if (strcmp("Andor Technology Multi-Channel File",title) != 0 &&
      strcmp("Oriel Instruments Multi-Channel File",title) != 0) {
  	printf("This is not a proper SIF file: The file may be corrupt\n");
    printf("Press any key to continue");
  	getch();
    free(title);
    return 1;
    }
  fprintf(out,"char*: title: %s: Terminated by 'Newline'\n",title);
  version = read_int(' '); //Version
  fprintf(out,"unsigned int: version: %ld: Terminated by 'Space'\n",version);
  for(p=0;p<5;p++){
  	if(!feof(in)){
  		is_present = read_int('\n');
  		fprintf(out,"int: is_present: %ld: Terminated by 'Newline'\n"
    		,is_present);
    	// Following section only is repeated number_of_images times.
  		if(is_present==1){
  			read_instaimage();
  			read_calibimage();
  			read_image_structure();
 			}//End if(is_present =1)
  	}//if(!feof(in))
	}//End for(p=0;p<4;p++)
  fscanf(in,"%c",&ch);
  if(feof(in)) printf("File has now been successfully read\n");
  else printf("End of file has not been reached\n");
  printf("Press any key to continue");
  getch();
  free(title);
  return 0;
}// End of read_all_data


/*****************************************************************************/
//Function: main
//Inputs:
//Outputs:
//The Main function opens an '.SIF' file for reading and a text file for
//outputting the structure
/*****************************************************************************/

int main(/*int argc, char * argv[]*/)
{
	char *infilepath,*outfilepath;
  infilepath = malloc(100);
	printf("Please enter the complete path of the SIF file to be read\n\n");
  printf("For example 'C:\\andor\\image.sif'\n\n");
  gets(infilepath);
  if((in = fopen(infilepath, "rb"))== NULL){
		printf("Cannot open input file: Please check the file path\n");
    printf("Press any key to continue");
    getch();
    free(infilepath);
    return 1;
  }
  outfilepath = malloc(100);
	printf("Please enter the complete path for the output file\n\n");
  gets(outfilepath);
  if ((out = fopen(outfilepath, "wt"))== NULL){
  	printf( "Cannot open output file: Please check path\n");
    printf("Press any key to continue");
    getch();
    free(outfilepath);
    return 1;
  }
  read_all_data();
  fclose(in);
  fclose(out);
  free(infilepath);
  free(outfilepath);
  return 0;
}
