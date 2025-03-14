// config.h

#define BOX_NUMBER "52" //number of the box

#define SAMPLES 2048 //number of samples taken, needs to be a power of 2 i.e. 2^n

#define EPOCH_TIME 1741966595 //Unix time corresponding to when the BRACHIOSAURUS is turned on

#define LOOPS_PER_WAKE_UP 10 //how many data points are taken in a wake up loop

#define SEC_TO_SLEEP 21600; //time (seconds) the microcontroller goes to sleep between taking data

#define DEPLOYMENT_START 1754020860 //reasonable satrt time for when the data can be recovered, set to August 1rst

#define DEPLOYMENT_END 1755144060 //reasonable end time for when the data can be recovered, set to August 14th

#define DEPLOYMENT_WAIT 600 //max number of seconds to wait for wifi option

//set DEPLOYMENT_START and DEPLOYMENT_END to 0 if you do not plan on using the wifi option