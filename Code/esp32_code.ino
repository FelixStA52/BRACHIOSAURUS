#include "arduinoFFT.h" //import for FFT

#include <Wire.h> //import for I2C communication

#include <Adafruit_LSM6DSOX.h> //import to interface with accelerometer board

#include <time.h> //import for easier timekeeping

#include "FS.h" //import for SPI communication with SD card
#include "SD.h" //import for easier code to write to files
#include "SPI.h" //import for SPI communication with SD card

// USER DEFINED VARIABLES //////////////////////////////////////////////////////////////////////////////////////////////////

const char data_filename[] = "/data_10.csv";
const char motion_filename[] = "/motion_10.bin";

const int samples = 2048; //number of samples taken, needs to be a power of 2 i.e. 2^n

RTC_DATA_ATTR int epochTime = 1737664847; //1720798500; //Unix time corresponding to when the BRACHIOSAURUS is turned on

const int loopsPerWakeUp = 10; //how many data points are taken in a wake up loop

const long int secToSleep = 20;//21600; //time (seconds) the microcontroller goes to sleep between taking data

const int timesAboveNoise = 10; //threshold for data sampling (FFT amplitude)

// PINOUT //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define CS_PIN 13  //chip Select pin for the SD card, could be any GPIO

const int power1 = 2; //digital pins used to power the peripherals
const int power2 = 4;
const int power3 = 12;
const int power4 = 16;
const int power5 = 17;

const int trigPin = 25; //trigger pin on ultrasonic sensor, could be any GPIO 
const int echoPin = 26; //same thing but echo pin

const int accSDA = 21; //SDA SCL pins for accelerometer
const int accSCL = 22;

// OTHER VARIABLES ////////////////////////////////////////////////////////////////////////////////////////////////////////

#define TIME_TO_SLEEP  secToSleep

#define uS_TO_S_FACTOR 1000000ULL  //seconds into micro seconds

float samplingFrequency = 200.0; //affects the arduino FFT object but not the code

Adafruit_LSM6DSOX sox; //accelerometer object

RTC_DATA_ATTR int counter = 0; //counter for times it went into deep sleep

int16_t ax, ay, az; //acceleration in x y z
int16_t gx, gy, gz; //gyration in x y z
float rot_mag; //gx**2 + gy**2 + gz**2
float temperatureC; //temperature recorded on accelerometer

float vReal[samples]; //2048 array to store rot_mag
float vImag[samples]; //2048 to store nothing, necessary for FFT

double distance; //recorded distance of ultrasonic sensor
double accFreq; //recorded max frequency of the accelerometer

unsigned long codeStart; //time stamps to time the code
unsigned long codeEnd;

sensors_event_t accel; //acceleration object modified and returned by accelerometer
sensors_event_t gyro; //gyration object modified and returned by accelerometer
sensors_event_t temp; //temperature object modified and returned by accelerometer

double peaks[2]; //saves the 2 peaks from the frequency spectrum

ArduinoFFT<float> FFT = ArduinoFFT<float>(vReal, vImag, samples, samplingFrequency); //fft object

#define SCL_FREQUENCY 0x02 //communication with the accelerometer

void setup()
{
  codeStart = millis(); //start of code time stamp

  pinMode(power1, OUTPUT); //power all the peripherals
  pinMode(power2, OUTPUT);
  pinMode(power3, OUTPUT);
  pinMode(power4, OUTPUT);
  pinMode(power5, OUTPUT);
  pinMode(trigPin, OUTPUT);  
	pinMode(echoPin, INPUT);
  digitalWrite(power1, HIGH);
  digitalWrite(power2, HIGH);
  digitalWrite(power3, HIGH);
  digitalWrite(power4, HIGH);
  digitalWrite(power5, HIGH);
  
  delay(1000); //allows the peripherals to boot properly

  Serial.begin(115200); //for testing
  while(!Serial);
  Serial.println("");

  Wire.begin(accSDA, accSCL); //begin communication with accelerometer

  for (uint16_t i=0; i<loopsPerWakeUp; i++){ //iterations per wake up cycle
    
    if (sox.begin_I2C()) { //test I2C communication
      sox.setGyroRange(LSM6DS_GYRO_RANGE_125_DPS); //set the gyration to 125 degress per second
      sox.setGyroDataRate(LSM6DS_RATE_208_HZ);
      sox.getEvent(&accel, &gyro, &temp); //take a temperature measurement this one will likely be wrong
      temperatureC = temp.temperature;
      Serial.print("Temp:");
      Serial.println(temperatureC);

      accRoutine(peaks); //get the main frequencies from the accelerometer routine function
      Serial.print("peak 1:");
      Serial.print(peaks[0]);
      Serial.println(" Hz");
      Serial.print("peak 2:");
      Serial.print(peaks[1]);
      Serial.println(" Hz");
      
      sox.getEvent(&accel, &gyro, &temp); //take a good measurement of temprature
      temperatureC = temp.temperature;
      Serial.print("Temp:");
      Serial.println(temperatureC);
    }

    distance = ultraRoutine(); //get the distance measurement from the depth sensor
    Serial.print("Distance: ");
    Serial.print(distance);
    Serial.println(" cm");

    Serial.print("Iteration: ");
    Serial.println(i + 1);

    Serial.print("Counter: ");
    Serial.println(counter);


    if (SD.begin(CS_PIN)) { //ensures communication with the SD card
      uint8_t cardType = SD.cardType(); 
      if (cardType != CARD_NONE) {
        Serial.println("Initializing SD card..."); //ensures there is an SD card

        File file = SD.open(data_filename, FILE_APPEND); //open file, must end in .csv

        if (file) { //ensures the file was opened properly
          file.print(counter); //write the data
          file.print(",");
          file.print(i);
          file.print(",");
          file.print(epochTime);
          file.print(",");
          file.print(temperatureC);
          file.print(",");
          file.print(peaks[0]);
          file.print(",");
          file.print(peaks[1]);
          file.print(",");
          if (distance < 550){
            file.print(distance);
          }
          file.println();
          

          Serial.println("File written successfully");
        }
        file.close();
      }
    }

    Serial.println("######################################");

  }
  //we are now outside of the main loop
  counter++; //increments the wake up counter

  Serial.println("BRAVO VI, going dark");

  codeEnd = millis(); //ends the code timer
  epochTime = epochTime + ((codeEnd - codeStart)/1000) + secToSleep; //computes code time elapsed

  Serial.println(epochTime);

  esp_sleep_enable_timer_wakeup(TIME_TO_SLEEP * uS_TO_S_FACTOR); //sets sleep time for esp32
  Serial.println("Going to sleep now");
  esp_deep_sleep_start(); //starts the deep sleep
  //ESP.deepSleep(secToSleep*conversionFactor);
}

void loop(){}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double ultraRoutine(){
  
  //delay(500);

  digitalWrite(trigPin, LOW); //signal sent to trigger the depth sensor
	delayMicroseconds(2);  
	digitalWrite(trigPin, HIGH);  
	delayMicroseconds(10);  
	digitalWrite(trigPin, LOW);

  float duration = pulseIn(echoPin, HIGH); //wait for the echo to come back
  double distance = (duration*.0343)/2;  //computation of the distance based on the time elapsed

  return distance;
}

void accRoutine(double *peaks){

  for (uint16_t i = 0; i < 30; i++){ //takes bogus data, the first few points (10-20) of data are overshot so taking bogus data ensures that the real data is all viable
    sox.getEvent(&accel, &gyro, &temp);
  }

  float t1 = micros(); //times the sampling
  
  for (uint16_t i = 0; i < samples; i++){ //takes all the required samples for one measurement of the frequency of oscillation
    sox.getEvent(&accel, &gyro, &temp);
    rot_mag = pow(gyro.gyro.x*100,2)+pow(gyro.gyro.y*100,2)+pow(gyro.gyro.z*100,2);
    vReal[i] = rot_mag;
    vImag[i] = 0;
    delay(4); //this time delay was found to give a sampling rate of 200Hz
  }

  float t2 = micros();

  if (SD.begin(CS_PIN)) { //for some boards, keeps all the acceleration datapoints in a separate file (useful for testing)
    uint8_t cardType = SD.cardType();
    if (cardType != CARD_NONE) {
      Serial.println("Initializing SD card...");

      // Open or create a file to write
      File data = SD.open(motion_filename, FILE_APPEND);

        // Check if the file was successfully opened
      if (data) {

        data.write((uint8_t*)&epochTime, sizeof(epochTime));

        for (int k=0; k<samples; k++){
          data.write((uint8_t*)&vReal[k], sizeof(vReal[k]));
        }

        Serial.println("Data written successfully");
      }
      data.close();
    }
  }

  float diff = t2-t1;
  Serial.print("Diff: ");
  Serial.println(diff);
  float avg_freq = samples * 1000000.0 / diff; //computes the average sampling frequency
  Serial.print("Avg freq: ");
  Serial.print(avg_freq);
  Serial.println(" Hz");
  samplingFrequency = avg_freq; //adjusts the sampling frequency for the FFT function
  FFT.compute(FFTDirection::Forward); //computes the FFT
  FFT.complexToMagnitude();
  findMax(vReal, (samples >> 1), peaks); //finds the maximum frequency of oscillation
}

void findMax(float *vData, uint16_t bufferSize, double *peaks){
  int first_max_index = 0;
  for (uint16_t k = 0; k < 2; k++){ 
    int maxIndex = 0;
    float maxMag = 0;
    float magSum = 0;
    float currentData;
    float averageMag;
    float maxFreq;
    for (uint16_t i = 4; i < bufferSize - 1; i++){ //iterates to find the maximal amplitude
      currentData = vData[i];
      if (currentData > maxMag && pow(i-first_max_index,2) > 25 && currentData > vData[i+1] && currentData > vData[i-1]){
        maxMag = currentData;
        maxIndex = i;
      }
    }
    int n = 0;
    for (uint16_t i = 0; i < bufferSize; i++){ //sums all the amplitudes that are not close to the maximal amplitude
      if ((i - maxIndex > 5) || (i - maxIndex < -5)){
        magSum = magSum + (vData[i]/1000000);
        n++;
      }
    }
    averageMag = (magSum/n)*1000000; //averages the amplitudes outside of the maximal amplitude this is the noise
    Serial.print("Max amp:");
    Serial.println(maxMag);
    Serial.print("Threshold:");
    Serial.println(timesAboveNoise*averageMag);
    Serial.print("Ratio:");
    Serial.println(maxMag/(averageMag));
    if (maxMag < timesAboveNoise*averageMag){ //makes sure that the maxinal amplitude is above a certain noise threshold
      maxIndex = 0;
    }
    first_max_index = maxIndex;
    maxFreq = ((maxIndex * 1.0 * samplingFrequency) / samples);
    peaks[k] = maxFreq;
  }
}
