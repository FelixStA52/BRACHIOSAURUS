#include "arduinoFFT.h" //import for FFT

#include <WiFi.h>
#include <WebServer.h>

// Set your desired AP credentials
const char* apSSID = "data";
const char* apPassword = "12345678";
WebServer server(80);
bool stopSharing = false;
bool sdCardAvailable = false;

#include <Wire.h> //import for I2C communication

#include <Adafruit_LSM6DSOX.h> //import to interface with accelerometer board

#include <time.h> //import for easier timekeeping

#include "FS.h" //import for SPI communication with SD card
#include "SD.h" //import for easier code to write to files
#include "SPI.h" //import for SPI communication with SD card

#include "config.h" // configuration file for user defined variables

// USER DEFINED VARIABLES //////////////////////////////////////////////////////////////////////////////////////////////////

const String box_n = BOX_NUMBER; //number of the box

const int samples = SAMPLES; //number of samples taken, needs to be a power of 2 i.e. 2^n

RTC_DATA_ATTR int epochTime = EPOCH_TIME; //Unix time corresponding to when the BRACHIOSAURUS is turned on

const int loopsPerWakeUp = LOOPS_PER_WAKE_UP; //how many data points are taken in a wake up loop

const long int secToSleep = SEC_TO_SLEEP; //time (seconds) the microcontroller goes to sleep between taking data

const long int wifiTimeSleep = DEPLOYMENT_WAIT;

const long int deploy_start = DEPLOYMENT_START;

const long int deploy_end = DEPLOYMENT_END;

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

String data_filename = "data_" + box_n + ".csv";
String motion_filename = "motion_" + box_n + ".bin";

#define TIME_TO_SLEEP  secToSleep

#define uS_TO_S_FACTOR 1000000ULL  //seconds into micro seconds

float samplingFrequency = 200.0; //affects the arduino FFT object but not the code

const int timesAboveNoise = 20; //threshold for data sampling (FFT amplitude)

Adafruit_LSM6DSOX sox; //accelerometer object

RTC_DATA_ATTR int counter = 0; //counter for times it went into deep sleep

RTC_DATA_ATTR int nextDataTime = 0;
RTC_DATA_ATTR int nextWiFiCheckTime = 0;
RTC_DATA_ATTR bool wifi_accessed = 0;

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

  if (nextDataTime == 0) {
    nextDataTime = epochTime + secToSleep;
    nextWiFiCheckTime = deploy_start;
  }

  if (epochTime >= nextDataTime) {
    power_on();
    main_routine();
    nextDataTime = epochTime + secToSleep;
  }

  if (! wifi_accessed && epochTime >= deploy_start && epochTime < deploy_end) {
    if (epochTime >= nextWiFiCheckTime) {
      performWiFiCheck();
      nextWiFiCheckTime = epochTime + wifiTimeSleep;
    }
  }

  int timeToData = nextDataTime - epochTime;
  int timeToWiFi = (epochTime >= deploy_start && epochTime < deploy_end) ? (nextWiFiCheckTime - epochTime) : (deploy_start - epochTime);
  int sleepTime = max(min(timeToData, timeToWiFi), 0);

  if (wifi_accessed){
    sleepTime = timeToData;
  }

  int realSleepTime = sleepTime * correctionFactor(box_n);
  realSleepTime = static_cast<int>(realSleepTime);

  Serial.begin(115200); //for testing
  while(!Serial);
  Serial.println(realSleepTime);

  codeEnd = millis();
  epochTime += (codeEnd - codeStart) / 1000 + sleepTime;

  esp_sleep_enable_timer_wakeup(realSleepTime * uS_TO_S_FACTOR);
  esp_deep_sleep_start();
}

void loop(){}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void power_on(){
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
  Serial.println("Serial active");
}

void main_routine(){
  Wire.begin(accSDA, accSCL); //begin communication with accelerometer

  struct tm *timeinfo;
  time_t current_time;
  current_time = epochTime;
  timeinfo = localtime(&current_time);
  char buffer[80];
  strftime(buffer, 80, "%Y%m%d", timeinfo);

  String buffer_str = buffer;
  Serial.println(buffer_str);

  data_filename = "/data_" + box_n + "_" + buffer_str + ".csv";
  motion_filename = "/motion_" + box_n + "_" + buffer_str + ".bin";

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

        SD.mkdir("/data_files_" + box_n);

        File file = SD.open("/data_files_" + box_n + data_filename, FILE_APPEND); //open file, must end in .csv

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

  // Serial.println("BRAVO VI, going dark");

  // codeEnd = millis(); //ends the code timer
  // epochTime = epochTime + ((codeEnd - codeStart)/1000) + secToSleep; //computes code time elapsed

  // Serial.println(epochTime);
}

void performWiFiCheck() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);
  int numNetworks = WiFi.scanNetworks();
  if (numNetworks > 0) {
    power_on();
    startAPDataSharing();
  }
}

float correctionFactor(String box_n){
    if (box_n == "2") return 1.009916f;
    if (box_n == "3") return 1.010026f;
    if (box_n == "4") return 1.009210f;
    if (box_n == "5") return 1.009423f;
    if (box_n == "6") return 1.008915f;
    if (box_n == "7") return 1.009478f;
    if (box_n == "8") return 1.009535f;
    if (box_n == "9") return 1.010207f;
    if (box_n == "10") return 1.009584f;
    if (box_n == "11") return 1.013200f;
    if (box_n == "12") return 1.009553f;
    if (box_n == "13") return 1.007473f;
    if (box_n == "14") return 1.008544f;
    if (box_n == "15") return 1.009852f;
    if (box_n == "16") return 1.009633f;
    if (box_n == "17") return 1.010457f;
    if (box_n == "18") return 1.020610f;
    return 1.009584f; // Median of all factors if box is not specified
}

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

      SD.mkdir("/motion_files_" + box_n);

      // Open or create a file to write
      File data = SD.open("/motion_files_" + box_n + motion_filename, FILE_APPEND);

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



// Wifi stuff #####################################################################################################################


// Function to list files and directories inside a given path
void listFiles(String path, String& html) {
  File dir = SD.open(path);
  if (!dir || !dir.isDirectory()) {
    html += "<li><strong>Error: Cannot open directory</strong></li>";
    return;
  }

  File file = dir.openNextFile();
  while (file) {
    String fileName = file.name();

    // Skip macOS hidden files (e.g., ._*, .DS_Store)
    if (fileName.startsWith("._") || fileName.startsWith(".")) {
      file.close();
      file = dir.openNextFile();
      continue; // Skip this file
    }

    // Normalize the path
    if (!fileName.startsWith("/")) {
      fileName = "/" + fileName;
    }
    fileName.replace("//", "/");

    html += "<li><a href=\"";
    if (file.isDirectory()) {
      html += "/list?path=" + fileName;
    } else {
      html += "/download?path=" + fileName;
    }
    html += "\">" + fileName + (file.isDirectory() ? " [DIR]" : "") + "</a></li>";

    file.close();
    file = dir.openNextFile();
  }
}

// Handler for root directory listing
void handleRoot() {
  String html = "<html><head><title>ESP32 Data Sharing</title></head><body>";
  html += "<h1>Files on SD Card</h1>";
  html += "<a href=\"/status\">View System Status</a><br>";
  
  if (!sdCardAvailable) {
    html += "<p style='color: red;'>Error: Unable to access SD card</p>";
  } else {
    html += "<ul>";
    listFiles("/", html);
    html += "</ul>";
  }
  
  html += "<form action=\"/stop\" method=\"POST\">";
  html += "<button type=\"submit\">Stop Sharing</button>";
  html += "</form>";
  html += "</body></html>";
  
  server.send(200, "text/html", html);
}

// Handler for listing files in a subdirectory
void handleListDir() {
  if (!sdCardAvailable) {
    server.send(500, "text/plain", "SD Card Not Available");
    return;
  }

  if (!server.hasArg("path")) {
    server.send(400, "text/plain", "Missing path parameter");
    return;
  }

  String path = server.arg("path");
  // Normalize path
  if (!path.startsWith("/")) path = "/" + path;
  path.replace("//", "/");

  String html = "<html><head><title>ESP32 Data Sharing</title>";
  html += "<style>body {font-family: sans-serif;} li {margin: 8px 0;} </style></head><body>";
  html += "<h1>Directory: " + path + "</h1>";
  html += "<div style='margin-bottom: 20px;'>";
  html += "<button onclick=\"downloadSelected()\">Download Selected</button>";
  html += "<button onclick=\"downloadAll()\" style='margin-left: 10px;'>Download All</button>";
  html += "</div><ul>";

  File dir = SD.open(path);
  if (!dir || !dir.isDirectory()) {
    html += "<li>Error opening directory</li>";
  } else {
    File file = dir.openNextFile();
    while (file) {
      String fileName = file.name();
      
      // Skip hidden files
      if (fileName.startsWith(".") || fileName.startsWith("_")) {
        file.close();
        file = dir.openNextFile();
        continue;
      }

      String fullPath = path + (path.endsWith("/") ? "" : "/") + fileName;
      fullPath.replace("//", "/");

      if (file.isDirectory()) {
        html += "<li>📁 <a href=\"/list?path=" + fullPath + "\">" + fileName + "</a></li>";
      } else {
        html += "<li><input type='checkbox' class='fileCheck' value='" + fullPath + "'> ";
        html += "<a href=\"/download?path=" + fullPath + "\">" + fileName + "</a>";
        html += " (" + String(file.size()) + " bytes)</li>";
      }
      
      file.close();
      file = dir.openNextFile();
    }
  }
  
  html += "</ul>";
  html += "<a href=\"" + (path.indexOf('/', 1) != -1 ? "/list?path=" + path.substring(0, path.lastIndexOf('/')) : "/") + "\">⬅ Back</a>";
  
  // JavaScript handling
  html += "<script>"
          "function downloadAll() {"
          "  document.querySelectorAll('.fileCheck').forEach(cb => cb.checked = true);"
          "  downloadSelected();"
          "}"
          "function downloadSelected() {"
          "  let checks = Array.from(document.querySelectorAll('.fileCheck:checked'));"
          "  if(checks.length === 0) return alert('No files selected!');"
          "  checks.forEach((cb, index) => {"
          "    setTimeout(() => {"
          "      let link = document.createElement('a');"
          "      link.href = '/download?path=' + encodeURIComponent(cb.value);"
          "      link.download = cb.value.split('/').pop();"
          "      document.body.appendChild(link);"
          "      link.click();"
          "      document.body.removeChild(link);"
          "    }, index * 1000);"  // 1-second interval between downloads
          "  });"
          "}"
          "</script>";
  
  html += "</body></html>";
  server.send(200, "text/html", html);
}

// Handler for downloading files
void handleDownload() {
  if (!sdCardAvailable) {
    server.send(500, "text/plain", "SD Card Not Available");
    return;
  }

  if (!server.hasArg("path")) {
    server.send(400, "text/plain", "Missing file path");
    return;
  }

  String path = server.arg("path");
  if (!path.startsWith("/")) {
    path = "/" + path;  // Ensure absolute path
  }

  File file = SD.open(path);
  if (!file || file.isDirectory()) {
    server.send(404, "text/plain", "File not found or is a directory");
    return;
  }

  // Extract filename from path
  String filename = path.substring(path.lastIndexOf('/') + 1);

  // Set headers to suggest the filename
  server.sendHeader("Content-Disposition", "attachment; filename=\"" + filename + "\"");
  
  server.streamFile(file, "application/octet-stream");
  file.close();
}

// Handler for stopping the access point
void handleStop() {
  server.send(200, "text/html", "<html><body><h1>Data Sharing Stopped</h1></body></html>");
  stopSharing = true;
  wifi_accessed = 1;
}

void handleStatus() {
  String html = "<html><head><title>ESP32 Status</title></head><body>";
  html += "<h1>System Status</h1>";

  // Wi-Fi Status
  html += "<h2>Wi-Fi Status</h2>";
  html += "<p>SSID: " + String(apSSID) + "</p>";
  html += "<p>IP Address: " + WiFi.softAPIP().toString() + "</p>";
  html += "<p>Connected Devices: " + String(WiFi.softAPgetStationNum()) + "</p>";

  // SD Card Status
  html += "<h2>SD Card Status</h2>";
  if (!sdCardAvailable) {
    html += "<p style='color: red;'>SD Card: Not Available</p>";
  } else {
    html += "<p style='color: green;'>SD Card: Connected</p>";

    // SD Card Storage Information
    uint64_t cardSize = SD.cardSize();
    uint64_t usedBytes = SD.usedBytes();
    uint64_t freeBytes = cardSize - usedBytes;

    html += "<p>Total Space: " + String(cardSize / 1e6) + " MB</p>";
    html += "<p>Used Space: " + String(usedBytes / 1e6) + " MB</p>";
    html += "<p>Free Space: " + String(freeBytes / 1e6) + " MB</p>";
  }

  // System Information
  html += "<h2>System Information</h2>";
  html += "<p>Box Number: " + box_n + "</p>";
  html += "<p>Current Epoch Time: " + String(epochTime) + "</p>";
  html += "<p>Wake-Up Counter: " + String(counter) + "</p>";

  // Add a link back to the root page
  html += "<br><a href=\"/\">Back to File List</a>";

  html += "</body></html>";

  server.send(200, "text/html", html);
}

// Function to start the access point and serve files
// void startAPDataSharing() {
//   Serial.println("Starting Access Point...");
//   WiFi.softAP(apSSID, apPassword);
//   Serial.print("AP IP address: ");
//   Serial.println(WiFi.softAPIP());

//   Serial.println("Initializing SD card...");
//   sdCardAvailable = SD.begin(CS_PIN); // Store initialization status
//   if (!sdCardAvailable) {
//     Serial.println("SD initialization failed!");
//     // DO NOT disconnect AP here - keep it active to show the error
//   } else {
//     Serial.println("SD card initialized successfully.");
//   }

//   server.on("/", handleRoot);
//   server.on("/list", handleListDir);
//   server.on("/download", handleDownload);
//   server.on("/status", handleStatus);
//   server.on("/stop", HTTP_POST, handleStop);

//   server.begin();
//   Serial.println("HTTP server started");

//   stopSharing = false;
//   unsigned long apStartTime = millis();
//   while (!stopSharing && (millis() - apStartTime < 1800000)) { //30 minutes timeout
//     server.handleClient();
//     delay(10);
//   }

//   server.stop();
//   WiFi.softAPdisconnect(true);
//   Serial.println("Access Point and server stopped");
// }

void startAPDataSharing() {
  Serial.println("Starting Access Point...");
  WiFi.softAP(apSSID, apPassword);
  Serial.print("AP IP address: ");
  Serial.println(WiFi.softAPIP());

  Serial.println("Initializing SD card...");
  sdCardAvailable = SD.begin(CS_PIN);
  if (!sdCardAvailable) {
    Serial.println("SD initialization failed!");
  } else {
    Serial.println("SD card initialized successfully.");
  }

  server.on("/", handleRoot);
  server.on("/list", handleListDir);
  server.on("/download", handleDownload);
  server.on("/status", handleStatus);
  server.on("/stop", HTTP_POST, handleStop);

  server.begin();
  Serial.println("HTTP server started");

  stopSharing = false;
  unsigned long apStartTime = millis();
  bool clientConnected = false;
  unsigned long connectionTime = 0;

  while (!stopSharing) {
    server.handleClient();
    delay(10);

    unsigned long currentTime = millis();

    // Check for clients if not already connected
    if (!clientConnected) {
      int clients = WiFi.softAPgetStationNum();
      if (clients > 0) {
        clientConnected = true;
        connectionTime = currentTime;
        Serial.println("Client connected. AP active for 30 minutes.");
      }
    }

    // Timeout conditions
    if (clientConnected) {
      if (currentTime - connectionTime >= 1800000UL) { // 30 minutes
        Serial.println("30 minutes elapsed. Stopping AP.");
        break;
      }
    } else {
      if (currentTime - apStartTime >= 300000UL) { // 5 minutes
        Serial.println("No clients in 5 minutes. Stopping AP.");
        break;
      }
    }
  }

  server.stop();
  WiFi.softAPdisconnect(true);
  Serial.println("Access Point and server stopped.");
}
