#include <WiFi.h>
#include <esp_sleep.h>

const char* ssid = "ESP32_Detector";  // SSID of the Wi-Fi network
const char* password = "123456789";   // Password (optional)
const int wifi_channel = 6;           // Wi-Fi channel to use

// Define the wake-up pin
#define WAKEUP_PIN GPIO_NUM_4  // Pin 4 (D12)

// Button pin (same as wake-up pin)
#define BUTTON_PIN WAKEUP_PIN

void setup() {
  Serial.begin(115200);

  // Setting up the Wi-Fi Access Point
  WiFi.softAP(ssid, password, wifi_channel);
  Serial.println("Wi-Fi Access Point created");
  Serial.print("SSID: ");
  Serial.println(ssid);
  Serial.print("IP address: ");
  Serial.println(WiFi.softAPIP());

  pinMode(26, OUTPUT);
  pinMode(25, OUTPUT);
  digitalWrite(26, HIGH);
  digitalWrite(25, LOW);
  
  // Configure the wake-up pin as an RTC wake-up source
  esp_sleep_enable_ext0_wakeup(WAKEUP_PIN, 1);  // Wake up when pin 4 is HIGH
  
  // Configure the button pin as input with an internal pull-down resistor
  pinMode(BUTTON_PIN, INPUT_PULLDOWN);

  // Check if the wake-up was caused by the external wake-up source
  if (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_EXT0) {
    Serial.println("Woke up from deep sleep due to RTC wake-up source");
  } else {
    Serial.println("ESP32 is awake");
  }
  delay(2000);
}

void loop() {
  // Check if the button is pressed (pin 4 goes HIGH)
  if (0) { // digitalRead(BUTTON_PIN) == HIGH
    Serial.println("Button pressed, going to deep sleep...");
    delay(1000);  // Debounce delay

    // Going to deep sleep
    Serial.flush();
    esp_deep_sleep_start();
  }

  // Main loop content
  delay(100);  // Reduce power consumption with a delay
}

