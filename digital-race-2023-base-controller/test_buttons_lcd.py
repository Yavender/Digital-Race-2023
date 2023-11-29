import RPi.GPIO as GPIO
import time

from platform_modules.lcd_driver import LCD
#from platform_modules.button_reader import ButtonReader
import time
import datetime
import config as cfg
import global_storage as gs

# Init LCD
lcd = LCD(cfg.LCD_ADDRESS)

# Select one button GPIO to test
test_button_pin = cfg.GPIO_BUTTON_1
# test_button_pin = cfg.GPIO_BUTTON_2
# test_button_pin = cfg.GPIO_BUTTON_3
# test_button_pin = cfg.GPIO_BUTTON_4

test_button_state = None

def main():
    high_st_count = 0
    low_st_count = -1

    prev_value = None

    # Pin Setup:
    GPIO.setmode(GPIO.BCM)  # BCM pin-numbering scheme from Raspberry Pi
    
    GPIO.setup(cfg.GPIO_BUTTON_1, GPIO.IN)  # set pin as an input pin
    GPIO.setup(cfg.GPIO_BUTTON_2, GPIO.IN)  # set pin as an input pin
    GPIO.setup(cfg.GPIO_BUTTON_3, GPIO.IN)  # set pin as an input pin
    GPIO.setup(cfg.GPIO_BUTTON_4, GPIO.IN)  # set pin as an input pin
    print("Starting demo now! Press CTRL+C to exit")
    try:
        while True:
            #lcd.lcd_clear()
            lcd.lcd_display_string("BUTTON - LCD TEST", 1)
            lcd.lcd_display_string("Button on GPIO: " + str(test_button_pin), 2)
                                   
            test_button_state = GPIO.input(test_button_pin)

            if test_button_state != prev_value:
                if test_button_state == GPIO.HIGH:
                    value_str = "HIGH"
                    high_st_count = high_st_count + 1
                    lcd.lcd_display_string("High Stage Count: " + str(high_st_count), 3)
                else:
                    value_str = "LOW"
                    
                    low_st_count = low_st_count + 1
                    lcd.lcd_display_string("Low Stage Count: " + str(low_st_count), 4)
                print("Value read from button - GPIO {} : {}".format(test_button_pin,
                                                           value_str))
                prev_value = test_button_state
            time.sleep(0.1)
    finally:
        GPIO.cleanup()

if __name__ == '__main__':
    main()
