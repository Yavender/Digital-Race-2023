from platform_modules.lcd_driver import LCD
import time
import datetime
import config

lcd = LCD(config.LCD_ADDRESS)

while True:
    lcd.lcd_clear()
    lcd.lcd_display_string("SamChopVaTocDo", 1)
    lcd.lcd_display_string("From BEE IT FPL Hà Nội", 2)
    lcd.lcd_display_string("Powered by Yavender Studio", 3)
    lcd.lcd_display_string(str(datetime.datetime.now().time()), 4)
    time.sleep(10)